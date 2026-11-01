"""キーボードでFR3の手先を動かし，エピソードごとにrosbagへ記録するノード．

    ros2 run franka_vla_sim teleop_keyboard

手先の目標位置をキーで動かし，逆運動学で関節角に変換して /arm_controller/commands に送る．
送った関節角がそのまま学習データの action になる．
"""
import random
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float64MultiArray, String

from franka_vla_sim.common import EpisodeRecorder, interpolate, load_task_config, set_cube_pose
from franka_vla_sim.kinematics import KinematicChain

HELP = """
===== FR3 手先テレオペ =====
  W / S : 前 / 後   (+x / -x)
  A / D : 左 / 右   (+y / -y)
  R / F : 上 / 下   (+z / -z)

  N : 新しいエピソード（キューブと姿勢をリセットして記録開始）
  M : 記録を終了して保存
  X : 記録を中止して破棄
  H : 記録せずにリセット（練習用）
  Q : 終了
============================
"""

KEY_TO_DIR = {
    'w': (1, 0, 0), 's': (-1, 0, 0),
    'a': (0, 1, 0), 'd': (0, -1, 0),
    'r': (0, 0, 1), 'f': (0, 0, -1),
}


class TeleopKeyboard(Node):

    def __init__(self):
        super().__init__('teleop_keyboard')
        self.declare_parameter('config', '')
        self.cfg = load_task_config(self.get_parameter('config').value)
        t = self.cfg['topics']
        self.fps = self.cfg['fps']
        self.home_q = np.array(self.cfg['home_q'])

        self.arm_pub = self.create_publisher(Float64MultiArray, t['arm_command'], 10)
        self.hand_pub = self.create_publisher(Float64MultiArray, t['hand_command'], 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self.on_description, qos)

        self.chain = None
        self.q_cmd = self.home_q.copy()      # 最後に送った関節角
        self.p_target = None                 # 手先の目標位置
        self.p_cmd = None                    # 実際に指令している手先位置（目標へ速度制限つきで近づく）
        self.R_target = None                 # 手先の姿勢（下向きで固定）
        self.home_traj = []                  # 初期姿勢へ戻る途中の関節角の列
        self.lock = threading.Lock()

        self.recorder = EpisodeRecorder(self.cfg)
        self.record_after_reset = False

        self.create_timer(1.0 / self.fps, self.on_timer)
        self.running = True
        threading.Thread(target=self.keyboard_loop, daemon=True).start()
        print(HELP)

    # ---------- ROS コールバック ----------
    def on_description(self, msg):
        self.chain = KinematicChain(msg.data, self.cfg['arm_joints'], 'world', 'fr3_hand_tcp')
        T = self.chain.fk(self.home_q)
        with self.lock:
            self.p_target = T[:3, 3].copy()
            self.p_cmd = T[:3, 3].copy()
            self.R_target = T[:3, :3].copy()
        self.get_logger().info(f'初期姿勢の手先位置: {np.round(T[:3, 3], 3)}')

    def on_timer(self):
        if self.chain is None:
            return
        with self.lock:
            if self.home_traj:
                self.q_cmd = self.home_traj.pop(0)
                if not self.home_traj:
                    self.on_reset_done()
            else:
                # 目標位置へ最大速度つきで近づけ，逆運動学で関節角にする
                max_step = self.cfg['teleop']['max_speed'] / self.fps
                d = self.p_target - self.p_cmd
                n = np.linalg.norm(d)
                if n > max_step:
                    d *= max_step / n
                self.p_cmd = self.p_cmd + d
                self.q_cmd = self.chain.ik(self.q_cmd, self.p_cmd, self.R_target, q_rest=self.home_q)
            q = self.q_cmd
        self.arm_pub.publish(Float64MultiArray(data=[float(v) for v in q]))
        self.hand_pub.publish(Float64MultiArray(data=[float(v) for v in self.cfg['hand_closed']]))

    # ---------- リセット・記録 ----------
    def reset_scene(self, record: bool):
        rx, ry = self.cfg['cube_spawn_range']['x'], self.cfg['cube_spawn_range']['y']
        x, y = random.uniform(*rx), random.uniform(*ry)
        if not set_cube_pose(self.cfg, x, y):
            print('[WARN] キューブの配置に失敗しました（Gazeboは起動していますか？）')
        with self.lock:
            self.home_traj = interpolate(self.q_cmd, self.home_q, int(2.0 * self.fps))
            T = self.chain.fk(self.home_q)
            self.p_target, self.p_cmd = T[:3, 3].copy(), T[:3, 3].copy()
            self.record_after_reset = record
        print(f'[INFO] リセット中...（キューブ: x={x:.2f}, y={y:.2f}）')

    def on_reset_done(self):
        if self.record_after_reset:
            self.record_after_reset = False
            self.start_recording()
        else:
            print('[INFO] リセット完了')

    def start_recording(self):
        path = self.recorder.start()
        print(f'[REC] 記録開始: {path}  → キューブをゴールまで押したら M で保存')

    def stop_recording(self, keep: bool):
        if not self.recorder.recording:
            print('[INFO] 記録していません（N で開始）')
            return
        path = self.recorder.stop(keep)
        print(f'[REC] {"保存" if keep else "破棄"}しました: {path}')

    # ---------- キーボード ----------
    def keyboard_loop(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            while self.running and rclpy.ok():
                if not select.select([sys.stdin], [], [], 0.05)[0]:
                    continue
                key = sys.stdin.read(1).lower()
                if self.chain is None:
                    print('[INFO] /robot_description 待ち...')
                    continue
                if key in KEY_TO_DIR:
                    ws = self.cfg['teleop']['workspace']
                    with self.lock:
                        p = self.p_target + self.cfg['teleop']['step'] * np.array(KEY_TO_DIR[key])
                        self.p_target = np.clip(p, [ws['x'][0], ws['y'][0], ws['z'][0]],
                                                [ws['x'][1], ws['y'][1], ws['z'][1]])
                elif key == 'n':
                    if self.recorder.recording:
                        print('[INFO] 記録中です．先に M（保存）か X（破棄）を押してください')
                    else:
                        self.reset_scene(record=True)
                elif key == 'm':
                    self.stop_recording(keep=True)
                elif key == 'x':
                    self.stop_recording(keep=False)
                elif key == 'h':
                    if not self.recorder.recording:
                        self.reset_scene(record=False)
                elif key == 'q':
                    self.running = False
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            if self.recorder.recording:
                self.stop_recording(keep=True)


def main():
    rclpy.init()
    node = TeleopKeyboard()
    try:
        while rclpy.ok() and node.running:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        time.sleep(0.2)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
