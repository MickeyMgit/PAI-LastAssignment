#!/usr/bin/env python3
"""teleop_keyboard の動作確認：キー入力を自動で送り，人がテレオペするのと同じ操作をする．

    （シミュレータを起動した状態で）
    python3 teleop_key_driver.py [--episodes 2] [--out /tmp/teleop_test]

1エピソードの操作：
  N でリセット・記録開始 → W/S・D で手先をキューブの横（ゴールと反対側）へ → F で下げる
  → A でキューブを横に滑らせてゴールへ → D・R で離れる → M で保存
最後に X（破棄）と H（記録なしリセット）の動作も確認する．
"""
import argparse
import os
import pty
import select
import subprocess
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, JointState

sys.path.insert(0, '/ws/src/franka_vla_sim')
from franka_vla_sim.common import is_success, load_task_config  # noqa: E402
from franka_vla_sim.kinematics import KinematicChain  # noqa: E402

BESIDE = 0.065   # キューブ中心から手先までの横方向の距離（押し始め）[m]


class Monitor(Node):
    def __init__(self, cfg):
        super().__init__('teleop_key_driver', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.cfg = cfg
        self.cube = None
        self.q = None
        self.image = None
        self.create_subscription(Pose, cfg['topics']['cube_pose'], self.on_cube, 10)
        self.create_subscription(JointState, cfg['topics']['joint_states'], self.on_js, 10)
        self.create_subscription(Image, cfg['topics']['camera_front'], lambda m: setattr(self, 'image', m), 1)

    def on_cube(self, m):
        self.cube = np.array([m.position.x, m.position.y])

    def on_js(self, m):
        pos = dict(zip(m.name, m.position))
        if all(j in pos for j in self.cfg['arm_joints']):
            self.q = np.array([pos[j] for j in self.cfg['arm_joints']])


class Driver:
    def __init__(self, out_dir):
        self.cfg = load_task_config('/ws/src/franka_vla_sim/config/task.yaml')
        urdf = subprocess.run(['xacro', '/ws/src/franka_vla_sim/urdf/fr3_vla.urdf.xacro'],
                              capture_output=True, text=True, check=True).stdout
        self.chain = KinematicChain(urdf, self.cfg['arm_joints'], 'world', 'fr3_hand_tcp')
        self.home_tcp = self.chain.fk(np.array(self.cfg['home_q']))[:3, 3]
        self.bag_root = os.path.expanduser(self.cfg['teleop']['bag_root'])
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.log = open(os.path.join(out_dir, 'teleop_stdout.log'), 'wb')
        self.mon = Monitor(self.cfg)
        self.target = self.home_tcp.copy()

        # teleop_keyboard を疑似端末（pty）の中で起動する＝人がターミナルで操作するのと同じ状態
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.execvp('bash', ['bash', '-c', 'source /ws/install/setup.bash && '
                               'exec ros2 run franka_vla_sim teleop_keyboard --ros-args -p use_sim_time:=true'])
        self.wait(8.0)

    # ---------- 道具 ----------
    def pump(self, sec):
        """sec 秒のあいだ，ROS のコールバックと teleop の出力を処理する（壁時計）．"""
        t_end = time.time() + sec
        while time.time() < t_end:
            rclpy.spin_once(self.mon, timeout_sec=0.01)
            if select.select([self.fd], [], [], 0)[0]:
                try:
                    self.log.write(os.read(self.fd, 4096))
                    self.log.flush()
                except OSError:
                    return

    wait = pump

    def tcp(self):
        return self.chain.fk(self.mon.q)[:3, 3] if self.mon.q is not None else None

    def key(self, k, n=1, interval=0.12):
        ws = self.cfg['teleop']['workspace']
        d = {'w': (1, 0, 0), 's': (-1, 0, 0), 'a': (0, 1, 0), 'd': (0, -1, 0), 'r': (0, 0, 1), 'f': (0, 0, -1)}
        for _ in range(n):
            os.write(self.fd, k.encode())
            if k in d:
                self.target = np.clip(self.target + self.cfg['teleop']['step'] * np.array(d[k]),
                                      [ws['x'][0], ws['y'][0], ws['z'][0]], [ws['x'][1], ws['y'][1], ws['z'][1]])
            self.pump(interval)

    def settle(self, tol=0.008, timeout=20.0):
        """手先が目標位置に追いつくまで待つ．"""
        t_end = time.time() + timeout
        while time.time() < t_end:
            p = self.tcp()
            if p is not None and np.linalg.norm(p - self.target) < tol:
                return True
            self.pump(0.1)
        return False

    def move_to(self, x=None, y=None, z=None):
        for axis, val, plus, minus in ((0, x, 'w', 's'), (1, y, 'a', 'd'), (2, z, 'r', 'f')):
            if val is None:
                continue
            n = int(round((val - self.target[axis]) / self.cfg['teleop']['step']))
            if n:
                self.key(plus if n > 0 else minus, abs(n))
        return self.settle()

    def episodes(self):
        return sorted(d for d in os.listdir(self.bag_root) if d.startswith('ep_')) if os.path.isdir(self.bag_root) else []

    # ---------- 1エピソード ----------
    def run_episode(self, i):
        before = set(self.episodes())
        self.key('n')
        self.target = self.home_tcp.copy()
        self.pump(6.0)  # 初期姿勢へ戻る（2秒）＋記録開始待ち
        new = sorted(set(self.episodes()) - before)
        cx, cy = self.mon.cube
        print(f'[ep{i}] 記録開始: {new}  キューブ=({cx:.3f}, {cy:.3f})', flush=True)

        ok1 = self.move_to(x=cx, y=cy - BESIDE)     # キューブの横（-y 側）へ
        ok2 = self.move_to(z=0.03)                  # 下げる
        print(f'[ep{i}] 横に移動: {ok1}, 下げる: {ok2}, 手先={np.round(self.tcp(), 3)}', flush=True)

        # A（+y）でキューブを横に滑らせる．手先が目標に追いついてから次のキーを押す
        goal_y = self.cfg['goal_center'][1]
        t0 = time.time()
        while self.mon.cube[1] < goal_y - 0.005 and time.time() - t0 < 90:
            self.key('a')
            self.settle(tol=0.012, timeout=3.0)
        print(f'[ep{i}] 押し終わり: キューブ={np.round(self.mon.cube, 3)} ({time.time() - t0:.0f}s)', flush=True)

        self.key('d', 4)
        self.key('r', 12)
        self.settle()
        success = is_success(self.cfg, self.mon.cube)
        self.key('m')
        self.pump(4.0)
        saved = sorted(set(self.episodes()) - before)
        print(f'[ep{i}] M で保存: {saved}  成功={success}', flush=True)
        return saved, success

    def check_discard_and_reset(self):
        before = set(self.episodes())
        self.key('n')
        self.pump(6.0)
        during = set(self.episodes()) - before
        self.key('x')
        self.pump(4.0)
        after = set(self.episodes()) - before
        print(f'[X] 記録中に作られた: {sorted(during)} → 破棄後に残った: {sorted(after)}', flush=True)
        cube0 = self.mon.cube.copy()
        self.key('h')
        self.pump(5.0)
        print(f'[H] キューブ {np.round(cube0, 3)} → {np.round(self.mon.cube, 3)}（記録なしでリセット）', flush=True)
        return bool(during) and not after

    def save_image(self, name):
        import cv2
        from cv_bridge import CvBridge
        if self.mon.image is not None:
            img = CvBridge().imgmsg_to_cv2(self.mon.image, 'bgr8')
            cv2.imwrite(os.path.join(self.out_dir, name), img)

    def quit(self):
        self.key('q')
        self.pump(3.0)
        try:
            os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=2)
    ap.add_argument('--out', default='/tmp/teleop_test')
    args = ap.parse_args()
    rclpy.init()
    d = Driver(args.out)
    results = []
    for i in range(args.episodes):
        results.append(d.run_episode(i))
        d.save_image(f'teleop_ep{i}_end.png')
    discard_ok = d.check_discard_and_reset()
    d.quit()
    print('=' * 50)
    for i, (saved, success) in enumerate(results):
        print(f'エピソード{i}: 保存={saved} 成功={success}')
    print(f'X（破棄）: {"OK" if discard_ok else "NG"}')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
