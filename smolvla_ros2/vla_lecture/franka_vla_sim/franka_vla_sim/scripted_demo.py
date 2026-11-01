"""【教員用】スクリプトで自動的にキューブを押すデモを作り，テレオペと同じ形式でrosbagに記録する．

    ros2 run franka_vla_sim scripted_demo --ros-args -p n_episodes:=50

配布用モデルの学習データ作成や，パイプラインの動作確認に使う．
成功したエピソードだけを保存する．
"""
import random

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float64MultiArray, String

from franka_vla_sim.common import EpisodeRecorder, interpolate, is_success, load_task_config, set_cube_pose
from franka_vla_sim.kinematics import KinematicChain

PUSH_Z = 0.03          # 押すときの手先の高さ [m]
BEHIND = 0.07          # キューブ中心から手先までの距離（押し始め）[m]


class ScriptedDemo(Node):

    def __init__(self):
        super().__init__('scripted_demo')
        self.declare_parameter('config', '')
        self.declare_parameter('n_episodes', 50)
        self.declare_parameter('record', True)
        self.cfg = load_task_config(self.get_parameter('config').value)
        self.n_episodes = self.get_parameter('n_episodes').value
        self.record = self.get_parameter('record').value
        t = self.cfg['topics']
        self.fps = self.cfg['fps']
        self.home_q = np.array(self.cfg['home_q'])
        self.goal = np.array(self.cfg['goal_center'])

        self.arm_pub = self.create_publisher(Float64MultiArray, t['arm_command'], 10)
        self.hand_pub = self.create_publisher(Float64MultiArray, t['hand_command'], 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self.on_description, qos)
        self.create_subscription(Pose, t['cube_pose'], self.on_cube, 10)
        self.recorder = EpisodeRecorder(self.cfg)

        self.chain = None
        self.cube = None
        self.q_cmd = self.home_q.copy()
        self.p_cmd = None
        self.R_down = None
        self.phase = 'idle'
        self.phase_ticks = 0
        self.waypoints = []
        self.u = None          # 押す方向（単位ベクトル）
        self.home_traj = []
        self.episode = 0
        self.n_saved = 0
        self.create_timer(1.0 / self.fps, self.on_timer)

    def on_description(self, msg):
        self.chain = KinematicChain(msg.data, self.cfg['arm_joints'], 'world', 'fr3_hand_tcp')
        self.R_down = self.chain.fk(self.home_q)[:3, :3]

    def on_cube(self, msg):
        self.cube = np.array([msg.position.x, msg.position.y])

    def set_phase(self, name):
        self.phase = name
        self.phase_ticks = 0

    def move_towards(self, target, speed):
        """手先を target へ speed [m/s] で近づけ，到達したら True．"""
        d = np.asarray(target) - self.p_cmd
        n = np.linalg.norm(d)
        step = speed / self.fps
        self.p_cmd = self.p_cmd + (d if n <= step else d * step / n)
        self.q_cmd = self.chain.ik(self.q_cmd, self.p_cmd, self.R_down, q_rest=self.home_q)
        return n <= step

    def on_timer(self):
        if self.chain is None or self.cube is None:
            return
        self.phase_ticks += 1

        if self.phase == 'idle':
            if self.episode >= self.n_episodes:
                self.get_logger().info(f'終了: {self.n_saved}/{self.n_episodes} エピソードを保存しました')
                raise SystemExit
            rx, ry = self.cfg['cube_spawn_range']['x'], self.cfg['cube_spawn_range']['y']
            set_cube_pose(self.cfg, random.uniform(*rx), random.uniform(*ry))
            self.home_traj = interpolate(self.q_cmd, self.home_q, int(2.0 * self.fps))
            self.set_phase('go_home')

        elif self.phase == 'go_home':
            if self.home_traj:
                self.q_cmd = self.home_traj.pop(0)
            else:
                self.p_cmd = self.chain.fk(self.q_cmd)[:3, 3].copy()
                if self.record:
                    self.recorder.start()
                self.set_phase('wait_rec')

        elif self.phase == 'wait_rec':
            # ros2 bag record の起動待ち（この間の記録は「静止」のデータになる）
            if self.phase_ticks >= int(1.5 * self.fps):
                self.u = self.goal - self.cube
                self.u /= np.linalg.norm(self.u)
                noise = np.random.uniform(-0.01, 0.01, 2)
                start = self.cube - self.u * BEHIND + noise
                self.waypoints = [(np.r_[start, 0.10 + random.uniform(0, 0.04)], 0.15),
                                  (np.r_[start, PUSH_Z], 0.10)]
                self.set_phase('approach')

        elif self.phase == 'approach':
            target, speed = self.waypoints[0]
            if self.move_towards(target, speed):
                self.waypoints.pop(0)
                if not self.waypoints:
                    self.set_phase('push')

        elif self.phase == 'push':
            # キューブの横ずれを少し補正しながら，ゴール方向へまっすぐ押す
            lateral = np.array([-self.u[1], self.u[0]])
            offset = np.dot(self.cube - self.p_cmd[:2], lateral)
            target = self.p_cmd[:2] + self.u * 0.05 + lateral * np.clip(offset, -0.01, 0.01)
            self.move_towards(np.r_[target, PUSH_Z], 0.06)
            remaining = np.dot(self.goal - self.cube, self.u)  # ゴール中心までの残り距離（押す方向）
            if remaining < 0.005 or self.phase_ticks > 20 * self.fps:
                back = self.p_cmd[:2] - self.u * 0.04
                self.waypoints = [(np.r_[back, PUSH_Z], 0.08), (np.r_[back, 0.15], 0.12)]
                self.set_phase('retreat')

        elif self.phase == 'retreat':
            target, speed = self.waypoints[0]
            if self.move_towards(target, speed) or self.phase_ticks > 10 * self.fps:
                self.waypoints.pop(0)
                if not self.waypoints:
                    self.set_phase('finish')

        elif self.phase == 'finish':
            if self.phase_ticks >= int(0.5 * self.fps):
                ok = is_success(self.cfg, self.cube)
                if self.record:
                    path = self.recorder.stop(keep=ok)
                    self.n_saved += int(ok)
                    self.get_logger().info(
                        f'[{self.episode + 1}/{self.n_episodes}] {"成功→保存" if ok else "失敗→破棄"}: {path}')
                else:
                    self.get_logger().info(f'[{self.episode + 1}/{self.n_episodes}] {"成功" if ok else "失敗"}')
                self.episode += 1
                self.set_phase('idle')

        self.arm_pub.publish(Float64MultiArray(data=[float(v) for v in self.q_cmd]))
        self.hand_pub.publish(Float64MultiArray(data=[float(v) for v in self.cfg['hand_closed']]))


def main():
    rclpy.init()
    node = ScriptedDemo()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node.recorder.recording:
            node.recorder.stop(keep=False)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
