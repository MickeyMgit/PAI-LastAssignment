"""評価ノード：決められたキューブ初期位置で方策を試行し，成功率を出す．

    ros2 run franka_vla_sim evaluator --ros-args -p n_trials:=5 -p label:=distributed_model

各試行：キューブを配置 → 初期姿勢へ → /vla/enable → 成功 or 制限時間まで待つ → 停止
結果は ~/vla_eval/<label>_<日時>.csv に保存される．
"""
import csv
import datetime
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger

from franka_vla_sim.common import is_success, load_task_config, set_cube_pose


class Evaluator(Node):

    def __init__(self):
        super().__init__('evaluator')
        self.declare_parameter('config', '')
        self.declare_parameter('n_trials', 5)
        self.declare_parameter('label', 'policy')
        self.declare_parameter('out_dir', '~/vla_eval')
        self.cfg = load_task_config(self.get_parameter('config').value)
        self.n_trials = self.get_parameter('n_trials').value
        self.label = self.get_parameter('label').value
        self.out_dir = os.path.expanduser(self.get_parameter('out_dir').value)

        self.cube_xy = None
        self.create_subscription(Pose, self.cfg['topics']['cube_pose'], self.on_cube, 10)
        self.enable_cli = self.create_client(SetBool, '/vla/enable')
        self.home_cli = self.create_client(Trigger, '/vla/go_home')
        self.done = False

    def on_cube(self, msg):
        self.cube_xy = np.array([msg.position.x, msg.position.y])

    # ---------- シミュレーション時間での待機・サービス呼び出し ----------
    def sim_time(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def sleep_sim(self, sec):
        t_end = self.sim_time() + sec
        while rclpy.ok() and self.sim_time() < t_end:
            time.sleep(0.02)

    def call(self, client, req):
        if not client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f'{client.srv_name} が見つかりません．action_executor は起動していますか？')
        event = threading.Event()
        future = client.call_async(req)
        future.add_done_callback(lambda _: event.set())
        event.wait(10.0)
        return future.result()

    def set_enabled(self, flag):
        req = SetBool.Request()
        req.data = flag
        self.call(self.enable_cli, req)

    # ---------- 評価ループ ----------
    def run(self):
        positions = self.cfg['evaluation']['cube_positions'][:self.n_trials]
        timeout = self.cfg['evaluation']['timeout']
        results = []
        for i, (x, y) in enumerate(positions):
            self.set_enabled(False)
            self.call(self.home_cli, Trigger.Request())
            self.sleep_sim(2.5)
            set_cube_pose(self.cfg, x, y)
            self.sleep_sim(1.0)
            self.get_logger().info(f'[{i + 1}/{len(positions)}] 開始 (キューブ: x={x:.2f}, y={y:.2f})')
            self.set_enabled(True)
            t0 = self.sim_time()
            success = False
            while self.sim_time() - t0 < timeout:
                if self.cube_xy is not None and is_success(self.cfg, self.cube_xy):
                    success = True
                    break
                self.sleep_sim(0.2)
            elapsed = self.sim_time() - t0
            self.set_enabled(False)
            final = self.cube_xy if self.cube_xy is not None else [np.nan, np.nan]
            dist = float(np.linalg.norm(np.asarray(final) - np.asarray(self.cfg['goal_center'])))
            results.append({'trial': i, 'cube_x0': x, 'cube_y0': y, 'success': int(success),
                            'time': round(elapsed, 2), 'final_dist_to_goal': round(dist, 3)})
            self.get_logger().info(f'    → {"成功" if success else "失敗"}  '
                                   f'({elapsed:.1f}s, ゴールまで {dist:.3f}m)')

        self.call(self.home_cli, Trigger.Request())
        rate = np.mean([r['success'] for r in results]) if results else 0.0
        os.makedirs(self.out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.out_dir, f'{self.label}_{stamp}.csv')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        self.get_logger().info('=' * 40)
        self.get_logger().info(f'成功率: {rate * 100:.0f}% ({int(rate * len(results))}/{len(results)})  [{self.label}]')
        self.get_logger().info(f'結果: {path}')
        self.done = True


def main():
    rclpy.init()
    node = Evaluator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        node.run()
    except KeyboardInterrupt:
        try:
            node.set_enabled(False)
        except Exception:  # noqa: BLE001
            pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
