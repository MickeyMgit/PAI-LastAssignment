"""推論ノード：カメラ画像と関節角を推論サーバ（REST API）に送り，action chunk を受け取る．

    ros2 run franka_vla_sim vla_client --ros-args -p server_url:=https://xxxx.trycloudflare.com

/vla/request_chunk (std_msgs/Empty) を受け取るたびに1回推論し，
結果を /vla/action_chunk (std_msgs/Float64MultiArray, 形状 [n_steps, 7]) に出す．
"""
import base64
import threading
import time

import cv2
import numpy as np
import rclpy
import requests
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Empty, Float64MultiArray, MultiArrayDimension

from franka_vla_sim.common import load_task_config


def encode_jpeg(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode('ascii')


class VlaClient(Node):

    def __init__(self):
        super().__init__('vla_client')
        self.declare_parameter('config', '')
        self.declare_parameter('server_url', 'http://localhost:8000')
        self.declare_parameter('task', '')
        self.cfg = load_task_config(self.get_parameter('config').value)
        self.url = self.get_parameter('server_url').value.rstrip('/')
        self.task = self.get_parameter('task').value or self.cfg['task']
        t = self.cfg['topics']

        self.bridge = CvBridge()
        self.latest = {'front': None, 'wrist': None, 'q': None}
        self.create_subscription(Image, t['camera_front'], lambda m: self.on_image('front', m), 1)
        self.create_subscription(Image, t['camera_wrist'], lambda m: self.on_image('wrist', m), 1)
        self.create_subscription(JointState, t['joint_states'], self.on_joint_states, 1)
        self.create_subscription(Empty, '/vla/request_chunk', self.on_request, 1)
        self.chunk_pub = self.create_publisher(Float64MultiArray, '/vla/action_chunk', 1)
        self.busy = False

        try:
            info = requests.get(f'{self.url}/health', timeout=10).json()
            self.get_logger().info(f'推論サーバに接続しました: {info}')
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'推論サーバに接続できません ({self.url}): {e}')
        self.get_logger().info(f'タスク: "{self.task}"')

    def on_image(self, name, msg):
        self.latest[name] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def on_joint_states(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if all(j in pos for j in self.cfg['arm_joints']):
            self.latest['q'] = [pos[j] for j in self.cfg['arm_joints']]

    def on_request(self, _msg):
        if self.busy:
            return
        if any(v is None for v in self.latest.values()):
            self.get_logger().warn('画像または関節角がまだ届いていません')
            return
        self.busy = True
        # 通信中もROSのコールバックが止まらないよう，別スレッドで推論する
        threading.Thread(target=self.infer, args=(dict(self.latest),), daemon=True).start()

    def infer(self, obs):
        payload = {
            'task': self.task,
            'state': obs['q'],
            'images': {'front': encode_jpeg(obs['front']), 'wrist': encode_jpeg(obs['wrist'])},
        }
        try:
            t0 = time.time()
            res = requests.post(f'{self.url}/predict', json=payload, timeout=30).json()
            actions = np.asarray(res['actions'], dtype=float)  # (n_steps, 7)
            self.get_logger().info(
                f'推論OK: {actions.shape[0]}ステップ分 / 往復 {time.time() - t0:.2f}s '
                f'(うちモデル {res.get("inference_time", 0):.2f}s)')
            msg = Float64MultiArray()
            msg.layout.dim = [MultiArrayDimension(label='steps', size=actions.shape[0], stride=actions.size),
                              MultiArrayDimension(label='joints', size=actions.shape[1], stride=actions.shape[1])]
            msg.data = actions.flatten().tolist()
            self.chunk_pub.publish(msg)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'推論に失敗しました: {e}')
        finally:
            self.busy = False


def main():
    rclpy.init()
    node = VlaClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
