"""実行ノード：推論ノードから受け取った action chunk を，一定周期でGazebo上のロボットに送る．

    ros2 run franka_vla_sim action_executor --ros-args -p n_action_steps:=10

- /vla/enable (std_srvs/SetBool)  : 方策での制御を開始 / 停止
- /vla/go_home (std_srvs/Trigger) : 初期姿勢へ戻す
- chunk の先頭 n_action_steps 個を fps で実行し，使い切ったら /vla/request_chunk で次を要求する
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64MultiArray
from std_srvs.srv import SetBool, Trigger

from franka_vla_sim.common import interpolate, load_task_config


class ActionExecutor(Node):

    def __init__(self):
        super().__init__('action_executor')
        self.declare_parameter('config', '')
        self.declare_parameter('n_action_steps', 10)
        self.declare_parameter('request_timeout', 10.0)
        self.cfg = load_task_config(self.get_parameter('config').value)
        self.n_action_steps = self.get_parameter('n_action_steps').value
        self.request_timeout = self.get_parameter('request_timeout').value
        t = self.cfg['topics']
        self.fps = self.cfg['fps']
        self.home_q = np.array(self.cfg['home_q'])

        self.arm_pub = self.create_publisher(Float64MultiArray, t['arm_command'], 10)
        self.hand_pub = self.create_publisher(Float64MultiArray, t['hand_command'], 10)
        self.request_pub = self.create_publisher(Empty, '/vla/request_chunk', 1)
        self.create_subscription(Float64MultiArray, '/vla/action_chunk', self.on_chunk, 1)
        self.create_subscription(JointState, t['joint_states'], self.on_joint_states, 1)
        self.create_service(SetBool, '/vla/enable', self.on_enable)
        self.create_service(Trigger, '/vla/go_home', self.on_go_home)

        self.enabled = False
        self.queue = []          # これから実行する関節角の列
        self.q_cmd = None        # 最後に送った関節角
        self.q_meas = None       # 計測された関節角
        self.last_request = None
        self.create_timer(1.0 / self.fps, self.on_timer)
        self.get_logger().info('待機中．/vla/enable で方策による制御を開始します')

    def on_joint_states(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if all(j in pos for j in self.cfg['arm_joints']):
            self.q_meas = np.array([pos[j] for j in self.cfg['arm_joints']])

    def on_chunk(self, msg):
        if not self.enabled:
            return
        n_joints = msg.layout.dim[1].size if len(msg.layout.dim) > 1 else len(self.cfg['arm_joints'])
        actions = np.asarray(msg.data).reshape(-1, n_joints)
        self.queue = list(actions[:self.n_action_steps])

    def on_enable(self, req, res):
        self.enabled = req.data
        self.queue = []
        self.last_request = None
        res.success = True
        res.message = '方策による制御を開始' if req.data else '停止'
        self.get_logger().info(res.message)
        return res

    def on_go_home(self, req, res):
        self.enabled = False
        start = self.q_cmd if self.q_cmd is not None else (self.q_meas if self.q_meas is not None else self.home_q)
        self.queue = interpolate(start, self.home_q, int(2.0 * self.fps))
        res.success = True
        return res

    def on_timer(self):
        now = self.get_clock().now()
        if self.queue:
            q = np.asarray(self.queue.pop(0))
            # 安全のため，1周期での変化量と関節可動域を制限する
            if self.q_cmd is not None:
                step = self.cfg['max_joint_step']
                q = self.q_cmd + np.clip(q - self.q_cmd, -step, step)
            self.q_cmd = q
            self.arm_pub.publish(Float64MultiArray(data=[float(v) for v in q]))
            self.hand_pub.publish(Float64MultiArray(data=[float(v) for v in self.cfg['hand_closed']]))
        elif self.enabled:
            # chunk を使い切った → 次の推論を要求（返ってこなければ一定時間後に再要求）
            timed_out = (self.last_request is not None
                         and (now - self.last_request).nanoseconds * 1e-9 > self.request_timeout)
            if self.last_request is None or timed_out:
                self.request_pub.publish(Empty())
                self.last_request = now
            return
        if not self.queue:
            self.last_request = None


def main():
    rclpy.init()
    node = ActionExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
