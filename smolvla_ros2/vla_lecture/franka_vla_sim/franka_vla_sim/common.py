"""各ノードで共有する設定の読み込みと小さな道具．"""
import os
import shutil
import signal
import subprocess

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory


def load_task_config(path: str = '') -> dict:
    """config/task.yaml を読み込む．"""
    if not path:
        path = os.path.join(get_package_share_directory('franka_vla_sim'), 'config', 'task.yaml')
    with open(os.path.expanduser(path)) as f:
        return yaml.safe_load(f)


def interpolate(q_from, q_to, n_steps: int) -> list:
    """q_from から q_to まで n_steps 回で到達する関節角の列（初期姿勢へ戻すときに使う）．"""
    q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
    return [q_from + (q_to - q_from) * (i + 1) / n_steps for i in range(n_steps)]


def set_cube_pose(cfg: dict, x: float, y: float) -> bool:
    """Gazeboのサービスを呼んでキューブを (x, y) に置き直す．"""
    req = (f'name: "{cfg["cube_name"]}", position: {{x: {x}, y: {y}, z: {cfg["cube_z"]}}}, '
           'orientation: {w: 1.0}')
    cmd = ['ign', 'service', '-s', f'/world/{cfg["world_name"]}/set_pose',
           '--reqtype', 'ignition.msgs.Pose', '--reptype', 'ignition.msgs.Boolean',
           '--timeout', '3000', '--req', req]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return 'true' in result.stdout


def is_success(cfg: dict, cube_xy) -> bool:
    """キューブ中心がゴールの円の中に入っていれば成功．"""
    d = np.linalg.norm(np.asarray(cube_xy) - np.asarray(cfg['goal_center']))
    return bool(d < cfg['goal_radius'])


class EpisodeRecorder:
    """ros2 bag record をエピソードごとに起動・停止する（bag_root/ep_000, ep_001, ...）．"""

    def __init__(self, cfg: dict):
        self.bag_root = os.path.expanduser(cfg['teleop']['bag_root'])
        os.makedirs(self.bag_root, exist_ok=True)
        t = cfg['topics']
        self.topics = [t['joint_states'], t['arm_command'], t['camera_front'], t['camera_wrist'],
                       t['cube_pose']]
        self.proc = None
        self.path = None

    @property
    def recording(self) -> bool:
        return self.proc is not None

    def start(self) -> str:
        used = [int(d[3:]) for d in os.listdir(self.bag_root) if d.startswith('ep_') and d[3:].isdigit()]
        idx = max(used) + 1 if used else 0
        self.path = os.path.join(self.bag_root, f'ep_{idx:03d}')
        # --use-sim-time: 記録の時刻をGazeboのシミュレーション時間にそろえる
        # stdin=DEVNULL: ros2 bag record はキー操作（一時停止など）を受け付けるので，
        #                端末をつなぐとテレオペのキー入力を横取りしてしまう
        self.proc = subprocess.Popen(
            ['ros2', 'bag', 'record', '--use-sim-time', '-o', self.path] + self.topics,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        return self.path

    def stop(self, keep: bool = True) -> str:
        os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
        self.proc.wait(timeout=20)
        self.proc = None
        if not keep:
            shutil.rmtree(self.path, ignore_errors=True)
        return self.path
