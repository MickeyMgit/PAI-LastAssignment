"""URDFから順運動学を組み立て，手先空間テレオペ用の逆運動学（減衰最小二乗法）を解く．

MoveItなどを使わずに，numpyだけで動く最小限の実装．
"""
import xml.etree.ElementTree as ET

import numpy as np


def _rpy_to_rot(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle_to_rot(axis, angle):
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class KinematicChain:
    """base_link から tip_link までの関節の鎖．"""

    def __init__(self, urdf_xml: str, joint_names: list, base_link: str, tip_link: str):
        root = ET.fromstring(urdf_xml)
        joints_by_child = {j.find('child').get('link'): j for j in root.findall('joint')}

        # tip から base に向かって親をたどる
        chain = []
        link = tip_link
        while link != base_link:
            j = joints_by_child[link]
            chain.append(j)
            link = j.find('parent').get('link')
        chain.reverse()

        self.joint_names = list(joint_names)
        self.elements = []  # (固定部分の変換, 可動関節の番号 or None, 回転軸)
        self.lower = np.full(len(joint_names), -np.inf)
        self.upper = np.full(len(joint_names), np.inf)
        for j in chain:
            origin = j.find('origin')
            xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()] if origin is not None else [0, 0, 0]
            rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()] if origin is not None else [0, 0, 0]
            T_fixed = _transform(_rpy_to_rot(*rpy), xyz)
            name = j.get('name')
            if j.get('type') in ('revolute', 'continuous') and name in self.joint_names:
                idx = self.joint_names.index(name)
                axis_el = j.find('axis')
                axis = np.array([float(v) for v in axis_el.get('xyz').split()]) if axis_el is not None \
                    else np.array([1.0, 0.0, 0.0])
                limit = j.find('limit')
                if limit is not None:
                    self.lower[idx] = float(limit.get('lower', -np.inf))
                    self.upper[idx] = float(limit.get('upper', np.inf))
                self.elements.append((T_fixed, idx, axis / np.linalg.norm(axis)))
            else:
                self.elements.append((T_fixed, None, None))

    def fk(self, q, return_joints=False):
        """関節角 q (7,) から手先の4x4同次変換行列を返す．"""
        T = np.eye(4)
        joints = []  # 各可動関節の (番号, 回転軸[world], 位置[world])
        for T_fixed, idx, axis in self.elements:
            T = T @ T_fixed
            if idx is not None:
                joints.append((idx, T[:3, :3] @ axis, T[:3, 3].copy()))
                T = T @ _transform(_axis_angle_to_rot(axis, q[idx]), [0, 0, 0])
        return (T, joints) if return_joints else T

    def jacobian(self, q):
        """幾何ヤコビアン (6x7)．上3行が並進，下3行が回転．"""
        T, joints = self.fk(q, return_joints=True)
        J = np.zeros((6, len(q)))
        for idx, z, p in joints:
            J[:3, idx] = np.cross(z, T[:3, 3] - p)
            J[3:, idx] = z
        return T, J

    @staticmethod
    def _error(T, p_target, R_target):
        """手先の位置誤差と姿勢誤差 (6,)．"""
        e_pos = p_target - T[:3, 3]
        R = T[:3, :3]
        e_rot = 0.5 * sum(np.cross(R[:, i], R_target[:, i]) for i in range(3))
        return np.concatenate([e_pos, e_rot])

    def ik(self, q_init, p_target, R_target, q_rest=None, iters=10, damping=0.05, null_gain=0.1):
        """手先を (p_target, R_target) にする関節角を q_init から反復的に求める．

        冗長自由度（7自由度-6次元）は q_rest（初期姿勢）に近づけるのに使う．
        """
        q = np.array(q_init, float)
        for _ in range(iters):
            T, J = self.jacobian(q)
            e = self._error(T, p_target, R_target)
            if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e[3:]) < 5e-3:
                break
            JJt = J @ J.T + (damping ** 2) * np.eye(6)
            J_pinv = J.T @ np.linalg.inv(JJt)
            dq = J_pinv @ e
            if q_rest is not None:
                dq += (np.eye(len(q)) - J_pinv @ J) @ (null_gain * (np.asarray(q_rest) - q))
            q = np.clip(q + dq, self.lower + 1e-3, self.upper - 1e-3)
        return q
