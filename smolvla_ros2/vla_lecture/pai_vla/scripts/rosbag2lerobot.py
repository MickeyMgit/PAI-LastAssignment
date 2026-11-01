#!/usr/bin/env python3
"""rosbag2（1エピソード = 1 bag）を LeRobotDataset (v3.0) に変換し，必要なら Hugging Face Hub に上げる．

    pixi run convert                  # 変換のみ
    pixi run convert --push           # 変換して Hub にアップロード
    pixi run convert --push --private # 非公開でアップロード

rosbagに記録されたトピックは非同期（カメラ10Hz, 関節角100Hz, 指令10Hz, ...）なので，
fps の等間隔の時刻列を作り，各時刻の値を取り出して同期させる．
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def read_bag(bag_dir: Path, topics: list[str]) -> dict:
    """bag から指定トピックの (時刻[ns], メッセージ) の列を読む．時刻は記録時刻（sim time）．"""
    out = {t: [] for t in topics}
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag_dir], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in out]
        for conn, stamp, raw in reader.messages(connections=conns):
            out[conn.topic].append((stamp, reader.deserialize(raw, conn.msgtype)))
    for t, msgs in out.items():
        if not msgs:
            raise ValueError(f'{bag_dir}: トピック {t} のメッセージがありません')
    return out


def sample(stamps: np.ndarray, values: np.ndarray, grid: np.ndarray, mode: str) -> np.ndarray:
    """grid の各時刻に対応する値を取り出す．

    mode='latest': 時刻以前で最新の値（観測: その時点でロボットが見ていたもの）
    mode='next'  : 時刻以降で最初の値（行動: その観測を見た直後に送った指令）
    """
    if mode == 'latest':
        idx = np.searchsorted(stamps, grid, side='right') - 1
    else:
        idx = np.searchsorted(stamps, grid, side='left')
    return values[np.clip(idx, 0, len(values) - 1)]


def image_to_numpy(msg) -> np.ndarray:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if msg.encoding == 'bgr8':
        img = img[..., ::-1]
    return np.ascontiguousarray(img[..., :3])


def convert_episode(bag_dir: Path, cfg: dict) -> dict:
    t = cfg['topics']
    cam_topics = t['cameras']
    msgs = read_bag(bag_dir, [t['joint_states'], t['action'], *cam_topics.values()])

    # 関節角（名前で並べ替え）
    js = msgs[t['joint_states']]
    js_t = np.array([s for s, _ in js])
    js_v = np.array([[dict(zip(m.name, m.position))[j] for j in cfg['joint_names']] for _, m in js])
    # 行動 = /arm_controller/commands に送った関節角の目標値
    act = msgs[t['action']]
    act_t = np.array([s for s, _ in act])
    act_v = np.array([np.asarray(m.data) for _, m in act])
    # 画像
    cams = {}
    for name, topic in cam_topics.items():
        cams[name] = (np.array([s for s, _ in msgs[topic]]), [m for _, m in msgs[topic]])

    # 全トピックがそろっている区間で，fps 間隔の時刻列を作る
    start = max(js_t[0], act_t[0], *(c[0][0] for c in cams.values()))
    end = min(js_t[-1], act_t[-1], *(c[0][-1] for c in cams.values()))
    grid = np.arange(start, end, int(1e9 / cfg['fps']))

    frames = {
        'observation.state': sample(js_t, js_v, grid, 'latest').astype(np.float32),
        'action': sample(act_t, act_v, grid, 'next').astype(np.float32),
    }
    for name, (stamps, images) in cams.items():
        idx = np.clip(np.searchsorted(stamps, grid, side='right') - 1, 0, len(images) - 1)
        frames[f'observation.images.{name}'] = [image_to_numpy(images[i]) for i in idx]

    if cfg.get('trim_idle', True):
        # 指令が変化していない先頭・末尾の区間を削る（少し余裕を残す）
        moving = np.where(np.linalg.norm(np.diff(frames['action'], axis=0), axis=1) > 1e-4)[0]
        if len(moving) > 0:
            s, e = max(0, moving[0] - 3), min(len(grid), moving[-1] + 5)
            frames = {k: v[s:e] for k, v in frames.items()}
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/conversion_config.yaml')
    parser.add_argument('--push', action='store_true', help='変換後に Hugging Face Hub へアップロード')
    parser.add_argument('--private', action='store_true', help='非公開リポジトリとしてアップロード')
    parser.add_argument('--overwrite', action='store_true', help='出力先が既にあれば削除して作り直す')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    bag_dir = Path(cfg['bag_dir']).expanduser()
    out_dir = Path(cfg['output_dir']).expanduser()
    episodes = sorted(p.parent for p in bag_dir.glob('ep_*/metadata.yaml'))
    if not episodes:
        raise FileNotFoundError(f'{bag_dir} に rosbag (ep_*/metadata.yaml) がありません')
    print(f'{len(episodes)} エピソードを変換します: {bag_dir}')

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f'{out_dir} は既に存在します．--overwrite で作り直せます')
        shutil.rmtree(out_dir)

    n = len(cfg['joint_names'])
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (n,), 'names': cfg['joint_names']},
        'action': {'dtype': 'float32', 'shape': (n,), 'names': cfg['joint_names']},
    }
    first = convert_episode(episodes[0], cfg)
    for name in cfg['topics']['cameras']:
        h, w, c = first[f'observation.images.{name}'][0].shape
        features[f'observation.images.{name}'] = {
            'dtype': 'video', 'shape': (h, w, c), 'names': ['height', 'width', 'channels']}

    dataset = LeRobotDataset.create(
        repo_id=cfg['repo_id'], fps=cfg['fps'], features=features, root=out_dir,
        robot_type=cfg.get('robot_type'), use_videos=True)

    for i, ep in enumerate(episodes):
        try:
            frames = first if i == 0 else convert_episode(ep, cfg)
        except Exception as e:  # noqa: BLE001
            print(f'  [SKIP] {ep.name}: {e}')
            continue
        length = len(frames['action'])
        for k in range(length):
            frame = {key: val[k] for key, val in frames.items()}
            frame['task'] = cfg['task']
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f'  {ep.name}: {length} フレーム ({length / cfg["fps"]:.1f}s)')

    dataset.finalize()
    print(f'保存しました: {out_dir}  ({dataset.meta.total_episodes} エピソード, '
          f'{dataset.meta.total_frames} フレーム)')

    if args.push:
        dataset.push_to_hub(private=args.private)
        print(f'アップロードしました: https://huggingface.co/datasets/{cfg["repo_id"]}')


if __name__ == '__main__':
    main()
