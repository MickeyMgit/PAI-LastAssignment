"""SmolVLA の推論用 REST API サーバ．

    python smolvla_server.py --model YOUR_HF_USERNAME/smolvla_franka_push --port 8000

エンドポイント
  GET  /health   : サーバとモデルの情報
  POST /predict  : 観測を送ると action chunk を返す
      入力 {"task": str, "state": [7], "images": {"front": base64 JPEG, "wrist": base64 JPEG}}
      出力 {"actions": [[7], ...], "inference_time": float}
"""
import argparse
import base64
import io
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# ROS側のカメラ名 → データセット（モデル）上の画像キー
CAMERA_KEYS = {'front': 'observation.images.camera1', 'wrist': 'observation.images.camera2'}


class Observation(BaseModel):
    task: str
    state: list[float]
    images: dict[str, str]


def decode_image(b64: str) -> torch.Tensor:
    """base64 JPEG → (3, H, W) の float32 テンソル（0〜1）．"""
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0


def create_app(model_id: str, device: str, n_steps: int | None = None) -> FastAPI:
    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy.to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, pretrained_path=model_id,
        preprocessor_overrides={'device_processor': {'device': device}},
        postprocessor_overrides={'device_processor': {'device': 'cpu'}})
    chunk_size = policy.config.chunk_size
    app = FastAPI()

    @app.get('/health')
    def health():
        return {'model': model_id, 'device': device, 'chunk_size': chunk_size,
                'cameras': CAMERA_KEYS}

    @app.post('/predict')
    def predict(obs: Observation):
        t0 = time.time()
        batch = {'observation.state': torch.tensor(obs.state, dtype=torch.float32), 'task': obs.task}
        for name, key in CAMERA_KEYS.items():
            batch[key] = decode_image(obs.images[name])
        with torch.inference_mode():
            batch = preprocess(batch)
            policy.reset()  # 過去の観測・行動のキューを消す（毎回chunkを丸ごと推論する）
            chunk = policy.predict_action_chunk(batch)  # (1, chunk_size, 7) 正規化された値
            # 正規化を戻す（postprocess は1ステップ分 (1, 7) ずつ受け取る）
            actions = torch.stack([postprocess(chunk[:, i]) for i in range(chunk.shape[1])], dim=1)[0]
        if n_steps is not None:
            actions = actions[:n_steps]
        return {'actions': actions.cpu().numpy().tolist(), 'inference_time': time.time() - t0}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='HF Hub のモデルID またはローカルのパス')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model, args.device), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
