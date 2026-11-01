#!/bin/bash
# Mac 本体（MPS）で SmolVLA の推論サーバを起動する
# 使い方：
#   ./start_server.sh <重みのフォルダ>                  # 例：Colab から持ってきた smolvla フォルダ
#   ./start_server.sh <ユーザー名>/smolvla_franka_push   # Hugging Face Hub から取ってくる（200ステップ時点）
# "Uvicorn running on http://0.0.0.0:8000" と出たら準備完了．止めるときは Ctrl+C
set -e
cd "$(dirname "$0")"
M=${1:?重みのフォルダ，または Hub のリポジトリ名を指定してください}
if [ ! -d "$M" ]; then
  P=checkpoints/000200/pretrained_model
  .venv/bin/hf download "$M" --include "$P/*" --local-dir ckpt
  M=ckpt/$P
fi
.venv/bin/python ../vla_lecture/server/smolvla_server.py --model "$M" --device mps --port 8000
