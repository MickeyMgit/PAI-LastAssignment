#!/bin/bash
# Mac 本体（Apple Silicon）に SmolVLA の推論サーバ用の Python 環境を作る（最初の1回だけ．約0.8GB）
# 必要なもの：uv（https://docs.astral.sh/uv/  インストール：curl -LsSf https://astral.sh/uv/install.sh | sh）
set -e
cd "$(dirname "$0")"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "lerobot[smolvla]==0.6.1" fastapi uvicorn
.venv/bin/python -c "import lerobot, torch; print('lerobot', lerobot.__version__, '/ MPS', torch.backends.mps.is_available())"
