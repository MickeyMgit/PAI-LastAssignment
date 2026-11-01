# ROS 2 × SmolVLA：データ収集から実行まで（第5回 参考資料）

Gazebo 上の Franka FR3 で「赤いキューブを緑のゴールへ押す」タスクを使い， **データ収集 → 変換 → 学習 → 実行・評価** を VLA（SmolVLA）で一通り行うための手順とファイル一式です．

| 段階 | 場所 | 使うもの |
| :---- | :---- | :---- |
| データ収集・変換 | コンテナ（ブラウザの VNC） | `franka_vla_sim`（ROS 2 パッケージ），`pai_vla`（pixi 環境） |
| 学習 | Google Colab（T4 GPU） | `notebooks/smolvla_finetune.ipynb` |
| 推論サーバ | **Mac 本体**（MPS） | `mac_server/`，`server/smolvla_server.py` |
| 実行・評価 | コンテナ（ブラウザの VNC） | `policy.launch.py`，`evaluator` |

&nbsp;

```
[コンテナ：ROS 2 + Gazebo]                                       [Mac 本体]
 Gazebo ◀─ action_executor ◀─ vla_client ──HTTP（画像・関節角）──▶ smolvla_server.py（MPS）
   ▲                                     ◀──── 行動 50ステップ分 ───┘
   └─ evaluator（決まった位置にキューブを置いて試行し，成功率を CSV に保存）
```

推論をコンテナの外で行うのは，コンテナの CPU が amd64 のエミュレーションで遅く（Mac の CPU でも1回約32秒）， メモリ（約4GB）も足りないためです．Mac の MPS なら1回 1.3〜2.0秒で推論できます．

## フォルダの中身

```
smolvla_ros2/
├── README.md                  # この手順書
├── vla_lecture/               # 教材一式（コンテナの ~/vla_lecture にコピーする）
│   ├── franka_vla_sim/        #   ROS 2 パッケージ（シミュレータ・テレオペ・推論/実行/評価ノード）
│   │   └── config/task.yaml   #   ★ 全体で共有する設定（fps・トピック・言語指示・評価条件）
│   ├── pai_vla/               #   rosbag → LeRobotDataset 変換（pixi 環境）
│   │   └── configs/conversion_config.yaml   # ★ 変換の設定（repo_id を自分のユーザー名に）
│   ├── notebooks/smolvla_finetune.ipynb     # Colab で学習
│   ├── server/smolvla_server.py             # 推論サーバ本体（FastAPI）
│   └── README.md              #   教材の詳しい説明（設計のポイントなど）
└── mac_server/
    ├── setup.sh               # Mac に推論サーバ用の Python 環境を作る（最初の1回）
    └── start_server.sh        # 推論サーバを起動する
```

## 0\. 準備

- 第5回のコンテナ `pai_ros2`（`airobotbook/ros2-desktop-ai-robot-book-humble`．作業フォルダを `/workspace` にマウントしたもの）  
- Hugging Face のアカウントと **Write 権限のトークン**  
- Mac に [uv](https://docs.astral.sh/uv/)（`curl -LsSf https://astral.sh/uv/install.sh | sh`）  
- このフォルダ `smolvla_ros2` を **作業フォルダ（コンテナの `/workspace`）** に置く

## 1\. 教材のコピーとビルド（コンテナ・最初の1回）

Mac のターミナルで `docker start pai_ros2` → ブラウザで [http\://localhost:6080/](http://localhost:6080/) → Terminator を開く．

```sh
cp -r /workspace/smolvla_ros2/vla_lecture ~/
cd ~/pai_ws/src
git clone -b 2.9.0 https://github.com/frankaemika/franka_description.git
ln -s ~/vla_lecture/franka_vla_sim .
cd ~/pai_ws && rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select franka_description franka_vla_sim
cd ~/vla_lecture/pai_vla && pixi install        # 変換用の環境（CPU 版 PyTorch，約240MB）
```

**新しいターミナルを開くたびに `source ~/pai_ws/install/setup.bash` を実行する**（以下のコマンドはすべてこの後に打つ）．

## 2\. データ収集（コンテナ）

```sh
# 端末A：シミュレータ（起動に1〜2分）
ros2 launch franka_vla_sim sim.launch.py

# 端末B：テレオペ（このターミナルをクリックしてからキーを押す）
ros2 run franka_vla_sim teleop_keyboard --ros-args -p use_sim_time:=true

# 端末C：手元を見やすくする
source ~/pai_ws/install/setup.bash
ros2 run rqt_image_view rqt_image_view /cameras/wrist/image
```

&nbsp;

| キー | 動作 |
| :---- | :---- |
| W / S，A / D，R / F | 手先を前後・左右・上下に動かす |
| N | 新しいエピソード（キューブを置き直し，初期姿勢から記録開始） |
| M / X | 記録を終えて保存 / 中止して破棄 |
| H | 記録せずにリセット（練習用） |
| Q | 終了 |

**rosbag の記録はテレオペが自動で行う**（`ros2 bag record` を自分で打つ必要はない）．N を押すと初期姿勢に戻ってから記録が始まり（端末B に `[REC] 記録開始: .../ep_000`），M で止めて保存する．中では `ros2 bag record --use-sim-time -o ~/dataset/rosbags/franka_push/ep_XXX /joint_states /arm_controller/commands /cameras/front/image /cameras/wrist/image /model/red_cube/pose` を実行している（`common.py` の `EpisodeRecorder`）．

キーはテレオペの端末をクリックしてから押す（rqt などを開くとキー入力の先が移る．入力モードは英数）．

コツ：F で手先をキューブの高さまで下げ，**キューブのゴールと反対側に回り込んでから**押す． 押しすぎ・外れたものは X で捨てる（失敗も手本として学習してしまう）．

```sh
ls ~/dataset/rosbags/franka_push/                    # ep_000, ep_001, ...
ros2 bag info ~/dataset/rosbags/franka_push/ep_000   # 各トピックの Count が 0 でないか
```

## 3\. 変換と Hugging Face へのアップロード（コンテナ）

```sh
cd ~/vla_lecture/pai_vla
pixi run hf auth login                     # Write 権限のトークン（入力は表示されない）
pixi run hf auth whoami                    # ユーザー名を確認
sed -i 's#YOUR_HF_USERNAME#<ユーザー名>#' configs/conversion_config.yaml
grep repo_id configs/conversion_config.yaml
pixi run convert --push                    # LeRobotDataset v3.0 に変換してアップロード（--private で非公開）
```

- コンテナに nano は入っていない（1行の書き換えは `sed`．vim / pluma も使える）  
- 例：rosbag 5本（310MB）→ 1.9MB（5エピソード・743フレーム・10fps・画像 256×256）

## 4\. 学習（Colab）

1. `vla_lecture/notebooks/smolvla_finetune.ipynb` を Colab で開く → ランタイムを **T4 GPU** に変更  
2. 🔑 シークレットに `HF_TOKEN`（Write 権限）を登録  
3. 設定のセルで `DATASET_REPO_ID = "<ユーザー名>/franka_push"`，`STEPS`・`BATCH_SIZE` を決めて上から実行  
4. 学習したモデルは Hub の `<ユーザー名>/smolvla_franka_push`（`checkpoints/000200/` など）に上がる

目安：T4 で200ステップが動作（時間はかかる）．Mac M2 の MPS は1ステップ17〜25秒で学習には向かない．

## 5\. 推論サーバ（Mac 本体のターミナル）

```sh
cd <作業フォルダ>/smolvla_ros2/mac_server         
./setup.sh                                          # 最初の1回だけ（約0.8GB）

./start_server.sh <ユーザー名>/smolvla_franka_push  # Hub から重みを取ってきて起動
# または，Colab から重みのフォルダ（config.json・model.safetensors などが入ったもの）を持ってきた場合：
./start_server.sh ~/.../<作業フォルダ>
# 作業フォルダはmodel.safetensorsがあるフォルダ
```

`Uvicorn running on http://0.0.0.0:8000` と出たら準備完了．**このターミナルは開いたままにする**（止めるときは Ctrl+C）． 確認：別のターミナルで `curl localhost:8000/health` → `"device":"mps"` が出れば OK．

`start_server.sh` の中身は次の1行と同じ（`.venv` の Python で起動しているだけ）：

```sh
python ../vla_lecture/server/smolvla_server.py --model <重みのフォルダ> --device mps --port 8000
```

## 6\. 実行・評価（コンテナ）

```sh
# 端末A：シミュレータ（推論と同時に動かすので必ず GUI なし）
ros2 launch franka_vla_sim sim.launch.py gui:=false

# 端末B：推論ノード＋実行ノード →「推論サーバに接続しました: {... 'device': 'mps' ...}」が出れば OK
ros2 launch franka_vla_sim policy.launch.py server_url:=http://host.docker.internal:8000

# 端末C：評価（決まった位置にキューブを置いて試行．結果は ~/vla_eval/ に CSV）
ros2 run franka_vla_sim evaluator --ros-args -p n_trials:=1 -p label:=mac_mps -p use_sim_time:=true
cat $(ls -t ~/vla_eval/*.csv | head -1)
```

- 端末B に `推論OK: 50ステップ分 / 往復 ○s` が1〜2秒ごとに出ていれば推論が回っている  
- 評価を使わずに1回動かす：`ros2 service call /vla/enable std_srvs/srv/SetBool "{data: true}"`（止める：`{data: false}`，初期姿勢へ：`/vla/go_home`）  
- 試行数を増やすときは `n_trials:=10`（初期位置は `task.yaml` の `cube_positions` の順）

### 可視化

```sh
# 端末D：カメラの映像（FR3 がキューブを押す様子）．重いときは閉じる
ros2 run rqt_image_view rqt_image_view /cameras/front/image      # 正面カメラ
ros2 run rqt_image_view rqt_image_view /cameras/wrist/image      # 手首カメラ

# キューブの位置（成功判定に使っている値）
ros2 topic echo /model/red_cube/pose --field position
```

- 集めたデータセットの確認：[https\://huggingface.co/spaces/lerobot/visualize\_dataset](https://huggingface.co/spaces/lerobot/visualize_dataset) に `<ユーザー名>/franka_push` を入力

### 終わり方

各端末で Ctrl+C → Mac の推論サーバも Ctrl+C → Mac のターミナルで `docker stop pai_ros2`