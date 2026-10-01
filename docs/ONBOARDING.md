# sam3.cpp 開発オンボーディング

更新: 2026-10-01。対象は `develop`。コマンドはリポジトリルートで実行します。
新しい開発者が、モデル選択、取得・変換、headless annotation、検証まで進むための入口です。

## 1. 最初に行うこと

1. [README](../README.md)、[CLAUDE.md](../CLAUDE.md)、[PLAN.md](../PLAN.md) を読む。
2. `git status --short --branch` と `git log -1 --oneline` で既存変更を確認する。
3. 下の表からモデルを選び、そのモデルの環境だけを構築する。
4. モデルなしの境界テストを通し、実重みを取得・変換する。
5. 画像1枚から始め、画像列の出力・順序・品質を確認する。
6. 推論を変更した場合は実モデル比較、最適化した場合は同条件で計測する。
7. 明示したファイルを commit・push して PR にする。

```sh
git clone --branch develop https://github.com/yuki-inaho/sam3.cpp.git
cd sam3.cpp
git switch -c feature/my-change
git status --short --branch
```

## 2. モデルと実行経路

| モデル | 入力・用途 | 推論 | 入口 |
| --- | --- | --- | --- |
| 同梱 EdgeTAM | 点・矩形、画像・動画 | C++14 / ggml、CPU・Metal | [SETUP_GUIDE](SETUP_GUIDE.md) |
| SAM 3 | テキスト・視覚プロンプト | C++14 / ggml | [README](../README.md)、モデル別 Feature Matrix |
| SAM 3.1 ConvRot INT8 | 初期画像の点指定、1〜16対象の前向き追跡 | `sam31/native/` の C++17 / CPU FP32・CBLAS | [SAM31_GUIDE](SAM31_GUIDE.ja.md) |
| EfficientSAM3 EV-M | テキストによる画像・明示順の画像列検出 | C++14 / ggml、CPU | [EFFICIENTSAM3](EFFICIENTSAM3.md) |

SAM3.1 の GGUF は元の INT8・尺度・回転を保持し、使用時に逆 ConvRot と FP32 復元を行います。
GGML Q8_0 への再量子化や GPU INT8 演算ではありません。推論は Python/PyTorch/ONNX Runtime を起動しません。

EV-M の公開 checkpoint に memory tracker はありません。画像列はフレームごとの独立検出です。
query ID を物体の追跡 ID として扱わず、PVS・tracker API の明示拒否を維持してください。
対応 variant は EfficientViT **b1** + MobileCLIP **S0**、context **16**、入力 **1008×1008** です。

## 3. 主要ファイルと責務

| パス | 用途 |
| --- | --- |
| `sam3.cpp` / `sam3.h` | SAM3・EdgeTAM・EV-M のライブラリと公開 API |
| `ggml/` / `stb/` | 固定された vendored 依存。ggml は submodule ではない |
| `examples/efficientsam3.cpp` | EV-M の headless 画像・画像列 CLI |
| `convert_efficientsam3_to_gguf.py` | EV-M の strict 検査と全 payload GGUF 照合 |
| `efficientsam3/` | CPU 専用 uv 環境、固定取得・参照・E2E |
| `sam31/native/` | SAM3.1 の GGUF 読込、演算、追跡、CLI |
| `sam31/tools/` / `sam31/tests/` | 変換、人工入力、実モデル E2E、境界・回帰検査 |
| `tests/test_efficient_*.py` | EV-M CLI 境界と opt-in 実モデル E2E |
| `docs/efficientsam3-validation/` / `docs/sam31-validation/` | 公開重みの変換・数値・性能・復元証跡 |
| `scripts/package_*.py` | source/docs/tests の zstd 梱包 |
| `models/`、`outputs/`、`runs/`、`build/`、`dist/` | 重み・生成物・作業用。新規生成物を commit しない |

## 4. 前提条件と環境

- Git、uv、CMake、C++ compiler。Python の実行は必ず `uv run` を使う。
- EV-M の Python は 3.11 または 3.12。CPU torch を `efficientsam3/.venv` に同期する。
- SAM3.1 は C++17、OpenBLAS、zstd を用意し、`sam31/.venv` を使う。root の torch 環境は不要。
- GUI は SDL2 / OpenGL、動画切り出しは ffmpeg が別途必要。以下の CLI は headless。
- 重み、Python環境、変換前後ファイル、検証出力の容量を見込む。重い export/E2E を同時実行しない。

### 4.1 EV-M：取得 → GGUF → annotation

```sh
uv sync --project efficientsam3 --locked
uv run --project efficientsam3 python efficientsam3/fetch.py
uv run --project efficientsam3 python convert_efficientsam3_to_gguf.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSAM3_BUILD_SAM31=OFF -DSAM3_METAL=OFF -DGGML_NATIVE=OFF
cmake --build build --target efficientsam3 --parallel 8
mkdir -p output/efficient
build/examples/efficientsam3 \
  --model models/efficientsam3_ev_m.gguf \
  --image outputs/efficientsam3/source/sam3/assets/dog_person.jpeg \
  --text dog --threads 4 --output output/efficient
```

画像列は `--image frame0.png --image frame1.png` を入力順に指定します。
出力は mask PNG と `annotations.json`（frame index、score、pixel box、時間）。
source revision と checkpoint SHA を固定し、799 keys・shape・dtype・有限値を検査して strict ロードします。
upstream の部分ロード helper を使用しません。実測 parameters は **97,435,030** で、上流 README 表の89.2Mと異なります。

### 4.2 SAM3.1：SafeTensors → GGUF → annotation

取得元の利用条件を確認し、必要な認証は端末の `HF_TOKEN` または HF credential から渡します。
トークン値を表示したり文書へ保存したりしないでください。

```sh
uv sync --project sam31 --python 3.11 --locked
uv run --project sam31 python sam31/tools/fetch_assets.py --models-dir models
uv run --project sam31 python sam31/tools/convert_sam31_to_gguf.py \
  --input models/sam3.1_multiplex_convrot_int8.safetensors \
  --output models/sam3.1_multiplex_convrot_int8.gguf --verify
cmake -S . -B build-sam31 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DSAM3_BUILD_SAM31=ON -DSAM3_METAL=OFF -DSAM31_BUILD_REFERENCE=OFF \
  -DPython3_EXECUTABLE="$PWD/sam31/.venv/bin/python"
cmake --build build-sam31 --parallel 8
build-sam31/sam31/sam31 synthetic --output runs/synthetic --frames 6
OPENBLAS_NUM_THREADS=16 build-sam31/sam31/sam31_image \
  --model models/sam3.1_multiplex_convrot_int8.gguf \
  --image runs/synthetic/image.png --point 1:0.25:0.32 --point 2:0.73:0.68 \
  --threads 16 --cache-mb 4096 --output runs/image
OPENBLAS_NUM_THREADS=16 build-sam31/sam31/sam31_video \
  --model models/sam3.1_multiplex_convrot_int8.gguf \
  --frames runs/synthetic/frames --point 1:0.25:0.32 --point 2:0.73:0.68 \
  --threads 16 --cache-mb 4096 --output runs/video
```

`--point ID:X:Y[:LABEL]` は正規化座標、正例1・負例0です。初期画像で複数点を指定できます。
動画は同寸法の2枚以上の画像列を自然順で読みます。MP4 は先に ffmpeg で切り出します。
出力先の再利用は拒否されるため、新しいディレクトリを指定してください。
mask PNG、FP32 logits、`report.json` の形式と独立した IoU 検査は [詳細ガイド](SAM31_GUIDE.ja.md) を参照します。

## 5. 検証と完了判定

| 変更 | 最初の検査 | 実モデルでの確認 |
| --- | --- | --- |
| EV-M 変換・CLI | 下の pytest | 全799 payload照合、PyTorchとの画像・6frame比較 |
| SAM3.1 演算・CLI | CTest の高速5グループ | 指定GGUFの画像2mask・動画12mask、finite・IoU・ID・memory |
| ggml encoder / PCS | 段階別 trace と既存対象テスト | 同じ前処理入力で最初にずれる stage を特定 |
| docs / 梱包 | 相対リンク・CLI引数、`git diff --check` | zstd整合性・SHA・復元ファイル・owner header |

```sh
uv run --project efficientsam3 pytest \
  tests/test_efficientsam3_conversion.py tests/test_efficient_cli.py -q
OPENBLAS_NUM_THREADS=1 ctest --test-dir build-sam31 -LE real --output-on-failure
```

実モデル E2E は source・checkpoint・GGUF・binary を配置後に明示実行します。

```sh
EFFICIENTSAM3_REAL=1 uv run --project efficientsam3 pytest tests/test_efficient_e2e.py -q
uv run --project sam31 python sam31/tools/run_native_e2e.py \
  --binary build-sam31/sam31/sam31 --model models/sam3.1_multiplex_convrot_int8.gguf \
  --threads 16 --output runs/native-e2e
```

skip は実モデル成功の証拠に数えません。`TEST_ONLY` の縮小・未学習 fixture と指定実重みを区別します。
2026-10-01 の EV-M 6frame PyTorch 比較は最小 IoU **0.999962**、同一CHWの最終mask比較は **0.999993**。
SAM3.1 は人工入力の全14maskが正解と IoU **1.0**。
SAM3.1 の一般画像精度や公式 PyTorch との全面的な数値一致は、この smoke の検証範囲に含みません。

## 6. アーキテクチャ・性能の契約

- `sam3.cpp` は structs / free functions、C++14、例外なし。内部 static 関数は `sam3_` prefix。
- ggml の各 stage は専用 context / graph / gallocr を持ち、CPU vector で受け渡す。
  前 stage の state tensor を新 graph の operand に入れない。祖先の encoder を再計算し、静かに出力が壊れる原因になる。
- model weights は永続 buffer。vendored ggml の更新は独立した検証を伴う変更として扱う。
- SAM3.1 は独立した C++17 module。CBLAS と OpenMP の thread 数は別で、
  `OPENBLAS_NUM_THREADS` と `--threads` を両方記録する。
- `SAM3_EFFICIENT_PROFILE=1` は EV-M stage 時間、SAM3.1 の `--profile` は operator 時間を記録する。
- 比較時は model SHA、入力、前処理、build flags、backend、threads、反復回数を揃える。
  I/O・初回load・cacheを含む全体時間と stage時間を区別する。

SAM3.1 の等価並列化では同条件1回の画像計測が44.872秒→35.354秒（21.2%短縮）、logitsはbyte一致。
EV-M の4threads約9.75秒 / 16threads約10.95秒も各1回の観測で、速度保証ではありません。
採用しなかった grouped GEMM 案を改善として報告しないでください。

## 7. 公開物・変更管理

Git に含めるのは source、tests、lockfiles、再現手順、公開情報だけの検証 JSON です。
追加の実重み、ONNX、annotation payload、個人画像、認証情報、shell設定、端末固有の絶対パス、
セッション transcript は含めません。同梱 EdgeTAM / sample data は既存の例外です。
source archive とモデルは別々に zstd 圧縮し、展開後の `FILES.sha256` とモデルSHAを検査します。
梱包手順は各詳細ガイドにあります。tar owner名を除き、環境cacheを含めないでください。

```sh
git diff --check
git diff --cached --stat
git diff --cached
```

作業は feature branch → PR を基本とし、対象 base（この実装は `develop`）を明記します。
ユーザーの明示した commit / push / merge の指示を実行します。CI の未実行・skip を成功と記録しません。
作業終了時は提出物と原本を保持し、今回作成した一時 checkout だけを整理します。
`uv cache prune` は `uv run` の子として起動せず、端末から直接実行します。

## 8. トラブルシューティング

| 症状 | 確認と対応 |
| --- | --- |
| SHA / revision / shape 不一致 | variant と取得元を確認。strict 検査を緩めず、固定 revision を取得し直す |
| SAM3.1 が極端に遅い | configure の `CBLAS acceleration enabled` を確認。OpenBLAS後入れならガイドの検出cache更新を使う |
| EV-M E2E が skip | source・checkpoint・GGUF・binary を配置し、`EFFICIENTSAM3_REAL=1` を指定 |
| 画像列の対応がおかしい | EV-M の明示入力順、SAM3.1 の自然順と全画像寸法を確認 |
| PVS / tracker が EV-M を拒否 | detector専用checkpointの仕様。追跡が必要ならSAM3.1を選ぶ |
| 既存出力先が拒否される | SAM3.1 の完了出力を保ち、新規出力先を指定 |
| CPU の結果が参照と異なる | 同じCHW入力とstage traceを比較。stb / Pillowのresize差を分離 |

## 9. オンボーディング完了チェックリスト

- [ ] モデル・backend・対応範囲を選んだ
- [ ] 作業ブランチ・既存変更を確認した
- [ ] 選んだ uv project と build target を構築した
- [ ] 固定取得元・SHA・変換照合を確認した
- [ ] headless 画像と画像列の PNG / JSON を確認した
- [ ] 境界・回帰検査と必要な実モデル E2E が成功した
- [ ] 計測条件と未検証範囲を記録した
- [ ] staged diff に認証情報・個人データ・新規大容量生成物がない

## 10. 参照と更新履歴

構成は関連リポジトリ FlashVSR / ZipMap の ONBOARDING の「入口・責務・環境・実行・検証・トラブル対応」を参考にしました。
実行内容は本リポジトリの CLI、CMake、uv lock、検証 JSON に照合しています。

- [公式 SAM3](https://github.com/facebookresearch/sam3): 演算順序とshapeの参照
- [EfficientSAM3 固定source](https://github.com/SimonZeng7108/efficientsam3/tree/bd0936c788fed8d51fa799437f05abd97b401b06): EV-M builder
- [対応する ONNX repo](https://github.com/yuki-inaho/sam3-video-tracking-onnx-export/blob/main/docs/ONBOARDING.md): CPU export / annotation UI
- [SAM1 C++移植](https://github.com/YavorGIvanov/sam.cpp): ggml graph構築の参考。使用APIは本repoのvendored例を優先

2026-10-01: SAM3.1 / EV-M のモデル別導線を統合。古い端末情報を除き、実モデルとfixture、追跡と独立検出、実測と保証の区別を更新。
