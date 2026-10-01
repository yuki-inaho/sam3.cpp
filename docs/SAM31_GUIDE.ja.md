# SAM 3.1 ConvRot INT8 を GGUF で動かす

## 対象と実行経路

`ussoewwin/SAM3.1-ConvRot-INT8` の `sam3.1_multiplex_convrot_int8.safetensors` を対象とします。変換は Python/NumPy、画像・動画推論は **C++ のみ**です。通常の `sam31`、`sam31_image`、`sam31_video` は Python/PyTorch/ONNX Runtime を呼びません。

GGUF は保存形式です。この版では元の INT8 値・尺度・回転説明を保存し、C++ 内で逆 ConvRot と FP32 復元を行います。計算は CPU FP32 / CBLAS。GGML Q8_0 への再量子化や GPU INT8 演算ではありません。ggml は GGUF 読み取りに使用し、SAM 3.1 グラフは `sam31/native/` に独立した C++17 module として実装しています。既存 SAM3 の C++14 API は別 target です。

対応範囲は、初期画像で点指定する 1～16 対象の画像分割と、同じ対象を前向きに追跡する画像列です。テキスト・矩形・マスク入力、途中の対象追加削除、逆方向追跡、GPU は未対応です。

## 1. 取得と環境

以下はリポジトリルートで実行します。Linux、CMake 3.16以上、C++17 compiler、OpenBLAS、uv、zstd が必要です。Ubuntu の例：

```sh
sudo apt-get install -y build-essential cmake libopenblas-dev zstd
uv sync --project sam31 --python 3.11 --locked
```

Python の依存は独立した `sam31/pyproject.toml` に固定しています。root project の torch 環境は不要です。

取得元の利用条件を確認し、必要な認証は `HF_TOKEN` 環境変数で設定してください。トークン値をコマンド・ドキュメント・ログへ書かないでください。

```sh
uv run --project sam31 python sam31/tools/fetch_assets.py --models-dir models
```

このコマンドは固定リビジョン `32c74f1ec2a9b615ddb9346d2155a999296c23b7` の配布メタデータとファイルサイズ・SHA-256 を照合し、既存ファイルの上書きを拒否します。すでに取得したファイルがあれば `models/sam3.1_multiplex_convrot_int8.safetensors` に配置し、次の値と比較します。

```sh
sha256sum models/sam3.1_multiplex_convrot_int8.safetensors
# 21fc2308ef4bf82a1d372e147170e2d48ed8f3688922b4eaa3e341828333debd
```

## 2. SafeTensors → GGUF

```sh
uv run --project sam31 python sam31/tools/convert_sam31_to_gguf.py \
  --input models/sam3.1_multiplex_convrot_int8.safetensors \
  --output models/sam3.1_multiplex_convrot_int8.gguf --verify
```

`roundtrip.exact=true` は全テンソルの dtype・shape・payload が元と一致したことを示します。今回の実重みは 3,136 tensors、906,735,111 payload bytes、651 INT8 layers、647 ConvRot layers。再量子化はありません。生成 GGUF の SHA-256 は `e59728d18a3e0f6bfd603eee86490b6116b9141d4e4cd3acdc0309ea443ec370` です。

同名出力は拒否します。明示的に再生成する場合だけ `--overwrite` を付けます。`TEST_ONLY` fixture は未学習で、別途 `--allow-test-fixture` が必要です。

## 3. ビルドと読み取り検査

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DSAM3_BUILD_SAM31=ON -DSAM3_METAL=OFF -DSAM31_BUILD_REFERENCE=OFF \
  -DPython3_EXECUTABLE="$PWD/sam31/.venv/bin/python"
cmake --build build --parallel 8
build/sam31/sam31 inspect models/sam3.1_multiplex_convrot_int8.gguf --read-all
```

configure の `SAM31 native: CBLAS acceleration enabled` を確認します。OpenBLAS を後から入れた場合は `cmake -S . -B build -U SAM31_HAVE_CBLAS_H` で検出キャッシュを更新します。CBLAS がない場合も build は可能ですが、実解像度の CPU 推論は大幅に遅くなります。

`inspect` は GGUF 全 payload の読み取り検査です。`native_model_validated=false` は inspect が推論グラフを実行していないことを示します。次の image/video が実際の graph validation です。

## 4. 人工入力を作り、実モデルで画像 annotation

```sh
build/sam31/sam31 synthetic --output runs/synthetic --frames 6
OPENBLAS_NUM_THREADS=16 build/sam31/sam31_image \
  --model models/sam3.1_multiplex_convrot_int8.gguf \
  --image runs/synthetic/image.png \
  --point 1:0.25:0.32 --point 2:0.73:0.68 \
  --threads 16 --cache-mb 4096 --output runs/image
uv run --project sam31 python sam31/tools/verify_synthetic_results.py \
  --output runs/image --inputs runs/synthetic --frames 1
```

`--point ID:X:Y[:LABEL]` は正規化座標 [0,1] です。LABEL は正例1（省略時）または負例0。同じ ID に複数の点を指定できます。ID は正の64ビット整数で、JSONでは文字列です。

ここで人工なのは入力画像です。重みは指定された学習済み SAM 3.1 です。検証ツールは推論後に入力色から正解を別途計算し、有限 logits、PNGとの対応、IoU≥0.8 を検査します。正解マスクを推論に渡していません。

## 5. 動画（連続画像）annotation

```sh
OPENBLAS_NUM_THREADS=16 build/sam31/sam31_video \
  --model models/sam3.1_multiplex_convrot_int8.gguf \
  --frames runs/synthetic/frames \
  --point 1:0.25:0.32 --point 2:0.73:0.68 \
  --threads 16 --cache-mb 4096 --output runs/video
uv run --project sam31 python sam31/tools/verify_synthetic_results.py \
  --output runs/video --inputs runs/synthetic --frames 6
```

初期フレームだけに点を指定し、その後は各フレームの画像と過去の memory・object pointer を使います。入力は2枚以上、同じ寸法の PNG/JPEG 画像列です。ファイル名は自然順（1,2,10）で並びます。MP4からの切り出しは別途 ffmpeg で行います。

画像経路は608 parameters、動画経路全体は867 parametersを使用します。残りの detector/text tensorsも GGUF 内に保持しますが、この点追跡経路では使いません。6フレームでは32層ViTを6回、memory attentionを5回（4層×5=20 blocks）実行します。

## 出力

各出力ディレクトリには次を保存します。

- `report.json`: model/source SHA、ID、フレーム順、演算回数、マスク統計、実行時間。
- `frame_000000_obj_1.png` 等: 入力画像と同じ寸法、背景0/前景255。
- `frame_000000.f32` 等: little-endian FP32 `[objects,height,width]` の logits。値>0 が前景。

出力先の再使用は拒否します。途中エラーでは staging directory を削除し、成功していない結果を完成出力として公開しません。

report内の `trained_checkpoint_verified=false` / `accuracy_validation_performed=false` は、汎用 C++ runtime 自体は配布元への照合や正解マスク比較を行わないという意味です。今回の配布元照合は `evidence/hf-metadata.json` と `conversion.json`、精度の smoke 検査は `real-*-validation.json` に別記録しています。一般画像の精度保証や公式 PyTorch との数値一致はこの人工入力検査からは主張しません。

## 6. 回帰・人工重み smoke

```sh
OPENBLAS_NUM_THREADS=1 ctest --test-dir build --output-on-failure
OPENBLAS_NUM_THREADS=1 sh sam31/tools/smoke_native.sh build runs/fixture-smoke
```

同梱 `TEST_ONLY_native_sam31.gguf` は未学習・縮小寸法の演算試験用です。上の実モデル経路とは区別してください。C++ quant/graph/operator、CLI・storage異常系、入力・点・memory介入、16対象、リセット再現性を検査します。Python は test driver と変換処理にだけ使用します。

## 7. zstd 成果物の復元

ソースと GGUF は別ファイルです。配布ディレクトリで checksum を確認して展開します。

```sh
sha256sum -c SHA256SUMS
zstd -t sam3cpp_sam31_source.tar.zst sam3.1_multiplex_convrot_int8.gguf.zst
tar --zstd -xf sam3cpp_sam31_source.tar.zst
cd sam3cpp-sam31
mkdir -p models
zstd -d ../sam3.1_multiplex_convrot_int8.gguf.zst \
  -o models/sam3.1_multiplex_convrot_int8.gguf
```

続いてこの文書の環境・build・image/videoを実行します。GGUFからの推論に元 SafeTensors は不要です。モデルのライセンスはモデルカードを参照し、リポジトリの MIT と区別します。

## 参照

- [対象モデルと利用条件](https://huggingface.co/ussoewwin/SAM3.1-ConvRot-INT8/tree/32c74f1ec2a9b615ddb9346d2155a999296c23b7)
- [SAM 3 公式実装の固定コミット](https://github.com/facebookresearch/sam3/tree/2345a4ad109ac29c569da749c91d84f10dc08c40)
- [ConvRot量子化実装](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/native_convert_int8.py): モデルカードにあるSylvesterという表現とは異なり、実装のregular-H4基底に合わせています。
- [追加参考のONNX export](https://github.com/yuki-inaho/sam3-video-tracking-onnx-export)
- [参考 zonnx](https://github.com/zerfoo/zonnx): 本版の依存には含めていません。

## 今回の実測結果

| 検証 | 結果 |
|---|---|
| 実重み→GGUF | 全3,136 tensors、906,735,111 payload bytesが完全一致 |
| 実モデル画像 | 80×64入力、2対象、2マスク、両IoU 1.0、約43.9秒 |
| 実モデル動画 | 6フレーム、2対象、12マスク、全IoU 1.0、約312秒 |
| memory | 5回のattention、4層×5=20 blocks、6フレームの状態保持 |
| 回帰 | CTest全5グループ成功、実checkpoint aliases/INT8 embeddingも検査 |

CPU16 threads・CBLAS・1008×1008内部解像度での一回の実測です。予測画像は [動画overlay](sam31-validation/real-video-overlay.png)、数値検証は [画像](sam31-validation/real-image-validation.json) と [動画](sam31-validation/real-video-validation.json) を参照してください。`quality` は復号器の生出力で、確率 [0,1] として解釈しません。
