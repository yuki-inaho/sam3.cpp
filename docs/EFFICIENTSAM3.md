# EfficientSAM3 EV-M：公開 checkpoint → GGUF → native ggml

EfficientSAM3 の Stage3 **EV-M** を画像・明示した順序の画像列に適用する。
推論は C++14、vendored ggml、stb のみ。Python/PyTorch は取得・変換・参照検証に使い、C++ の推論時には起動しない。
公開重みには tracker がなく、画像列はフレーム独立検出。SAM3.1 の memory tracking と区別する。

## 固定構成

| 項目 | 値 |
| --- | --- |
| Source | [SimonZeng7108/efficientsam3](https://github.com/SimonZeng7108/efficientsam3/tree/bd0936c788fed8d51fa799437f05abd97b401b06) |
| HF | [Simon7108528/EfficientSAM3](https://huggingface.co/Simon7108528/EfficientSAM3/tree/85b05896928f974e308f889d7ccb2eefc069de98/efficientsam3_ft) |
| ファイル | `efficientsam3_efficientvit.pt` |
| SHA256 | `086b04b2e7da7cc98aa4621b70c7291608aa9d187357b98d03bd4d6533ed5a17` |
| Vision | EfficientViT b1、幅16/32/64/128/256、projection→1024、bilinear→72×72 |
| Text | MobileCLIP-S0、512幅、8heads、前後RepMixer＋4 transformer blocks、非causal、context16 |
| PCS | 既存SAM3 geometry/fusion/DETR/segmentation head |
| 実測parameters | 97,435,030（公開README表89.2Mとは異なる） |
| 保存 | 799 tensors、全model payload 389,822,768 bytes |

## リポジトリのルートから実行

uvとCMake/C++コンパイラを用意する。CPU専用Python環境を追加機能のディレクトリに作る。

```sh
uv sync --project efficientsam3 --locked
uv run --project efficientsam3 python efficientsam3/fetch.py
uv run --project efficientsam3 python convert_efficientsam3_to_gguf.py
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target efficientsam3 --parallel 16
mkdir -p output/efficient
./build/examples/efficientsam3 --model models/efficientsam3_ev_m.gguf --image outputs/efficientsam3/source/sam3/assets/dog_person.jpeg --text dog --output output/efficient
```

取得先とrevisionは固定。source cacheは `outputs/efficientsam3/source`、入力重みは `models/efficientsam3_ev_m.pt`、変換後は同ディレクトリの `.gguf`。
source/modelが既にある場合は `--source` / `--checkpoint` / `--output` で場所を明示できる。SHA不一致や不足/shape不一致を拒否する。

変換器はraw checkpointの `model` dictionaryをstrictロードし、799 tensorsの値を再量子化せず保存する。
scalarは1要素vector、geometryの7×7pool weightは等価reshape。それぞれpayload bytesをGGUFから読み直し全件照合する。
trainer metadata、認証情報、利用者の絶対パスは保存しない。GGUFへBPE vocab/mergesと公開モデルidentityを同梱する。

## 画像列annotation

`--image` を入力順に繰り返す。一つのC++プロセスがmodelを保持し、各画像を順に処理する。

```sh
mkdir -p output/efficient-sequence
./build/examples/efficientsam3 --model models/efficientsam3_ev_m.gguf --text dog --image frame0.png --image frame1.png --image frame2.png --output output/efficient-sequence --threads 16 --threshold 0.5
```

出力はmask PNGと `annotations.json`（frame index、score、pixel box、時間）。フレームをまたぐ物体IDを生成する経路ではない。
対象variantは固定EV-M、CPU backend。PVS/メモリtracker APIはこのモデルを明示拒否する。

## 実装・検証

`sam3.cpp` 内の独立したggml stageとして学生encoderを処理する。各段階は新しいcontext/graph/gallocrと入力tensorを持ち、CPU vectorを介して次へ渡す。
F32重みを保存し、CPU用F32 depthwise kernelを使用する。BatchNormのaffine係数をロード時に事前計算する。
通常の1×1 convolutionはmatrix multiplyで処理し、im2col bufferを省く。grouped projectionは各groupのconvolutionを使う。

```sh
uv run --project efficientsam3 pytest tests/test_efficientsam3_conversion.py tests/test_efficient_cli.py -q
uv run --project efficientsam3 python efficientsam3/e2e.py
EFFICIENTSAM3_REAL=1 uv run --project efficientsam3 pytest tests/test_efficient_e2e.py -q
```

モデル無し検査は不正GGUF、数値引数、変換境界を検査する。実モデルE2Eは公開dog画像を1008へ前処理し、人工的に移動した6フレームを通常CLIで推論する。
参照PyTorchと非空mask/scoreを比較しmask IoU >=0.90を要求する。C++ subprocessのPATHは実行ファイルのない `/NO_EXECUTABLES` とする。
レポートの保存先は `evidence/efficient-e2e/e2e.json`。実測値は `docs/efficientsam3-validation/` に保存する。

細かい数値調査には `--preprocessed input.f32` を使える（CHW/F32、3×1008×1008、画像1枚）。この経路では前処理差を除いてencoderを比較する。
`SAM3_EFFICIENT_TRACE_DIR` へ既存ディレクトリを指定すると学生stageとFPNのraw出力を保存する。`SAM3_EFFICIENT_PROFILE=1` でstage時間をstderrへ表示する。
通常の画像入力はstb decodeとC++ bilinear resizeを使う。ORT側Pillow resizeとの前処理差をモデル演算誤差と混同しない。

同モデルのONNX export/runtimeは [sam3-video-tracking-onnx-export](https://github.com/yuki-inaho/sam3-video-tracking-onnx-export) の `docs/EFFICIENTSAM3.md` を参照する。モデル/source/context/前処理/sequence_modeの対応を上記と統一している。

## 実測結果（2026-10-01）

- [変換照合](efficientsam3-validation/conversion.json)：799 tensors / 389,822,768 payload bytesが完全一致。
- [native E2E](efficientsam3-validation/native-e2e.json)：同一プロセスの6フレームで最小mask IoU **0.999962**、約10.88–12.16秒/フレーム（16threads）。
- [同じCHW入力でのstage比較](efficientsam3-validation/stage-comparison.json)：text最大誤差0.0000026、最終mask IoU **0.999993**。
- [性能記録](efficientsam3-validation/performance.json)：同じ1フレームを16threadsで10.95秒、4threadsで9.75秒。両PNGはbyte一致。各条件1回の計測であり、性能保証ではない。

公開dog画像とその人工的な移動を検証入力に使用した。モデル無しの境界検査は12件、既存SAM31のCTestは5グループ成功。
LiteMLAのgrouped projectionをbatched GEMMへ変更した案は実測が遅く、採用しなかった。

## zstd sourceの作成と検査

Gitで追跡したsource/docs/testのみを、利用者名を含まないtar headerと相対SHA256一覧へ梱包する。
models、build、outputs、環境cacheはsource archiveへ含めない。

```sh
uv run --project efficientsam3 python scripts/package_efficientsam3.py --name sam3cpp-efficient --output dist/sam3cpp_efficientsam3_source.tar.zst
zstd -t dist/sam3cpp_efficientsam3_source.tar.zst
mkdir -p restore
tar --zstd -xf dist/sam3cpp_efficientsam3_source.tar.zst -C restore
(cd restore/sam3cpp-efficient && sha256sum -c FILES.sha256)
zstd -T4 -3 models/efficientsam3_ev_m.gguf -o dist/efficientsam3_ev_m.gguf.zst
```
