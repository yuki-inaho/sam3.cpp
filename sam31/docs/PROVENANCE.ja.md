# 入力・参照・権利表示

本版は `yuki-inaho/sam3.cpp` の develop commit `42c28f21a606e57e09848686de5c634c2a8caa35` を新規cloneし、`feat/sam31-convrot-gguf` で作成しました。

ユーザー提供 `sam3cpp_sam31_NATIVE_CPP_SMOKE.tar.zst` の `sam31/` のソースと TEST_ONLY fixture を導入し、実重みに対応する tensor aliases、INT8 embedding、独立uv環境・回帰テスト・ガイドを追加しました。提供アーカイブの過去テストログを今回の実行証跡として流用していません。

対象学習済みモデルは `ussoewwin/SAM3.1-ConvRot-INT8` revision `32c74f1ec2a9b615ddb9346d2155a999296c23b7`。ユーザー提供ファイルのSHA-256は配布元LFS値と一致しています。全変換照合は `evidence/conversion.json`、実推論は `evidence/real-image/` と `evidence/real-video/`、結果判定は `evidence/real-*-validation.json` です。重みをGitには含めません。

native graphの構造参照は [公式SAM](https://github.com/facebookresearch/sam3/tree/2345a4ad109ac29c569da749c91d84f10dc08c40)。regular-H4の数学定義は [量子化実装](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/native_convert_int8.py) を参照しています。ONNX export repo由来のCPU source patcherは任意のPython参照経路だけに含まれ、その由来はソース先頭に記載しています。公式Pythonソース全体は同梱していません。

[zerfoo/zonnx](https://github.com/zerfoo/zonnx) は参考です。本版はONNX RuntimeやGoを依存にしません。GGUFを読み取るggml、PNGを扱うstb、元repoのLICENSEと権利表示を保持しています。モデルは [モデルカードの利用条件](https://huggingface.co/ussoewwin/SAM3.1-ConvRot-INT8/blob/32c74f1ec2a9b615ddb9346d2155a999296c23b7/README.md) に従い、repoのMITライセンスの対象とみなしません。

C++ graphは通常推論で外部実行ファイルを呼びません。ggmlの異常終了時debugger機構について、提供アーカイブ側の変更を本版へ導入したという主張はしません。全人工回帰と学習済み画像・動画の検証結果は今回実行したものです。公式実装との全数値一致、自然動画での精度評価、GPU対応は未検証です。
