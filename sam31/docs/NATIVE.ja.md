# 純C++実装

実行手順と現在の検証範囲は [SAM31ガイド](../../docs/SAM31_GUIDE.ja.md) を参照してください。

| ファイル | 責務 |
|---|---|
| `include/sam31_native.h` | Model/Session公開API |
| `native/weights.cpp` | 867必須parameters、dtype/shape/aliases検査、INT8とembeddingの復元cache |
| `native/ops.cpp` | FP32行列積・畳み込み・attention・RoPE・補間 |
| `native/graph.cpp` | 32層ViT、TriHead neck、interactive/multiplex decoder、4層memory attentionと状態更新 |
| `native/main.cpp`, `native/io.cpp` | CLI、stb画像入出力、atomic output |
| `src/gguf.cpp`, `src/quant.cpp` | ggml GGUF読み取りとregular-H4逆変換 |

保存は元INT8・尺度・回転説明を維持し、演算時にFP32復元します。C++ native graphはggml computation graphではなく、独立したeager CPU graphです。`SAM31_BUILD_REFERENCE=OFF` が既定で、Python参照launcherは通常targetへリンクしません。

未学習fixtureは入力16×16・内部幅32に縮小し、32/4/2層数・16slotsを維持します。実モデルは1008×1008・ViT幅1024、32層、memory幅256です。人工fixtureの成功だけを実モデル成功とみなさず、今回の学習済みcheckpoint結果を `evidence/` に記録しています。
