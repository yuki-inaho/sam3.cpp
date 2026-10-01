# 検査用データ

`TEST_ONLY_convrot_tensors.safetensors` と `.gguf` は、乱数と定数から生成した22テンソルの検査専用データです。学習済みSAM 3.1ではなく、画像や動画を推論できません。通常の推論入口はこのデータを拒否します。

`synthetic_inputs/` は640×384の2物体・6フレームの人工入力です。赤い円と青い四角が移動します。`ground_truth/` と `ground_truth.npz` は入力生成時の正解であり、モデルの予測ではありません。この提出物にはSAM 3.1で生成した予測マスクはありません。

再生成には `sam31/tools/make_test_fixture.py --output ...`、`sam31/tools/convert_sam31_to_gguf.py --allow-test-fixture ...`、および `sam31 synthetic --output ...` を使用します。元ファイルや既存出力は上書きしません。
