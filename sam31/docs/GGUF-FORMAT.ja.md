# GGUF保存形式とConvRotの定義

## 保存契約

GGUF version 3、リトルエンディアン、32バイト境界、`general.architecture=sam3_1_multiplex`、`sam31.storage_version=1` とします。これはこの追加実装が定義した保存形式であり、SAM 3.1の公認・汎用的なGGUF実行仕様を主張するものではありません。

変換時には元SafeTensorsの全テンソルを保持し、重みの値、FP16/BF16、尺度、U8の量子化説明を再計算しません。`int8_tensorwise` として量子化された重みを `Q8_0` とみなしたり、再量子化したりしません。

| 元の型 | GGUFのテンソル型番号 | 補足 |
|---|---:|---|
| F32 / F16 | 0 / 1 | バイト列を保存 |
| I8 / U8 / BOOL | 24 | 物理的にはI8。論理型は対応表で保存 |
| I16 / I32 / I64 | 25 / 26 / 27 | バイト列を保存 |
| F64 / BF16 | 28 / 30 | バイト列を保存 |

ggmlのテンソル名の長さ制約に合わせ、物理名を `t_` + 元の名前のSHA-256の先頭60桁とします。`sam31.tensor_map` に物理名から元の名前・論理型・元の形状への対応表をJSON文字列で保存します。衝突があれば変換を拒否します。元の名前を切り詰めて対応関係を失う方式ではありません。

GGUFの軸表記はggmlの順序に反転しますが、ペイロードの並びは変更しません。スカラーは物理形状 `[1]`、5次元以上は物理的に1次元に格納し、元の形状を対応表から復元します。SAM 3.1で5次元重みが存在することを主張するものではなく、保存形式の一般性のための取り扱いです。

主な追加メタデータは、`sam31.source.sha256`、`sam31.source.header`、`sam31.source.metadata`、`sam31.tensor_map`、`sam31.test_fixture`、`sam31.convrot_basis=regular-h4-kron`、INT8層数、回転対象層数です。元ファイルの名前とハッシュも保持します。テンソルごとに読み出して書くため、変換時に学習済みモデル全体をPyTorchに展開しません。

## 量子化の取り扱い

対象層の `.weight` はI8、`.weight_scale` は浮動小数点、`.comfy_quant` はJSONを格納したU8/I8テンソルです。今回の実装が受理するJSONキーは `format`、`convrot`、`convrot_groupsize` です。未知の方式・追加意味を無視して進むことはありません。尺度はスカラーまたは出力チャネルごとで、0・負値・NaN・無限大を拒否します。

regular-H4の基底は以下です。

```text
H4 = 1/2 * [ 1  1  1 -1
             1  1 -1  1
             1 -1  1  1
            -1  1  1  1 ]
```

群サイズは4のべき乗で、入力チャネル数を割り切る必要があります。群サイズ16/64/256等では、この行列のKronecker積を用います。Hは対称・直交であり、元の量子化コードの `W_rot = W H^T` に対応する復元は `W_restored = (Q * scale) H` です。通常のSylvester基底に置き換えることはできません。Python・C++ではradix-four変換を独立に実装し、試験では別に生成した明示行列と比較します。

Conv2dの形状 `[出力, 入力, kH, kW]` では**入力チャネル軸だけ**を群ごとに回転します。`入力 × kH × kW` を一括して平坦化し回転すると別の演算になるため、採用していません。

`sam31_linear_w8a32` は入力側を回転してI8重みと積和し、出力別尺度とbiasを適用するCPU部品です。全モデルのグラフには未接続です。C APIの配列は行優先、長さは要素数、入出力バッファの重なりは禁止です。群サイズ0は回転なし、その他は4以上の4のべき乗です。成功0、不正入力1、メモリ確保失敗2を返します。

## 参照根拠

対象モデルのカード：
https://huggingface.co/ussoewwin/SAM3.1-ConvRot-INT8/blob/main/README.md

量子化側の実装。モデルカードの基底表現だけでなく、こちらのregular-H4行列と入力軸の定義に合わせました：
https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/native_convert_int8.py

標準GGUFの型・ヘッダーの読み取りは、ユーザー添付の `ggml/include/gguf.h` と `ggml/src/gguf.cpp` によります。C++の検査も同じggmlライブラリで実行しました。

参考に指定されたzonnxは、今回のSAM 3.1実行処理には組み込んでいません。重みの保存形式への変換と、複数物体追跡グラフの実装は別の要件です：
https://github.com/zerfoo/zonnx

指定実モデルを取得し、3,136 tensors・651 INT8 layers・647 ConvRot layersの全payload変換一致を確認しました。現在の手順と証跡は `docs/SAM31_GUIDE.ja.md` と `evidence/` を参照してください。
