# 作業計画書 兼 記録書: SAM31 E2E・修正・CPU最適化

## 1. 作業目的

### 1.1 ゴール要求分析

レビューから再現可能な不具合を修正し、実重みの画像・動画E2Eを自動化する。推論の計測結果に基づきCPU実行時間を短縮する。単独作業、秘密情報なし、Pythonはuv、通常推論はC++のみ。

- TR-1: 複数点指定で公式のsingle mask/stability分岐を再現する。
- TR-2: 実学習済みGGUFで画像2対象・動画6フレーム2対象を検査するE2Eを追加する。
- TR-3: 同じ入力・重み・設定で計測し、マスク品質を維持して実行時間を短縮する。画像速度20%以上改善を目標とし、未達なら計測に戻る。
- TR-4: regression、手順、公開証跡、PR、zstd成果物を更新する。
- 範囲外: GPU、テキスト検出、途中の対象追加、公式モデル全出力の数値一致。
- リスク: thread設定による丸め差、環境に依存した性能、試験中の既存成果物上書き。
- 前提: named real GGUFがmodels/に存在し、CBLAS/OpenMP buildが可能。

### 1.2 サブゴールと追跡

| サブゴール | 要素 | 証跡 |
|---|---|---|
| SG-1/TR-1 | 公式構造照合、RED E2E、mask選択修正 | loader/CLI logs |
| SG-2/TR-2 | 取得済み実重みを指定するE2E runner | image/video report、IoU |
| SG-3/TR-3 | baseline、profile、実装最適化 | elapsed/比率/品質比較 |
| SG-4/TR-4 | 回帰、ガイド、梱包、PR | CTest、SHA、公開状態 |

## 2. 作業内容

調査→設計レビュー→RED→修正→計測→最適化→E2E→提出。境界不具合は狭いテストで固定し、時間の改善は実モデルで判断する。人工fixtureだけの速度を実モデル速度と扱わない。

## 3. チェックリスト

### 手順1: レビューとbaseline

- [x] 🖐 操作: 公式固定commitの関連forwardと現在のコードを照合し、BLAS threads=1/16を計測。
- [x] 🔎 確認: 時間embeddingは条件/非条件で公式と一致し、誤って修正しない。
- [x] 🧪 テスト: CTest既存5 groups、実画像2対象のbaselineとIoU。
- [x] 🛠 エラー時対処: 意図的な範囲外との差は不具合と混同しない。

### 手順2: mask選択RED

- [x] 🖐 操作: 制御fixtureで1点と2点・stable/unstableを区別するCLI E2Eを追加。
- [x] 🔎 確認: 現在版で複数点stable mask0が選択されず失敗。
- [x] 🧪 テスト: 改善対象の失敗ログを保存。
- [x] 🛠 エラー時対処: 公式dynamic stability設定を確認して期待値を独立に決める。

### 手順3: mask選択修正・整理

- [x] 🖐 操作: decoderの候補選択を読みやすい関数へ整理しsingle/stability分岐を実装。
- [x] 🔎 確認: 初期点数0..1はmultimask、2点以上はstable mask0またはfallback。
- [x] 🧪 テスト: REDがGREEN、既存回帰成功。
- [x] 🛠 エラー時対処: memory pointerも選択したtokenへ対応する。

### 手順4: 実モデルE2Eを追加

- [x] 🖐 操作: Python driverでC++ synthetic/image/videoと独立結果検査を連続実行。
- [x] 🔎 確認: 実重みを明示するopt-in gate。fixtureとの混同、PATHで推論外部委譲を拒否。
- [x] 🧪 テスト: images/video全14 masksのIoU>=0.8、finite、ID、memory blocks、SHAを検査。
- [x] 🛠 エラー時対処: 不完全・wrong model・不正結果はnonzeroで終了。

### 手順5: 計測に基づくCPU最適化

- [x] 🖐 操作: baseline/profileから支配的な演算・thread経路を特定し変更。
- [x] 🔎 確認: public threads optionとBLAS設定が矛盾しない。
- [x] 🧪 テスト: 実画像のbefore/after、同IoU、速度比。動画E2Eでも確認。
- [x] 🛠 エラー時対処: 品質低下・回帰・速度悪化なら変更を再検討し無理な成功を主張しない。

### 手順6: 回帰と納品

- [x] 🖐 操作: 全回帰とE2E、guide/evidence更新、commit/push/PR merge、zstd更新。
- [x] 🔎 確認: 書かれた結果が新しい実行証跡と一致、秘密・環境cache・実重みをGitに含めない。
- [x] 🧪 テスト: zstd/SHA/展開manifest、作業書のDoD監査。
- [x] 🛠 エラー時対処: 未証明項目は未完了のまま継続。

## 4. 使用コマンド

```sh
OPENBLAS_NUM_THREADS=1 ctest --test-dir build --output-on-failure
uv run --project sam31 python sam31/native/test_cli.py
cmake --build build --parallel 8
uv run --project sam31 python sam31/tools/run_native_e2e.py --help
```

CLI testはSAM31_BINARYを絶対パスで指定。E2E runnerのoptionは手順4で確定し、ガイドへ記録する。既存出力は拒否して新しいrun directoryを使う。性能は他の重いプロセスと同時に計測しない。

## 6. 完了の定義

- [x] D1: mask選択の公式差を修正しstable/unstable・1点/複数点E2E成功。
- [x] D2: 学習済みGGUFで全14 masks、IoU>=0.8、finite、stable IDs、memory20 blocks。
- [x] D3: 同条件の計測で画像20%以上短縮、品質維持、実行時間・設定を記録。
- [x] D4: CTest全成功、guide/evidence/PR/成果物一致、checksum成功。

## 7. 作業記録

開始前にdateで時刻を読み、項目完了直後に結果と証跡を記録する。エラーと解決を残す。未検証の結果をチェックしない。

| 日付・時刻 | 作業 | 結果 |
|---|---|---|
| 2026-10-01 09:34:48 JST+0900 | 新branch、レビューとbaseline開始 | fix/sam31-e2e-performance、既存CTest5/5成功。時間embedding疑義は公式と一致して解消。 |

## 8. 設計レビュー

PASS。TR1..4とD1..4を対応付け、E2E real/fixtureを明示。公式完全一致を今回の成功条件に入れず、既知forward不具合と品質維持を検査する。branch/working rootはsam3.cpp、すべてPython操作はuv。公開データのみを記録する。

| 2026-10-01 09:42:09 JST+0900 | 手順1完了 | 旧44秒、新profile44.872秒、BLAS1は74.432秒。時間embeddingは公式と一致。CTest5/5。 |

| 2026-10-01 09:42:09 JST+0900 | 手順2完了 | review-selection-red.log: 2点stable mask0 quality1000期待、実際100で失敗。 |

| 2026-10-01 09:42:09 JST+0900 | 手順3完了 | review-selection-green.log: stable単一候補とunstable fallback成功。mask selection/pointer対応を整理。 |

| 2026-10-01 09:42:09 JST+0900 | 手順4/5開始 | E2E runner追加・fixture拒否、profile集計、exact GELUとnormの独立行を並列化。実計測中。 |

| 2026-10-01 09:45:40 JST+0900 | 手順5完了 | review-performance.json: 44.872→35.354秒、21.21%短縮、raw logits byte-exact、両IoU1.0。正規化2243→580ms。 |

| 2026-10-01 09:49:16 JST+0900 | 手順4/5 E2E完了 | real GGUF SHA確認、全14 masks IoU1.0、image34965ms/video250482ms、memory20 blocks。 |

| 2026-10-01 09:49:28 JST+0900 | 手順6検査完了・公開準備 | CTest5/5、実E2E成功、source展開SHA照合、.gitなしの再梱包も2285 files成功。公開結果は実行後に追記。 |

| 2026-10-01 09:51:30 JST+0900 | 手順6公開完了 | commit 1e500cf push、PR#2 develop merge済み。全D1..D4達成。 |

## 追加作業：EfficientSAM3 EV-M

ユーザー追加要求により、SAM31 の完了後は ONNX と C++/ggml 両リポジトリで EfficientSAM3 EV-M を対応する。実重みの構成確認から始め、画像・複数画像の実モデル比較、再現手順、commit/push/PR/merge までを続ける。詳細の正本は [EfficientSAM3 作業計画書](workdoc_Oct01-2026_efficientsam3_ev_m.md)。write-workdoc-uv と review-written-workdoc を適用し、今回の未完了を SAM31 完了状態と混同しない。
