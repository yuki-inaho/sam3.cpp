# 作業計画書 兼 記録書: SAM 3.1 ConvRot INT8 / GGUF / native C++

**日付:** 2026年10月01日（開始 08:45:53 JST+0900）
**作業ディレクトリ・リポジトリ:** `sam3.cpp` 新規クローン。以下のコマンドはルートで実行。
**作業者:** Codex 単独。サブエージェントを使用しない。
**ブランチ:** `feat/sam31-convrot-gguf`。ベース develop `42c28f21a606e57e09848686de5c634c2a8caa35`。

## 1. 作業目的

### 1.1 ゴール要求分析

ユーザーが指定した学習済み `sam3.1_multiplex_convrot_int8.safetensors` をGGUFへ変換し、`sam3.cpp` 上のC++画像分割・連続画像追跡で実行する。ダウンロード、変換、ビルド、画像・動画実行、検証を自己完結した文書にまとめ、ソースと証跡をzstdで納品する。

- 明示要求: 新規クローン・ブランチ、指定実モデル、GGUF変換、Pythonへの推論委譲なし、人工画像と複数画像のsmoke test、write/reviewスキルで作業書作成、DoD達成まで実行、zstd成果物。
- 入力: Downloadsに配置済みの `sam3.1_multiplex_convrot_int8.safetensors` と `sam3cpp_sam31_NATIVE_CPP_SMOKE.tar.zst`。後者はソース・人工重み・過去証跡を含む提供物で、実モデルの証拠とは扱わない。
- 暗黙制約: トークン値・個人環境を公開ソースに含めない。Python作業はuv。再量子化せずINT8・scale・descriptorを保持。欠損テンソル、非対応入力は明示的に失敗。人工重みや正解マスクへの置換は禁止。
- 非ゴール: テキスト検出、GPU、高速INT8カーネル、公式API全機能の移植は今回要求に含まれない。推論はCPU FP32復元経路とし、この制限を記載する。
- 成功条件: 実重み3136テンソルのdtype/shape/payload一致、C++が必要パラメーターを全て使用し、人工画像1枚・6フレーム・2対象で有限な出力とPNGを保存。依存・コマンドが同一の文書に揃い、展開した納品物でもsmoke test成功。
- 前提・リスク: アーカイブ内実装は縮小人工重みのみ検証済み。学習済みファイルの `tracker.` alias、量子化embedding、実寸法、性能、公式演算との差は未確認。調査→失敗再現→修正→実モデル再実行の順で解決する。

### 1.2 サブゴール構造

| ID | サブゴール | 成果物 | 検証 |
|---|---|---|---|
| SG-1 / TR-1 | 入力とベースの確定 | provenance、SHA-256、ブランチ | Git状態、モデルheader、提供物一覧 |
| SG-2 / TR-2 | lossless GGUF変換 | convertスクリプト、GGUF | 全payload byte比較、C++ read-all |
| SG-3 / TR-3 | native C++実モデル推論 | sam31ライブラリ・CLI | 実モデル画像/6フレーム、有限logits、全パラメーター使用 |
| SG-4 / TR-4 | 人工データsmokeと回帰 | tests、evidence | fixture CTest、実重みsmoke、入力・prompt・memory依存 |
| SG-5 / TR-5 | 一貫した手順・納品 | README、作業記録、tar.zst、checksum | 新規展開で再ビルド/再実行、zstd検査 |

### 1.3 トレーサビリティ方針

各手順にTRを付与し、実行前後の時刻と結果を第7章へ即時記録する。ログは `evidence/` に保存。過去アーカイブのログを今回の成功証跡に流用しない。実モデルとTEST_ONLY人工モデルの結果は別ディレクトリ。DoDの各項目は現在の出力を読み直してチェックする。

## 2. 作業内容

### フェーズ1: 調査

TR-1/2/3: CLAUDE.md・PLAN.md、develop、提供ソース `sam31/native`、GGUF format/quant、実モデルheaderを読む。zonnxの対応範囲を調べ、独自ConvRot descriptor保持が必要か判断する。ここでは実装を変更しない。

### フェーズ2: 設計・レビュー

TR-2/3/5: 提供物の独立 `sam31/` を既存SAM3の追加モジュールとして導入する。既存C++14ライブラリは保持。提供実装のC++17 filesystem/エラー処理を追加モジュールに限定する理由を文書化する。計算グラフは提供native CPU演算で、ggmlはGGUF読取に使う。公式数値再現の未確認箇所をsmoke成功と混同しない。作業書をreview-written-workdoc rubricで評価し、Major/Blockerを修正してから実装する。

### フェーズ3: 実装

TR-2/3/4: 提供コード・fixtureを選択導入、独立uv環境を追加。実モデルで観測したalias/量子化/shape問題の回帰テストを先に追加し、失敗→成功を記録。必要な計算の修正は独立演算期待値で検証する。

### フェーズ4: 検証

TR-2/3/4: CMake build、CTest、Python storage tests、lossless conversion、実モデルimage/video、有限値/マスク数/ID/状態依存を確認。演算修正には対象テスト、必要ならsanitizerを実行する。モデル精度の主張には画像上の観察と妥当性が必要で、smokeだけで精度一致とは言わない。

### フェーズ5: 記録・納品

TR-5: 新しい日本語ガイド、provenance、機械可読status、checksum、zstd source bundleを作る。現在の実重みGGUFも別zstdとして納品する。公開Gitに実重みをコミットしない。過去タスクのディレクトリ削除指示を新規成果物に自動適用しない。

## 3. 作業チェックリスト

### フェーズ1: 調査

### 手順 1: 入力・ブランチを確定 (TR-1)
- [x] 🖐 **操作**: developをcloneして `feat/sam31-convrot-gguf` を作成する。
- [x] 🔎 **確認**: ベースは42c28f21、指定SafeTensorsとtar.zstが存在する。
- [x] 🧪 **テスト**: 調査のためTDD不要。Git HEADとファイルサイズを確認。
- [x] 🛠 **エラー時対処**: bwrap障害は実行設定更新後に解消。ユーザートークンは出力しない。

### 手順 2: 提供実装と実headerを調査 (TR-1/2/3)
- [x] 🖐 **操作**: GGUF converter、native weights/graph、SafeTensors headerを読み差分を記録する。
- [x] 🔎 **確認**: dtype・prefix・寸法・descriptorと未検証点が第7章にある。
- [x] 🧪 **テスト**: 実header3136 tensors、INT8 651、F16 1183を確認しfixtureと区別する。
- [x] 🛠 **エラー時対処**: 不足や曖昧aliasは明示エラーとして回帰対象にする。

### フェーズ2: 設計・レビュー

### 手順 3: 作業書をレビュー (TR-5)
- [x] 🖐 **操作**: review rubricで本書を全章レビューし `temp/workdoc_review.md` を保存する。
- [x] 🔎 **確認**: 修正後Verdict PASS、残存Blocker/Majorなし。
- [x] 🧪 **テスト**: 各手順の4行、Trace/DoD対応、具体的なコマンドを検査。
- [x] 🛠 **エラー時対処**: 実装不確定点は調査手順として明記し、成功結果を捏造しない。

### フェーズ3: 実装

### 手順 4: 追加moduleを導入 (TR-3)
- [x] 🖐 **操作**: 提供アーカイブのsam31ソースとfixtureのみ追加しroot CMakeにsam31 optionを接続する。
- [x] 🔎 **確認**: 既存SAM3 APIに変更なくsam31 targetが定義される。
- [x] 🧪 **テスト**: CMake configureで新targetが未存在→存在へ変わる。
- [x] 🛠 **エラー時対処**: C++17は追加moduleだけに設定、BLAS未発見は明示表示する。

### 手順 5: uv変換環境を追加 (TR-2)
- [x] 🖐 **操作**: `sam31/pyproject.toml` を追加しuv lockを生成する。
- [x] 🔎 **確認**: numpy/Pillow/pytestだけで変換・storage testが動く。torch不要。
- [x] 🧪 **テスト**: `uv run --project sam31 python sam31/tools/convert_sam31_to_gguf.py --help`。
- [x] 🛠 **エラー時対処**: rootのtorch環境を暗黙使用せず独立projectを指定する。

### 手順 6: 実モデル互換の失敗を固定 (TR-3/4)
- [x] 🖐 **操作**: 実headerに基づくprefix/量子化embeddingのテストを追加する。
- [x] 🔎 **確認**: 導入版で失敗するケースと理由のログがある。
- [x] 🧪 **テスト**: native CLI/manifest境界testを実行して初期失敗を保存。
- [x] 🛠 **エラー時対処**: 必須パラメーター不足を無視せずalias候補とshapeを調べる。

### 手順 7: 実モデル互換を修正 (TR-3)
- [x] 🖐 **操作**: `sam31/native/weights.cpp` のaliasと量子化parameter条件を実headerへ対応する。
- [x] 🔎 **確認**: 全必須specが正しいdtype/shapeへmappingされる。
- [x] 🧪 **テスト**: 手順6のテストが成功、元fixture regressionも成功。
- [x] 🛠 **エラー時対処**: 追加shape差は公式実装と照合して修正し、推測でreshapeしない。

### フェーズ4: 検証

### 手順 8: Release build (TR-3)
- [x] 🖐 **操作**: `cmake --build build --parallel 8`。
- [x] 🔎 **確認**: sam31/image/videoと全テストがリンク成功。
- [x] 🧪 **テスト**: build logとexit codeを保存。
- [x] 🛠 **エラー時対処**: compiler/API差を最小箇所で修正、build cacheは必要時だけ分離。

### 手順 9: 人工重みnative smoke (TR-4)
- [x] 🖐 **操作**: `sh sam31/tools/smoke_native.sh build evidence/fixture-smoke`。
- [x] 🔎 **確認**: 画像2マスク・動画12マスク、有限logits、memory更新。
- [x] 🧪 **テスト**: C++ native/quant/operatorテストも全成功。
- [x] 🛠 **エラー時対処**: fixture成功を実モデル成功と扱わない。出力先再使用は拒否する。

### 手順 10: 実重みをGGUF変換 (TR-2)
- [x] 🖐 **操作**: 第4章のconvertコマンドを `--verify` で実行する。
- [x] 🔎 **確認**: 3136 tensorsのdtype/shape/payload一致、numerical_requantization=false。
- [x] 🧪 **テスト**: roundtrip exact=true、C++ inspect read-all成功。
- [x] 🛠 **エラー時対処**: descriptor不正や元ファイル変更は失敗させ、モデルを代替しない。

### 手順 11: 実重みimage smoke (TR-3/4)
- [x] 🖐 **操作**: 第4章のimageコマンドを実モデルGGUFで実行する。
- [x] 🔎 **確認**: test_fixture=false、2 IDs、2 PNG、finite f32、image経路608 tensors使用（全867はvideoで検証）。
- [x] 🧪 **テスト**: reportとPNGを開き、寸法・非空mask・クリック対象との対応を確認。
- [x] 🛠 **エラー時対処**: 演算不足は実装へ戻って修正。NaNや全空maskは成功としない。

### 手順 12: 実重みvideo smoke (TR-3/4)
- [x] 🖐 **操作**: 第4章のvideoコマンドを6フレームで実行する。
- [x] 🔎 **確認**: 6 frames、2 IDs、12 PNG、5回memory attention、finite f32。
- [x] 🧪 **テスト**: frame0とframe5を視認し、保持ID/メモリ更新をreportで確認。
- [x] 🛠 **エラー時対処**: 途中failureはpartial outputをpublishしない。プロセスは確認してから再開。

### 手順 13: 回帰・品質検証 (TR-2/3/4)
- [x] 🖐 **操作**: `ctest --test-dir build --output-on-failure`。
- [x] 🔎 **確認**: native/quant/operator/storage/CLI tests全成功。
- [x] 🧪 **テスト**: storage異常系、ID上限、入力/点/過去memory介入の既存テストを含む。
- [x] 🛠 **エラー時対処**: 新しい失敗は原因と修正を記録して対象再実行。

### フェーズ5: 記録・納品

### 手順 14: 一貫したガイドとstatus (TR-5)
- [x] 🖐 **操作**: `docs/SAM31_GUIDE.ja.md` と `evidence/status.json` を現在の証跡から作成する。
- [x] 🔎 **確認**: 取得→変換→build→image→video→testがルートからcopy-paste可能。
- [x] 🧪 **テスト**: docsのファイル/option/出力名称をCLIと照合、秘密情報scan。
- [x] 🛠 **エラー時対処**: 過去提供物の未検証記述は今回のstatusと混同しないよう履歴扱いにする。

### 手順 15: zstd source/model納品物を作成 (TR-5)
- [x] 🖐 **操作**: ソースbundleとGGUFのzstdを `dist/` に作成する。
- [x] 🔎 **確認**: ソース・fixture・ガイド・作業書・今回evidenceを含み、秘密・cache・.gitを含まない。
- [x] 🧪 **テスト**: zstd -t、SHA256、別展開ディレクトリでfixture smoke再実行。
- [x] 🛠 **エラー時対処**: モデルライセンスを保持し、Gitには重みをstageしない。

### 手順 16: DoDを証跡から監査 (TR-1..5)
- [x] 🖐 **操作**: 第6章の全項目を現在のファイル/ログ/reportと照合する。
- [x] 🔎 **確認**: 欠落証拠・未実行・弱い検証がゼロ。
- [x] 🧪 **テスト**: archive復元とchecksum、実モデル両経路、doc review PASSを確認。
- [x] 🛠 **エラー時対処**: 未証明項目があれば未完了として次の操作を続行する。

## 4. 作業に使用するコマンド参考情報

```sh
uv sync --project sam31
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DSAM3_METAL=OFF -DSAM31_BUILD_REFERENCE=OFF -DPython3_EXECUTABLE="$PWD/sam31/.venv/bin/python"
cmake --build build --parallel 8
uv run --project sam31 python sam31/tools/convert_sam31_to_gguf.py --input models/sam3.1_multiplex_convrot_int8.safetensors --output models/sam3.1_multiplex_convrot_int8.gguf --verify
build/sam31/sam31 inspect models/sam3.1_multiplex_convrot_int8.gguf --read-all
build/sam31/sam31 synthetic --output evidence/real-inputs --frames 6
OPENBLAS_NUM_THREADS=16 build/sam31/sam31_image --model models/sam3.1_multiplex_convrot_int8.gguf --image evidence/real-inputs/image.png --point 1:0.25:0.32 --point 2:0.73:0.68 --threads 16 --cache-mb 4096 --output evidence/real-image
OPENBLAS_NUM_THREADS=16 build/sam31/sam31_video --model models/sam3.1_multiplex_convrot_int8.gguf --frames evidence/real-inputs/frames --point 1:0.25:0.32 --point 2:0.73:0.68 --threads 16 --cache-mb 4096 --output evidence/real-video
OPENBLAS_NUM_THREADS=1 ctest --test-dir build --output-on-failure
```

モデルはmodels/へコピーまたはローカルsymlinkで配置。実物の機械依存絶対パスは配布文書に含めない。第4章のcommandsは追加module導入後の契約で、導入前は存在しない。justfileなし、root pyprojectはtorch等を要求するため本タスク用sam31 projectを明示する。

## 6. 完了の定義

- [x] D-1/TR-1: 新規clone・feature branch・入力SHA256・由来が記録されている。
- [x] D-2/TR-2: 指定実重み3136 tensorsのGGUF変換と全payload一致、C++ read-all成功。
- [x] D-3/TR-3: 学習済みGGUFのC++ image出力が2対象、有限値、正しい寸法、image経路608 parameter使用（video経路867と別）。
- [x] D-4/TR-3/4: 学習済みGGUFのC++ videoが6フレーム・2対象・12マスク、memory更新、有限値。
- [x] D-5/TR-4: 人工重みsmoke・既存回帰testsが成功。実重み結果との区別が明確。
- [x] D-6/TR-5: 作業書レビューPASS、実行手順・制限・provenance・作業記録が自己完結。
- [x] D-7/TR-5: zstd source/model成果物、checksum、展開後smokeと整合性検査成功。

## 7. 作業記録

**重要な注意事項：**

- 作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` コマンドで現在時刻を確認し、正確な日時を記録します。
- 各作業項目を開始する際と完了する際の両方で記録を行うこと。
- 作業内容は具体的なコマンドや操作手順を詳細に記載すること。
- 結果・備考欄には成功／失敗、エラー内容、解決方法、重要な気づきを必ず記入すること。
- 複数のフェーズがある場合は、フェーズごとに開始・完了の記録を取ること。
- コード変更を行った場合は、変更したファイル名と変更内容の概要を記録すること。
- エラーが発生した場合は、エラーメッセージと解決策を詳細に記録すること。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
|---|---|---|---|---|
| 2026-10-01 | 08:45:53 JST | Codex | 手順1開始、clone/branch・入力確認 | 実行環境復旧、提供物存在 |
| 2026-10-01 | 08:45:53 JST | Codex | 手順1完了、手順2開始 | develop 42c28f21、feature branch作成、header調査 |

## 8. 作業書レビュー記録

初回レビューは手順3で実施し、`temp/workdoc_review.md` にrubricの全観点を保存する。チェック済みは結果の証拠がある項目のみ。

レビュー修正: 実モデルのtracker直下aliasとINT8 embeddingも対象とし、人工画像での推論が精度の証明ではないことを明記。stageした入力はmodels/のローカルsymlinkで、納品にはsymlinkの機械依存パスを含めない。

| 2026-10-01 | 08:51:24 JST | Codex | 手順2/3完了、手順4開始/完了、手順5開始 | 提供物コード選択導入。実headerは3136 tensors、651 INT8、647 ConvRot、tracker直下alias、embedding量子化。review PASS。 |

2026-10-01: 手順5 uv独立環境・lock生成済み。手順6 loader-red.logでtracker prefix不足とINT8 embedding拒否を再現。

| 2026-10-01 | 08:57:43 JST+0900 | Codex | 手順7 | 成功: loader-green.log と ctest.log: prefix、INT8 embedding、Sequential MLP・norm aliasesを修正、全CLI試験成功 |

| 2026-10-01 | 08:57:43 JST+0900 | Codex | 手順8 | 成功: build.log exit0、sam31_native compile flags SAM31_USE_BLAS=1; OpenBLAS導入 |

| 2026-10-01 | 08:57:43 JST+0900 | Codex | 手順9 | 成功: fixture-smoke.log と fixture-smoke/: native C++ image/video/quant/operator成功 |

| 2026-10-01 | 08:57:43 JST+0900 | Codex | 手順10 | 成功: conversion.json: 3136 tensors 全906735111 payload bytes exact; inspect.json all_payloads_read=true |

注: 実行再開時の先行調査で変換と一部回帰を先に実施。履歴を順番どおりに実行したと偽らず、上記証跡で記録を補完した。

| 2026-10-01 | 08:58:46 JST | Codex | 手順11完了 | real-image/report.json、real-image-validation.json: 2 masks IoU=1.0、finite logits、32 ViT blocks、608 image parameters。overlayを視認済み。 |

| 2026-10-01 | 09:04:27 JST+0900 | Codex | 手順12完了 | real-video/report.jsonとvalidation: 6 frames/12 masks 全IoU1.0、867 tensors、memory5回20blocks、312019ms |

| 2026-10-01 | 09:04:27 JST+0900 | Codex | 手順13完了 | ctest.log: 5/5 groups PASS、fixture-regeneration.json: 独立writer再生成も1487 tensors roundtrip exact |

| 2026-10-01 | 09:05:50 JST+0900 | Codex | 手順14完了 | SAM31_GUIDE、ONBOARDING、provenance/status、workdoc再レビューPASS。実モデル2経路検証済み。 |

| 2026-10-01 | 09:07:19 JST+0900 | Codex | 手順15完了 | zstd source/model検査、GGUF復元SHA一致。archive-files-check.log全2194ファイルOK、別展開Release buildとarchive-smoke.log PASS |

| 2026-10-01 | 09:07:19 JST+0900 | Codex | 手順16完了 | DoD D1..D7: actual identity/全payload/image2/video12/memory20/CTest5/docs review/archive restored smoke全確認 |
