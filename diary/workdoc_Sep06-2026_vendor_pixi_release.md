# 作業計画書 兼 記録書: sam3.cpp リポジトリ自己完結化 + pixi環境整備 + プリビルドバイナリ作成

---

**日付:** 2026年09月06日
**作業ディレクトリ・リポジトリ:** `/home/inaho-omen/Project/sam3.cpp` (git repo: `https://github.com/yuki-inaho/sam3.cpp.git`)
**作業者:** opencode (z-ai/glm-5.3-flash)

---

## 1. 作業目的

本日の作業は、以下の目標を達成するために実施します。

*   **目標1:** ggml を git submodule からリポジトリ内に固定 commit (`331b9cba52b23d895bc4ad218c007eb5e667540f`) で同梱 (vendor) し、clone だけで自己完結する構成にする
*   **目標2:** EdgeTAM モデル 3 種 (`edgetam_f16.ggml` / `edgetam_q8_0.ggml` / `edgetam_q4_0.ggml`) とサンプルデータ (`data/test_image.jpg`, `data/test_video.mp4`) をリポジトリに同梱し、同梱デモコードが即座に動くようにする
*   **目標3:** pixi 環境ベースのビルド/デモ実行を整備する
*   **目標4:** ユーザー指定の実行環境 (Debian 13 trixie / x86_64 / glibc 2.41 / GCC 14.2 / AVX-512 対応 EPYC / 5 vCPU) で「即動く」プリビルドバイナリを作り、GitHub Releases に置ける状態にする

### 1.1 ゴール要求分析

*   **ユーザーの直観的・直截的な目的:** リポジトリを clone (submodule 更新不要) すればモデル・サンプルデータ・ビルド環境まで揃っていてデモが動き、さらにリリースページからダウンロードしたプリビルドバイナリをユーザーの Debian 13 コンテナで展開するだけで EdgeTAM が CPU 推論できる状態にしたい。
*   **明示要求:**
    1. ggml は submodule ではなく固定 commit でリポジトリ内に self-contained に同梱
    2. `edgetam_q8_0.ggml` / `edgetam_q4_0.ggml` / `edgetam_f16.ggml` をリポジトリ内に同梱
    3. サンプルデータ・デモコードが動作する状態にする
    4. pixi 環境ベースでできるように整備する
    5. Debian 13 (trixie) / x86_64 / glibc 2.41 / GCC 14.2 環境で即動くプリビルドバイナリをリリースページに置ける形で作成
    6. 作業書 (write-workdoc スキル) を作り、review スキルでレビューする
*   **暗黙制約:**
    *   C++14 ビルド (CMakeLists で `CMAKE_CXX_STANDARD 14`)。ユーザー環境 GCC 14.2 / Clang 17 でも問題なし
    *   重量変換スクリプト (torch 依存) の Python 環境は既存 uv (`pyproject.toml` + `uv.lock`) を維持。C++ ビルド/実行は pixi に集約 (pixi と uv の役割分担を明示。暗黙 fallback は禁止のため両者を明記)
    *   ビデオデコードは実行時にシステム `ffmpeg` 依存 (`sam3.cpp:12288` の `sam3_decode_video_frame`)。画像推論は ffmpeg 不要
    *   対象環境は GUI なしコンテナ → SDL2 GUI は対象外。ヘッドレス CLI (`sam3_seg` 新規追加) と `sam3_benchmark` / `sam3_profile_edgetam` / `sam3_quantize` を同梱
    *   互換性重視のビルド: `GGML_NATIVE=OFF` + AVX2/FMA (対象 EPYC は AVX-512 対応だが AVX2 基準で広範に動作)。glibc 2.41 はビルド環境より新しいため、システム gcc (glibc 2.39) ビルドでも動作するが、conda-forge sysroot (glibc 2.17 相当) を使った pixi ビルドを採用し互換性を最大化
    *   コミットはユーザー指示があるまで行わない (git index への staging までは実施)
*   **非ゴール:**
    *   SAM 2 / SAM 3 系モデルの同梱 (EdgeTAM 3 種のみ)
    *   Windows / macOS プリビルドのローカル作成 (既存 `release.yml` の CI に任せる)
    *   Metal / GPU 側の改修
    *   重み変換スクリプトの pixi 化 (torch 重依存のため uv 維持)
*   **成功条件:**
    1. `git status` 上で ggml が submodule (mode 160000) ではなく通常ファイルとして staged になり、`.gitmodules` が削除済み
    2. `models/edgetam_{f16,q8_0,q4_0}.ggml` と `data/{test_image.jpg,test_video.mp4}` が git tracked
    3. `pixi run build` → `pixi run demo-cpu` がローカルで成功 (EdgeTAM CPU 推論、mask PNG 出力)
    4. リリース tarball `sam3-linux-x86_64-<version>.tar.gz` (version は `git describe --tags --always`、例: `01832ef`) を展開し、同梱バイナリで EdgeTAM 推論が成功
    5. 作業書が本書として存在し、レビュー結果が反映されている
*   **リスクと前提:**
    *   リポジトリへモデル 64 MB + データ数 MB を追加するためリポジトリが肥大化する (ユーザー明示要求のため許容)
    *   ggml vendor 化により upstream 更新の追従が手作業になる (固定化はユーザー要求どおり)
    *   pixi の conda-forge コンパイラ (gcc) でビルドした場合の glibc シンボル互換は sysroot 2.17 由来で広い。ただしローカル実機でのみ検証可能 (Debian 13 実機なし) → 対策: ldd で要求 glibc バージョンを確認し、ローカル (glibc 2.39) + ユーザー環境 (2.41) の双方で動く要件を満たすか記録
    *   `sam3_benchmark` は ffmpeg を実行時呼び出し → tarball には同梱せず README に明記 + pixi 環境で提供

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
| :--- | :--- | :--- | :--- | :--- |
| SG-1 | ggml vendor 化 (固定 commit 331b9cba) | 目標1 / 明示要求1 | 通常ディレクトリ化した `ggml/`、削除済み `.gitmodules` | `git ls-files -s ggml` の mode が 100644 系、`git status` で staged |
| SG-2 | モデル + サンプルデータ同梱 | 目標2 / 明示要求2,3 | `models/edgetam_*.ggml` x3、`data/test_image.jpg`, `data/test_video.mp4`、`.gitignore` 調整 | `git ls-files models data` で 5 ファイル確認 |
| SG-3 | pixi 環境整備 | 目標3 / 明示要求4 | `pixi.toml`、README/CLAUDE.md 更新 | `pixi run build` 成功 |
| SG-4 | ヘッドレス CLI デモ追加 | 目標2,4 / 明示要求3,5 | `examples/sam3_seg.cpp`、examples/CMakeLists.txt 更新 | `sam3_seg` で mask.png 生成 |
| SG-5 | 動作検証 (CPU EdgeTAM) | 目標2 / 明示要求3 | 実行ログ (作業記録) | `pixi run demo-cpu` / `pixi run benchmark-cpu` 成功 |
| SG-6 | プリビルドバイナリ + パッケージング | 目標4 / 明示要求5 | `scripts/package_release.sh`、`dist/sam3-linux-x86_64-bundle.tar.gz` | tarball 展開→同梱バイナリ実行→mask.png 生成 |
| SG-7 | CI workflow 調整 | 目標4 / 明示要求5 | `.github/workflows/release.yml` 更新 (vendor 化対応 + モデル同梱) | yaml 構文確認 (act 実行は非ゴール) |
| SG-8 | 作業書レビュー | 明示要求6 | 本書のレビュー反映 | review-written-workdoc スキル実行ログ |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
| :--- | :--- | :--- | :--- |
| TR-1 | ggml 固定 commit vendor 化 | フェーズ2 手順1-2 | `git ls-files -s ggml` 出力、`.gitmodules` 削除 |
| TR-2 | EdgeTAM モデル同梱 | フェーズ2 手順3 | `git ls-files models` 出力、ファイルサイズ記録 |
| TR-3 | サンプルデータ同梱 | フェーズ2 手順4 | `git ls-files data` 出力 |
| TR-4 | pixi 環境 | フェーズ2 手順5 | `pixi.toml`、`pixi run build` ログ |
| TR-5 | ヘッドレス CLI | フェーズ2 手順6 | `examples/sam3_seg.cpp`、実行ログ |
| TR-6 | CPU EdgeTAM 動作検証 | フェーズ3 手順8-9 | profile/seg/benchmark 実行ログ |
| TR-7 | プリビルドバイナリ (Debian13 想定) | フェーズ2 手順7, フェーズ3 手順10 | `scripts/package_release.sh`、ldd 出力、tarball 検証ログ |
| TR-8 | ドキュメント整備 | フェーズ2 手順8 | README.md / CLAUDE.md diff |
| TR-9 | 作業書レビュー | フェーズ3 手順11 | review スキル実行、本書更新 |

---

## 2. 作業内容

### フェーズ 1: 調査・設計フェーズ (見積: 0.5h)

このフェーズでは、実装に着手する前の準備作業を行います。

1.  **現状構成の分析:**
    *   **タスク内容:** `.gitmodules` (url: PABannier/ggml, path: ggml, commit 331b9cba / sam3-metal-ops ブランチ先端)、`.gitignore` (`models/`, `*.ggml`, `data/` が除外対象)、`scripts/download_test_data.sh` (COCO 画像 + samplelib 動画を取得)、`examples/` 構成 (GUI は SDL2 必須、`sam3_benchmark` は実行時 ffmpeg 呼び出し) を確認。
    *   **目的:** vendor 化・同梱・pixi 化の影響範囲を正確に把握する。
    *   **対応サブゴール/Trace ID:** SG-1〜SG-3 / TR-1〜TR-4
2.  **ggml ビルドオプション確認:**
    *   **タスク内容:** `ggml/CMakeLists.txt` で `GGML_NATIVE` / `GGML_AVX2` / `GGML_FMA` / `GGML_CPU_ALL_VARIANTS` (要 GGML_BACKEND_DL) の仕様を確認。
    *   **目的:** プリビルドの互換性戦略 (AVX2 静的構成) を確定する。`GGML_CPU_ALL_VARIANTS` は DL 必須のため採用しない。
    *   **対応サブゴール/Trace ID:** SG-6 / TR-7
3.  **設計方針の文書化:**
    *   **タスク内容:** vendor 化手順 (deinit → gitlink 削除 → ggml/.git 削除 → 再 add)、pixi タスク設計、tarball 構成 (bin/models/data/README) を本書に記載済みのとおり確定。
    *   **目的:** フェーズ2 を迷いなく実行できるようにする。
    *   **対応サブゴール/Trace ID:** SG-1〜SG-6 / TR-1〜TR-7

### フェーズ 2: 実装フェーズ (見積: 2.0h)

1.  **ggml vendor 化:**
    *   **タスク内容:** `git submodule deinit -f ggml` → `git rm -f ggml` (gitlink 削除) → `.git/modules/ggml` 削除 → `ggml/.git` 削除 → `git add ggml/` で全ファイルを通常追跡へ。`.gitmodules` を削除。作業ツリーの ggml 内容 (commit 331b9cba) は変更しない。
    *   **目的:** 明示要求1 を満たす。
    *   **対応サブゴール/Trace ID:** SG-1 / TR-1
2.  **モデル・データ同梱:**
    *   **タスク内容:** Downloads から 3 モデルを `models/` へコピー (q8_0 と `q8_0 (1)` は同一 md5 につき 1 つのみ)。`scripts/download_test_data.sh` 相当のデータを `data/` へ取得。`.gitignore` を `models/*` + `!models/edgetam_*.ggml`、`data/*` + `!data/test_image.jpg` + `!data/test_video.mp4` 形式へ修正。`git add`。
    *   **目的:** 明示要求2,3 を満たす。
    *   **対応サブゴール/Trace ID:** SG-2 / TR-2, TR-3
3.  **pixi 環境整備:**
    *   **タスク内容:** `pixi.toml` を新規作成。依存: cmake, ninja, cxx-compiler, ffmpeg, patchelf。タスク: `configure`, `build`, `demo-cpu`, `benchmark-cpu`, `profile-cpu`, `release-build`。README.md / CLAUDE.md に pixi 手順追記。
    *   **目的:** 明示要求4 を満たす。
    *   **対応サブゴール/Trace ID:** SG-3 / TR-4
4.  **ヘッドレス CLI (`sam3_seg`) 追加:**
    *   **タスク内容:** `examples/sam3_seg.cpp` を新規作成。`sam3_load_image` → `sam3_encode_image` → `sam3_segment_pvs` (点プロンプト) → `sam3_save_mask` で mask PNG 出力。オプション: `--model --image --x --y --out --cpu --n-threads --box`。`examples/CMakeLists.txt` に登録。
    *   **目的:** GUI なしコンテナでデモ可能にする (明示要求3,5)。
    *   **対応サブゴール/Trace ID:** SG-4 / TR-5
5.  **パッケージングスクリプト:**
    *   **タスク内容:** `scripts/package_release.sh` 新規作成。Release ビルド (`-DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_OPENMP=OFF -DCMAKE_SKIP_RPATH=ON` + 静的 libstdc++) → `dist/sam3-linux-x86_64-bundle/` に bin 4 種 + models 3 種 + data 2 種 + README (リポジトリの `README-release.md` を `README.md` として同梱) を収集 → `sam3-linux-x86_64-<version>.tar.gz`。
    *   **目的:** 明示要求5 のプリビルドを作る。
    *   **対応サブゴール/Trace ID:** SG-6 / TR-7
6.  **CI workflow 調整:**
    *   **タスク内容:** `.github/workflows/release.yml` から `submodules: recursive` を削除 (vendor 化済み)、Linux x86_64 に AVX2 固定オプションとモデル/データ同梱を追加。
    *   **目的:** リリースページ運用を CI で再現可能にする。
    *   **対応サブゴール/Trace ID:** SG-7 / TR-7

### フェーズ 3: テストと動作検証 (見積: 1.0h)

1.  **ビルド検証:**
    *   **タスク内容:** `pixi run build` で全ターゲット (GUI は SDL2 未導入のためスキップ) をビルド。
    *   **目的:** vendor 化後のビルド壊れがないこと。
    *   **対応サブゴール/Trace ID:** SG-3, SG-5 / TR-4
2.  **CPU EdgeTAM 動作検証:**
    *   **タスク内容:** `pixi run demo-cpu` (sam3_seg + profile_edgetam) と `pixi run benchmark-cpu` (3 フレーム) を実行し、mask PNG 生成と検出数を確認。
    *   **目的:** 同梱モデル+データでデモが完結すること。
    *   **対応サブゴール/Trace ID:** SG-5 / TR-6
3.  **プリビルド検証:**
    *   **タスク内容:** tarball を `/tmp` に展開し、同梱バイナリで sam3_seg 実行。`ldd` で要求 glibc バージョン確認。
    *   **目的:** ユーザー Debian 13 環境 (glibc 2.41) で即動くことをローカル (glibc 2.39) から検証。
    *   **対応サブゴール/Trace ID:** SG-6 / TR-7
4.  **作業書レビュー:**
    *   **タスク内容:** `review-written-workdoc` スキルで本書をレビューし、安全な改善を反映。
    *   **目的:** 明示要求6 を満たす。
    *   **対応サブゴール/Trace ID:** SG-8 / TR-9

---

## 3. 作業チェックリスト

*作業が完了したら `[ ]` を `[x]` に変更します。*

### フェーズ 1: 調査・設計フェーズ

### 手順 1: 現状構成の確認
- [x] 🖐 **操作**: `.gitmodules`, `.gitignore`, `scripts/download_test_data.sh`, `examples/CMakeLists.txt`, `.github/workflows/release.yml`, `ggml/CMakeLists.txt` を読み、影響範囲を記録する。
- [x] 🔎 **確認**: ggml commit 331b9cba (作業ツリー・クリーン)、.gitignore 除外対象、GUI/ffmpeg 依存、`GGML_CPU_ALL_VARIANTS` は DL 必須の 4 点が本書に記録済み。
- [x] 🧪 **テスト**: 調査フェーズのため自動テスト不要。`git ls-files -s ggml` が mode 160000 を示すことを記録済み。
- [x] 🛠 **エラー時対処**: ファイルが想定と異なる場合は `rg` / `git log` で実態を確認し、前提不一致を作業記録に残す。

### フェーズ 2: 実装フェーズ

### 手順 2: ggml の vendor 化
- [x] 🖐 **操作**: `git submodule deinit -f ggml && git rm -f ggml && rm -rf .git/modules/ggml && rm .gitmodules && git add ggml/` を実行する (実行時は rsync バックアップ → deinit → 復元方式。作業記録参照)。
- [x] 🔎 **確認**: `git ls-files -s ggml | head` の mode が 100644/100755 になり、`git status` で `.gitmodules` 削除と ggml 追加 (1864 ファイル) が staged 表示される。
- [x] 🧪 **テスト**: `pixi run build` (手順7) が成功することを TDD の成功条件とする (vendor 化前は submodule 前提から変更後ビルドが通ることを確認)。
- [x] 🛠 **エラー時対処**: deinit で作業ツリーが消えた場合は `git checkout -- ggml` 後に再実行。`git add` で "embedded repository" 警告が出たら `ggml/.git` 残存を疑い削除する。

### 手順 3: EdgeTAM モデルの同梱
- [x] 🖐 **操作**: `cp /home/inaho-omen/Downloads/edgetam_{f16,q8_0,q4_0}.ggml models/` を実行する。
- [x] 🔎 **確認**: `models/edgetam_f16.ggml` (27MB), `models/edgetam_q8_0.ggml` (19MB), `models/edgetam_q4_0.ggml` (15MB) が存在する。
- [x] 🧪 **テスト**: 後続手順 8 の sam3_seg ロード成功を TDD 成功条件とする。
- [x] 🛠 **エラー時対処**: Downloads 側に重複 (`edgetam_q8_0 (1).ggml`) があるため md5 照合し、同一なら 1 つのみ採用。

### 手順 4: サンプルデータの同梱と .gitignore 調整
- [x] 🖐 **操作**: `scripts/download_test_data.sh` の取得元と同一 URL から `data/test_image.jpg`, `data/test_video.mp4` を取得し、`.gitignore` の `models/`, `data/` をホワイトリスト方式に書き換えて `git add models data .gitignore`。
- [x] 🔎 **確認**: `git ls-files models data` に 5 ファイル列挙される。
- [x] 🧪 **テスト**: `git check-ignore -v models/edgetam_f16.ggml` が非 0 (無視されない) を確認。
- [x] 🛠 **エラー時対処**: ネットワーク不通時は既存 `tests/cat.jpg` での代替を検討せず、ダウンロード成功まで明示的にブロッカーとして記録する (暗黙 fallback 禁止)。

### 手順 5: pixi.toml の作成
- [x] 🖐 **操作**: `pixi.toml` を新規作成し (channels: conda-forge / platforms: linux-64, osx-64, osx-arm64 / 依存: cmake, ninja, cxx-compiler, ffmpeg, patchelf)、タスク `configure`, `build`, `demo-cpu`, `benchmark-cpu`, `profile-cpu`, `release-build` を定義する。
- [x] 🔎 **確認**: `pixi task list` でタスク一覧が表示される。
- [x] 🧪 **テスト**: 手順 7 の `pixi run build` 成功を TDD 成功条件とする。
- [x] 🛠 **エラー時対処**: conda 依存解決失敗時はバージョン制約を緩和し、原因を記録する。

### 手順 6: sam3_seg (ヘッドレス CLI) の追加
- [x] 🖐 **操作**: `examples/sam3_seg.cpp` を新規作成し、`examples/CMakeLists.txt` に `add_executable(sam3_seg ...)` + `target_link_libraries(sam3_seg PRIVATE sam3)` を追加する。
- [x] 🔎 **確認**: `build/examples/sam3_seg --help` 相当 (オプション不正時 usage 表示) が動く。
- [x] 🧪 **テスト**: 手順 8 の mask.png 生成 (検出 1 件、スコア出力) を TDD 成功条件とする。
- [x] 🛠 **エラー時対処**: API 名不一致は `sam3.h` の宣言 (`sam3_load_image`, `sam3_encode_image`, `sam3_segment_pvs`, `sam3_save_mask`, `sam3_pvs_params`) と突き合わせて修正。

### 手順 7: パッケージングスクリプトと CI 調整
- [x] 🖐 **操作**: `scripts/package_release.sh` を新規作成し、`.github/workflows/release.yml` から `submodules: recursive` を削除・Linux x86_64 に AVX2 固定 + bundle 化を反映する。
- [x] 🔎 **確認**: スクリプトが `dist/sam3-linux-x86_64-<version>.tar.gz` (実績: `01832ef`) と `dist/sam3-linux-x86_64-bundle/` を生成する。
- [x] 🧪 **テスト**: 手順 10 の tarball 検証を TDD 成功条件とする。
- [x] 🛠 **エラー時対処**: ビルド失敗時は CMake オプション名を `ggml/CMakeLists.txt` と突き合わせる。

### 手順 8: README / CLAUDE.md の更新
- [x] 🖐 **操作**: README.md Quick Start に pixi 手順・同梱モデル/データ・プリビルド使い方を追記、CLAUDE.md の Build 節に pixi を追記する。
- [x] 🔎 **確認**: clone → `pixi run build` → `pixi run demo-cpu` の 3 ステップがドキュメントだけで完結する。
- [x] 🧪 **テスト**: ドキュメント記載コマンドをそのまま実行して成功する (手順 9)。
- [x] 🛠 **エラー時対処**: 記載コマンドと実挙動の不一致を発見したらドキュメントを実挙動に合わせる。

### フェーズ 3: テストと動作検証

### 手順 9: pixi ビルドと CPU EdgeTAM 検証
- [x] 🖐 **操作**: `pixi run build` → `pixi run demo-cpu` → `pixi run benchmark-cpu` を実行する。
- [x] 🔎 **確認**: ビルド成功、`output/mask.png` 生成、benchmark が edgetam 3 種で検出 1 件以上を報告。
- [x] 🧪 **テスト**: vendor 化前は不可能だった「clone 即デモ」が成功する (失敗→成功の転換点)。
- [x] 🛠 **エラー時対処**: ffmpeg 不在エラー時は pixi 環境内 ffmpeg を使っているか確認 (`pixi run bash -c 'which ffmpeg'`)。

### 手順 10: プリビルド tarball の検証
- [x] 🖐 **操作**: `pixi run release-build` 後、tarball を `/tmp` に展開し、同梱 `bin/sam3_seg` で推論実行 + `ldd` / `objdump` で glibc 要求バージョンを確認する。
- [x] 🔎 **確認**: tarball 展開のみで mask.png が生成され、glibc 要求 ≤ 2.39 (ローカル) かつ AVX2 固定。
- [x] 🧪 **テスト**: 展開→実行の E2E 成功を記録する。
- [x] 🛠 **エラー時対処**: glibc 要求が高すぎる場合は conda-forge sysroot ビルドへ切り替え、原因を記録する。

### 手順 11: 作業書レビューと反映
- [x] 🖐 **操作**: `review-written-workdoc` スキルで本書をレビューし、指摘を反映する。
- [x] 🔎 **確認**: レビュー指摘が作業記録に記録され、安全な改善が適用されている。
- [x] 🧪 **テスト**: レビュー観点 (目的分析/フェーズ分離/チェックリスト形式) の自己評価を記録。
- [x] 🛠 **エラー時対処**: レビュー指摘で構造修正が必要な場合は編集し、変更前後を残す。

---

## 4. 作業に使用するコマンド参考情報

### 基本的な開発ワークフロー (pixi)

```bash
# 環境作成 + ビルド
pixi run build

# CPU EdgeTAM デモ (同梱モデル+同梱画像)
pixi run demo-cpu

# 動画トラッキングベンチマーク (要 pixi 環境の ffmpeg)
pixi run benchmark-cpu

# プリビルド tarball 作成
pixi run release-build
```

### 重量変換ツール (uv 維持)

```bash
# PyTorch checkpoint → ggml 変換は uv 環境 (torch 依存)
uv run python convert_edgetam_to_ggml.py --model edgetam.pt --output models/edgetam_f16.ggml --ftype 1
```

### 特定機能の実行・デバッグ例

```bash
./build/examples/sam3_seg --model models/edgetam_q8_0.ggml --image data/test_image.jpg \
  --x 315 --y 250 --out output/mask.png --cpu
./build/examples/sam3_profile_edgetam --model models/edgetam_q8_0.ggml --image data/test_image.jpg --cpu
```

---

## 6. 完了の定義

*作業が最後まで完了したら `[ ]` を `[x]` にしつつ、作業が本当に完了したかをチェックします*

- [x] 観点1: 成功条件 1〜4 (vendor 化、同梱、pixi デモ、tarball 検証) を満たしている
- [x] 観点2: すべての Trace ID に対応する証跡が作業記録に残っている
- [x] 観点3: ビルド・デモ・ベンチマークが pixi タスクとして成功している
- [x] 観点4: 暗黙 fallback を使わず、未対応事項 (commit 未実施、実リリース未実施等) は明示的に記録されている

---

## 7. 作業記録

**重要な注意事項:**

*   作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` コマンドで現在時刻を確認し、正確な日時を記録します。
*   各作業項目を開始する際と完了する際の両方で記録を行うこと。
*   作業内容は具体的なコマンドや操作手順を詳細に記載すること。
*   結果・備考欄には成功／失敗、エラー内容、解決方法、重要な気づきを必ず記入すること。
*   複数のフェーズがある場合は、フェーズごとに開始・完了の記録を取ること。
*   コード変更を行った場合は、変更したファイル名と変更内容の概要を記録すること。
*   エラーが発生した場合は、エラーメッセージと解決策を詳細に記録すること。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-09-06 | 15:38:21 JST | opencode | フェーズ1開始: 現状構成調査 | `.gitmodules`(PABannier/ggml), commit 331b9cba (作業ツリー・クリーン), .gitignore 除外対象, GUI は SDL2+OpenGL 依存, `sam3_decode_video_frame` が実行時 ffmpeg/ffprobe CLI 呼び出し (sam3.cpp:12288) を確認 |
| 2026-09-06 | 15:40:00 JST | opencode | ggml ビルドオプション確認 | **重要発見**: `GGML_CPU_ALL_VARIANTS` は `GGML_BACKEND_DL` 必須 → AVX2 固定の静的ビルドを採用 |
| 2026-09-06 | 15:41:00 JST | opencode | フェーズ1完了: 調査・設計 | vendor 手順・pixi タスク設計・tarball 構成を確定 |
| 2026-09-06 | 15:42:00 JST | opencode | フェーズ2開始: ggml vendor 化 | rsync バックアップ → `git submodule deinit -f ggml` → `git rm -f ggml` → `.git/modules/ggml`・`ggml/.git` 削除 → 復元 → `.gitmodules` 削除 → `git add` |
| 2026-09-06 | 15:44:30 JST | opencode | vendor 化検証 | ✅ `git ls-files -s ggml` mode 100644/100755 化。初回 add で 1103 ファイルと漏れ → **発見**: 再 `git add ggml` で `ggml/examples/yolo/data`(761 ファイル, 上流 commit に tracked であることを GitHub API で確認) を追加し合計 1864 ファイル |
| 2026-09-06 | 15:47:30 JST | opencode | モデル同梱 | ✅ edgetam_f16(27MB)/q8_0(19MB)/q4_0(15MB) を models/ へ。q8_0 重複ファイルは md5 一致 (afd6bcda…) につき 1 本化 |
| 2026-09-06 | 15:48:00 JST | opencode | サンプルデータ取得 | ✅ data/test_image.jpg (COCO 39769, 640x480, 173KB), data/test_video.mp4 (sample-5s, 2.8MB) |
| 2026-09-06 | 15:49:00 JST | opencode | .gitignore 調整 | ✅ `models/*` + `!models/edgetam_*.ggml`, `data/*` + `!data/test_{image.jpg,video.mp4}`, `dist/`, `output/`, `.pixi/`, `build-release/` 追加。`git check-ignore` で非除外を確認 |
| 2026-09-06 | 15:52:00 JST | opencode | pixi.toml 作成 | ✅ conda-forge / linux-64+osx-64+osx-arm64 / cmake, ninja, cxx-compiler, ffmpeg, patchelf。タスク: configure, build, demo-cpu, profile-cpu, benchmark-cpu, release-build |
| 2026-09-06 | 15:54:00 JST | opencode | sam3_seg 追加 | ✅ examples/sam3_seg.cpp (点/ボックスプロンプト→mask PNG, --cpu 対応) + examples/CMakeLists.txt 登録 |
| 2026-09-06 | 15:56:30 JST | opencode | package_release.sh + release.yml 調整 | ✅ AVX2 固定・静的 libstdc++ ビルド→bin 4種+models+data+README を tar.gz 化。release.yml は `submodules: recursive` 削除・Linux x86_64 を bundle 化 |
| 2026-09-06 | 15:57:30 JST | opencode | ビルド試行 1 | ❌ 失敗: `Could NOT find OpenGL` — SDL2 は見つかるが OpenGL なしの headless 環境で GUI 例が configure 失敗 → **修正**: examples/CMakeLists.txt を OpenGL 未検出時スキップに変更 |
| 2026-09-06 | 15:58:30 JST | opencode | ビルド試行 2 | ❌ 失敗: conda 環境の ccache が `libhiredis.so.0.14` 未解決でリンクエラー → **修正**: pixi.toml と package_release.sh に `-DGGML_CCACHE=OFF` |
| 2026-09-06 | 15:59:30 JST | opencode | ビルド成功 + demo-cpu | ✅ 全ターゲット (sam3/seg/benchmark/profile/quantize) ビルド成功。demo-cpu: 検出 1 件 score=0.511, box=(91,217)-(347,377), output/mask.png (640x480) 生成 |
| 2026-09-06 | 16:00:30 JST | opencode | profile-cpu | ✅ ステージ別内訳出力 (FPN neck 848ms 等) |
| 2026-09-06 | 16:00:50 JST | opencode | benchmark-cpu 試行 | ❌ 3/3 FAIL `decode frame failed` — **原因**: conda の ffmpeg 8 は廃止済み `-vsync` を解釈せず `Unrecognized option` |
| 2026-09-06 | 16:02:00 JST | opencode | ffmpeg 互換修正 | **修正**: sam3.cpp:12295 から `-vsync vfr` を削除 (`-frames:v 1` + select で冗長)。conda ffmpeg 8 とシステム ffmpeg 3.4.13 の双方で frame 0 デコード (6,220,800 bytes) を確認 |
| 2026-09-06 | 16:04:00 JST | opencode | 再ビルド + benchmark-cpu | ✅ 3/3 OK (edgetam_f16/q8_0/q4_0, CPU, 3 フレーム, 検出 1 件ずつ) |
| 2026-09-06 | 16:10:00 JST | opencode | release-build v1 | ✅ dist/sam3-linux-x86_64-01832ef.tar.gz (77MB)。**発見**: RPATH が pixi env を指し `libgomp.so.1` が conda 環境依存 → 移植性問題 |
| 2026-09-06 | 16:14:00 JST | opencode | 移植性修正 | **修正**: `-DGGML_OPENMP=OFF` (libgomp 依存解消, ggml スレッドプールで代替・性能同等以上) + `-DCMAKE_SKIP_RPATH=ON` + patchelf `--remove-rpath`。ldd が libc/libm/libpthread/libdl のみに、GLIBC 要求 ≤ 2.27 (対象 glibc 2.41 ✓) |
| 2026-09-06 | 16:15:30 JST | opencode | bundle E2E (sam3_seg) | ✅ tarball 展開のみで edgetam_q8_0 推論成功 (score=0.511, 開発ビルドと同値, graph 760ms と高速化) |
| 2026-09-06 | 16:17:00 JST | opencode | bundle E2E 最終 | ✅ 展開→ `bin/sam3_seg` (q4_0, score=0.422) + `bin/sam3_benchmark` (q8_0, 2 フレーム 1/1 OK)。README/CLAUDE.md 更新済み、全変更を git stage 完了 (commit はユーザー指示待ち) |
| 2026-09-06 | 16:22:00 JST | opencode | フェーズ3: 作業書レビュー (review-written-workdoc スキル) | **指摘反映**: ①tarball 名を実際の `sam3-linux-x86_64-<git describe>.tar.gz` 命名に統一 (成功条件4/SG-6/手順7) ②フェーズ2項目3 のタスク一覧を実装と一致 (release-verify 削除, patchelf 追加) ③README-release.md の実ファイル名に修正 ④手順2 の実際の実行方式 (rsync バックアップ) 注記 ⑤手順9 エラー対処の pixi コマンド修正 ⑥「sysroot 2.17 由最大」タイポ修正 ⑦手順8 テストの自己参照修正。**判定: PASS_WITH_NOTES → 全 Minor/Major 指摘を反映済み** |
| 2026-09-06 | 16:24:00 JST | opencode | 全作業完了 | 成功条件 1〜5 すべて達成。未実施: git commit (ユーザー指示待ち), GitHub Releases への実アップロード (リポジトリ push 権限とタグ作成が必要なため手順書のみ提供) |
| | | | | |
