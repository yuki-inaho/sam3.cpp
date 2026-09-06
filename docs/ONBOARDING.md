# LLMオンボーディングサマリー

> 新任 LLM エージェントが `sam3.cpp` に参加する際の初期資料。
> 記載事実には検証状態を `確認済み` / `未検証` / `推定` で明示する。
> 最終更新: 2026-09-06 / 対象コミット: `develop` @ `e7c98a8`

## 1. プロジェクト概要と目的

- **プロジェクト名称・領域:** `sam3.cpp` — Meta の SAM 3 (Segment Anything Model 3) を **C++14 + ggml** に移植した画像/動画セグメンテーション推論エンジン。CPU と Metal (Apple GPU) で動作する。
- **最終成果物:**
  - ライブラリ本体 `sam3.cpp` (14,439 行) + 公開 API `sam3.h` (519 行) の 1 ライブラリ構成
  - CLI 実行ファイル: `sam3_seg` (ヘッドレス点/ボックス→mask PNG)、`sam3_benchmark`、`sam3_profile_edgetam`、`sam3_quantize`
  - GUI 実行ファイル: `sam3_image` / `sam3_video` (SDL2 + OpenGL が見つかった場合のみビルド。`examples/CMakeLists.txt:17-35`)
  - リリース用可搬バンドル `sam3-linux-x86_64.tar.gz` (bin 4 種 + モデル 3 種 + サンプルデータ + README、約 70MB)
- **ビジネス背景・価値:** PyTorch ランタイムなしで SAM 系モデルを動かす。リポジトリは **self-contained** (clone → build → run) を設計目標とし、ggml・EdgeTAM モデル・サンプルデータをすべて同梱している。
- **現時点の進捗サマリ:**
  - アーキテクチャ実装は `PLAN.md` の Phase 0〜10 (重み変換 / BPE トークナイザ / ViT / テキストエンコーダ / DETR デコーダ / PVS / 動画トラッキング / Visual-Only モデル) の構成で進行。
  - **`確認済み`** self-contained 化 (ggml の vendor 化、モデル・データ同梱)、pixi 環境、Linux x86_64 リリース自動化までが `develop` ブランチに入っている。
  - **`確認済み`** `main` は `01832ef` のまま。`develop` が 5 コミット先行しており、**未マージ**。

---

## 2. クリティカルな要求・制約

> 「壊してはいけない」品質・仕様ライン。

- **【最重要】ggml のグラフ分離** — パイプラインの各ステージは**専用の `ggml_context` / `ggml_cgraph` / `ggml_gallocr`** で実行し、ステージ間は CPU 側 `std::vector<float>` で受け渡す。**state tensor (`state.neck_trk[*]` 等) をグラフのオペランドにしてはいけない**。`ggml_build_forward_expand` が依存ツリー全体を辿り、ViT 全体の再計算 (2500+ ノード / 約 40 秒) を引き込む。違反すると**クラッシュせず数値だけが静かに壊れる**ため発見が極めて困難。詳細と正誤例は `CLAUDE.md` の「ggml graph isolation (CRITICAL)」章。
- **コード構造:** struct と自由関数のみ。**クラス・継承・仮想関数・多態を使わない**。内部 static 関数は `sam3_` プレフィックス。
- **例外を使わない。** 失敗し得る関数は `bool` か `nullptr` を返す。診断は `fprintf(stderr, ...)` (`std::cerr` は使わない)。
- **依存関係の固定:** ライブラリ本体の依存は **ggml (in-tree) / stb / C++14 標準ライブラリのみ**。SDL2 と ImGui は examples 専用。新規サードパーティ依存を足さない。
- **vendored ggml を勝手に更新しない。** `ggml/` はサブモジュールではなくピン留めスナップショット (`331b9cba52b23d895bc4ad218c007eb5e667540f`、PABannier/ggml の sam3-metal-ops ブランチ) を in-tree 展開したもの。
- **リリースバンドルの可搬性条件** (`.github/workflows/release.yml` の `Audit portability` が CI で強制):
  - GLIBC 要求が **2.41 以下** (デプロイ先 Debian 13 trixie の glibc)
  - **libgomp / libstdc++ を動的リンクしない** (`GGML_OPENMP=OFF`、`-static-libstdc++ -static-libgcc`)
  - `GGML_NATIVE=OFF` + AVX2/FMA 固定 (ビルドマシン依存の命令を焼き込まない)
- **Python は `uv` のみ。** 素の `python` / `pip` を使わず、必ず `uv run python` / `uv pip install`。
- **速度は第一級の要求。** 不要なコピーを避け、in-place な ggml 演算 (`_inplace`) を優先し、バックエンドが対応するなら手書き attention より `ggml_flash_attn_ext` を使う。

---

## 3. 参照すべき合意済み資料

| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| 開発規約 (最重要) | `CLAUDE.md` | アーキテクチャ方針、**ggml グラフ分離ルール**、コードスタイル、ビルド/ベンチマーク手順。作業前に必読。 |
| 実装計画 | `PLAN.md` | Phase 0〜10 の全体計画。ディレクトリ構成、重みバイナリ形式、変換スクリプト仕様、各 Phase の検証基準。 |
| 利用者向け README | `README.md` | Quick Start (pixi / プリビルド / 手動ビルド)、ベンチマーク結果、Model Zoo、Feature Matrix。 |
| リリース同梱 README | `README-release.md` | 配布バンドルの中身と実行要件 (AVX2+FMA、glibc >= 2.38、ffmpeg)。バンドルに `README.md` として同梱される。 |
| 作業記録 | `diary/workdoc_Sep06-2026_vendor_pixi_release.md` | vendor 化 / pixi / プリビルド作業の目的分析・手順 11 件のチェックリスト・作業記録。 |
| 公開 API | `sam3.h` | `sam3_load_model` / `sam3_encode_image` / `sam3_segment_pcs` / `sam3_segment_pvs` / `sam3_create_tracker` / `sam3_track_frame` など。`sam3_test_*` は検証用の内部 API。 |
| CI / リリース | `.github/workflows/release.yml`, `.github/workflows/ci.yml` | リリースは Linux x86_64 のみ。CI はビルド検証のみ (macOS/Linux/Windows)。 |
| 上流実装 | https://github.com/facebookresearch/sam3 | テンソル形状・演算順序・活性化関数の**ground truth**。迷ったら Python ソースを読む。 |
| 参考移植 | https://github.com/YavorGIvanov/sam.cpp | SAM 1 の C++/ggml 移植。グラフ構築・重み読み込みのパターン参考。 |
| ggml 実例 | `ggml/examples/` | ピン留めバージョンに対して常に正しい API 使用例 (backend init、`ggml_gallocr`、`ggml_backend_graph_compute`)。 |

**既知課題リスト:** `未確認` — 専用ファイルは存在しない。次に確認すべきは GitHub Issues (`gh issue list`) と `PLAN.md` 各 Phase の検証基準。

---

## 4. タスク境界（任せること / 任せないこと）

### 任せるタスク
- `sam3.cpp` / `sam3.h` の推論パス実装・バグ修正 (グラフ分離ルールを守る前提)
- `examples/` 以下の CLI・GUI ツールの追加や改善
- ggml カーネル選択の最適化、プロファイリング (`sam3_profile_edgetam`、`sam3_benchmark`)
- ビルドシステム (`CMakeLists.txt`、`pixi.toml`)、CI/リリースワークフローの整備
- 重み変換スクリプト (`convert_*_to_ggml.py`) の修正 — 実行は `uv run python`
- `tests/` 以下の Python 比較スクリプトによる数値検証
- ドキュメント (`README.md`、`docs/`、`diary/` の作業書) の更新

### 任せないタスク
- **`ggml/` 配下の直接編集** — ピン留めスナップショット。修正が必要なら上流と pin の更新方針を人間に確認する。
- **`models/*.ggml` の差し替え・再生成** — リポジトリ同梱の EdgeTAM 重み (計 64MB) は意図的に vendor されている。
- **グラフ分離ルールを崩す「最適化」** — 複数ステージを 1 グラフに統合するのは禁止。速くなるように見えて出力が壊れる。
- **リリースの公開 (`v*` タグの push)** — 公開リポジトリの Release ページに出る不可逆操作。人間の明示的な指示が必要。
- **`main` への直接 push / マージ** — 現状の作業は `develop` 上。マージ可否は人間が判断する。
- **会話ログ・PII のコミット** — `temp/` はグローバル gitignore 済み。セッション抽出物は public リポジトリに入れない (`diary/` には作業書のみ置く運用)。
- **新規サードパーティ依存の追加** — 依存構成は固定。

---

## 5. インタラクション方針

- **回答スタイル:** 日本語。見出し + 箇条書き/表を優先し、冗長な散文を避ける。技術用語・コード識別子は原語のまま。
- **回答手順:** 前提 (何を確認したか) → 実施内容 → 検証結果 → 残作業・判断が必要な点、の順。
- **禁止事項・注意:**
  - **未実行のコマンドを「検証済み」と書かない。** 実行して終了コードと出力を確認したものだけを確認済みとする。
  - 出力が途中で切れている場合、それを「正常終了」と読み違えない (実際にこのリポジトリで SIGPIPE 失敗を出力切れと誤読した事例あり。`diary/` 参照)。
  - 数値的に壊れる変更は静かに通るため、推論パスを触ったら必ず既知スコアと突き合わせる。
- **秘匿情報の扱い:** 認証情報・トークン・鍵をコミットやドキュメントに含めない。会話ログや PII を public リポジトリに置かない。ローカル絶対パス (`/home/<user>/...`) の記載は最小限にする。

---

## 6. 試行タスク（オンボーディング演習）

1. **ビルドと推論の再現** — `pixi run demo-cpu` を実行し、`output/mask.png` が生成され、標準出力に `Detections: 1` と `score=0.511` が出ることを確認する。
   - `確認済み` (2026-09-06、Linux x86_64 + debian:13 コンテナの双方で `score=0.511 iou=0.511 box=(91.0, 217.0, 347.0, 377.0)`)
2. **動画パスの確認** — `pixi run benchmark-cpu` を実行し、`SUMMARY: 3 runs, 3 OK, 0 FAIL` 相当が出ることを確認する。ffmpeg は pixi 環境が提供する。
   - `確認済み` (2026-09-06、EdgeTAM 3 種で `3/3 OK`。単一モデル 3 フレームでも `1 OK / 0 FAIL`)
3. **可搬性の理解** — `pixi run release-build` でバンドルを作り、`objdump -T dist/.../bin/sam3_seg | grep GLIBC_` で最大 GLIBC 要求を求め、`ldd` に libgomp / libstdc++ が現れないことを確認する。なぜこの 2 条件が必要かを `README-release.md` と `release.yml` の `Audit portability` から説明できるようにする。
   - `確認済み` (ローカル conda ツールチェーンでは 2.27、CI の ubuntu-24.04 ビルドでは 2.38。いずれもデプロイ先の 2.41 以下)
4. **グラフ分離ルールの読解** — `CLAUDE.md` の CRITICAL 章を読み、`sam3_segment_pcs` (5 サブグラフ) が「なぜ」1 グラフではないのかを説明できるようにする。

---

## 7. 運用ルール・変更管理

- **ドキュメント更新時の記載ルール:** 事実には `確認済み` / `未検証` / `推定` を付す。確認済みには確認日と根拠 (コマンド・出力・run ID) を併記する。
- **TBD の扱い:** 不明な項目は空欄にせず `未確認` と書き、**次に確認すべき情報源**を添える。
- **レビュー/承認フロー:** `未確認` — ブランチ保護ルールや必須レビューは設定を確認していない。次に確認するのは `gh api repos/yuki-inaho/sam3.cpp/branches/main/protection`。現状の運用実績としては、作業は `develop` で行い、`main` へのマージとタグ打ちは人間が判断している。
- **作業書の運用:** 大きな作業は `temp/workdoc_<Mon><D>-<YYYY>_<slug>.md` に作業書を作成し、完了後に `diary/` へ複製して追跡する (`diary/workdoc_Sep06-2026_vendor_pixi_release.md` が実例)。
- **リリース手順:**
  1. `develop` / `main` への push で Release ワークフローの build + verify が走る (**公開はされない**)
  2. グリーンを確認してから `git tag vX.Y.Z && git push origin vX.Y.Z`
  3. `release` ジョブは `needs: [build, verify-linux]` かつタグ ref 限定。**ターゲット環境で動作実証されていないバンドルは Release ページに出ない**
- **その他:** `.gitignore` は `models/` と `data/` を原則除外しつつ、同梱対象のファイルだけを `!` で復活させている。ファイルを増やす場合はこの構造を壊さない。

---

### 付録: 参考情報

**主要リポジトリ/ディレクトリ**

| パス | 役割 |
|------|------|
| `sam3.cpp` / `sam3.h` | ライブラリ本体と公開 API (この 2 ファイルが実装のすべて) |
| `ggml/` | vendored ggml (pin `331b9cba`、サブモジュールではない) |
| `examples/` | `sam3_seg` / `benchmark` / `profile_edgetam` / `quantize` / GUI 2 種 |
| `models/` | EdgeTAM 重み `edgetam_{f16,q8_0,q4_0}.ggml` (計 64MB、同梱) |
| `data/` | `test_image.jpg` (猫 2 匹) / `test_video.mp4` (同梱) |
| `tests/` | Python 数値比較スクリプト群 + C++ テスト (`-DSAM3_BUILD_TESTS=ON`) |
| `scripts/` | `package_release.sh` / `download_model.sh` / `download_test_data.sh` / `verify_tokenizer.py` |
| `stb/` | stb_image / stb_image_write |
| `diary/` | 作業書のアーカイブ |
| `temp/` | 作業用 (グローバル gitignore 済み、コミットしない) |

**代表的なコマンド**

```bash
# ビルド (pixi が cmake/ninja/コンパイラ/ffmpeg を用意する)
pixi run build

# ヘッドレス CPU デモ -> output/mask.png
pixi run demo-cpu

# エンコーダのステージ別レイテンシ
pixi run profile-cpu

# 動画トラッキングのベンチマーク
pixi run benchmark-cpu

# 可搬バンドル -> dist/sam3-linux-x86_64-<version>.tar.gz
pixi run release-build

# pixi を使わない場合
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel

# テストを含める
cmake -B build -DSAM3_BUILD_TESTS=ON

# ベンチマークの高速イテレーション (同梱の EdgeTAM 3 種のみ)
./build/examples/sam3_benchmark --models-dir models --video data/test_video.mp4 \
    --filter edgetam --cpu-only --n-frames 3
```

**依存ライブラリ**

- ライブラリ本体: ggml (in-tree) / stb (in-tree) / C++14 標準ライブラリ **のみ**
- examples 専用: SDL2 + ImGui (見つからなければ GUI 例をスキップ)
- 実行時 (動画機能のみ): `ffmpeg` CLI が PATH 上にあること
- ビルド環境 (pixi): `cmake>=3.31` / `ninja>=1.11` / `cxx-compiler>=1.7` / `ffmpeg>=6.0` / `patchelf>=0.17`
- Python: `uv` 管理 (`pyproject.toml` / `uv.lock`)。重み変換とテスト用。

**CI / リリース状況** (`確認済み` 2026-09-06)

| 項目 | 状態 |
|------|------|
| Release ワークフロー | 登録・実行済み。最新 run `34021512702` は **成功** (build ✅ / verify ✅ / release skipped) |
| Release の対象 | **Linux x86_64 のみ**。macOS / arm64 / Windows は release.yml から除外済み |
| verify の内容 | `debian:13` コンテナで実バンドルを実行し、`sam3_seg` の mask 生成と `sam3_benchmark` の `0 FAIL` を assert |
| CI ワークフロー (`ci.yml`) | **`未検証`** — `main` への push / PR でのみ起動する設定のため、このリポジトリでは一度も実行されていない |
| Metal バックエンド | **`未検証`** — 検証環境が Linux のため、この作業では確認していない |
| SAM 3 本体モデル | **`未検証`** — 同梱されているのは EdgeTAM のみ。テキストプロンプト (PCS) 経路は別途重みの取得が必要 |

**デプロイ先環境** (利用者から提示された実測値)

| 項目 | 値 |
|------|-----|
| OS / Kernel | Debian GNU/Linux 13 (trixie) / Linux 6.18.35 |
| アーキテクチャ | x86_64 (AMD EPYC 9V74、論理 5 CPU、KVM) |
| 命令セット | SSE〜SSE4.2 / AVX / AVX2 / AVX-512 / FMA / VNNI |
| glibc | 2.41 |
| ツールチェーン | GCC 14.2 / Clang 17 (Clang 用 OpenMP なし) / CMake 3.31.6 / Ninja 1.12.1 |
| 未導入 | Eigen3 / fmt / spdlog / GoogleTest / Catch2 |

**連絡先/責任者:** `未確認` — リポジトリオーナーは GitHub `yuki-inaho`。責任分担の定義ファイルは存在しない。

**既知のドキュメント乖離** (`確認済み` 2026-09-06)

| 箇所 | 内容 |
|------|------|
| `CLAUDE.md` の「Quick-iteration recipe」 | `--filter-prec f16,q4_0` を案内しているが、`examples/benchmark.cpp:461-470` にこのフラグは実装されていない。実在するのは `--models-dir` / `--video` / `--point-x` / `--point-y` / `--n-frames` / `--n-threads` / `--encode-img-size` / `--cpu-only` / `--gpu-only` / `--filter` / `--help`。 |
| `CLAUDE.md` の Build 節 | macOS 前提の `make -j$(sysctl -n hw.ncpu)` を案内している。Linux では `sysctl` のこの用法は使えない。 |
| `README.md` の Model Zoo | Hugging Face 上の 52 モデルを参照しているが、リポジトリに同梱されているのは EdgeTAM 3 種のみ。 |

> 上記は本ドキュメント作成時に実ファイルと突き合わせて検出したもの。修正する場合は該当ファイル側を直し、この表からも削除すること。
