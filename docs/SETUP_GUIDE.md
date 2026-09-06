# 環境セットアップ・オンボーディングガイド

**作成日**: 2026-09-06
**対象**: 新しいセッションの LLM エージェント / 新規参加の開発メンバー
**プロジェクト**: sam3.cpp (`yuki-inaho/sam3.cpp`)
**目的**: 同じ実行環境を再現し、**現状の生成物とドキュメントを決まった順序で確認したうえで**作業を継続できるようにする。

> このガイドは「**何をどの順でやるか**」を扱う。「**何を知っておくべきか**」(制約・タスク境界・禁止事項) は
> [`docs/ONBOARDING.md`](./ONBOARDING.md) にある。**両方に目を通すこと。**
> 事実には `確認済み` / `未検証` / `未確認` を付す。確認済みは 2026-09-06 時点の実測。

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [現在のプロジェクト状態](#2-現在のプロジェクト状態)
3. [前提条件の確認](#3-前提条件の確認)
4. [環境セットアップ手順](#4-環境セットアップ手順)
5. [動作確認](#5-動作確認)
6. [トラブルシューティング](#6-トラブルシューティング)
7. [次のステップ](#7-次のステップ)
8. [環境セットアップ完了チェックリスト](#8-環境セットアップ完了チェックリスト)
9. [更新履歴](#9-更新履歴)

---

## 1. プロジェクト概要

### プロジェクト名

**sam3.cpp**
Meta の SAM 3 (Segment Anything Model 3) を **C++14 + ggml** に移植した、画像・動画セグメンテーションの推論エンジン。PyTorch ランタイムなしで CPU と Metal (Apple GPU) 上で動作する。

### 最終目標

PyTorch に依存せず SAM 系モデルを動かせる可搬なライブラリを提供する。
リポジトリは **self-contained** (clone → build → run) を設計目標とし、ggml・EdgeTAM 重み・サンプルデータをすべて同梱する。
デプロイ形態として、**Debian 13 (trixie) x86_64 コンテナでそのまま動くプリビルドバイナリ**を GitHub Releases で配布する。

### 主要コンポーネント

* **ライブラリ本体** — `sam3.cpp` (14,439 行) + 公開 API `sam3.h` (519 行)。この 2 ファイルが実装のすべて。
* **ヘッドレス CLI** — `sam3_seg` (点/ボックス → mask PNG)、`sam3_benchmark` (動画トラッキング計測)、`sam3_profile_edgetam` (ステージ別レイテンシ)、`sam3_quantize` (量子化)
* **GUI 例** — `sam3_image` / `sam3_video`。**SDL2 と OpenGL が両方見つかったときだけビルドされる** (`examples/CMakeLists.txt:17-35`)。ヘッドレス環境では自動的にスキップされ、これは異常ではない。
* **ビルド/実行環境** — `pixi` (conda-forge から cmake/ninja/コンパイラ/ffmpeg/patchelf を供給)。Python 側 (重み変換) だけ `uv`。
* **リリース自動化** — `.github/workflows/release.yml`。Linux x86_64 バンドルをビルドし、**`debian:13` コンテナで実際に実行して検証してから**公開する。

---

## 2. 現在のプロジェクト状態

### 完了済み

| 分類 | 状態 | 説明 |
|------|------|------|
| **ggml の vendor 化** | 🟢 | サブモジュール廃止。pin `331b9cba` を in-tree 展開 (1864 ファイル)。`.gitmodules` は削除済み |
| **モデル同梱** | 🟢 | `models/edgetam_{f16,q8_0,q4_0}.ggml` (計 64MB) |
| **サンプルデータ同梱** | 🟢 | `data/test_image.jpg` (猫 2 匹) / `data/test_video.mp4` |
| **pixi 環境** | 🟢 | `pixi.toml` + `pixi.lock`。タスク 5 種 (`build` / `demo-cpu` / `profile-cpu` / `benchmark-cpu` / `release-build`) |
| **ヘッドレス CLI** | 🟢 | `examples/sam3_seg.cpp` を新設 (GUI 依存なし) |
| **リリース自動化** | 🟢 | run `34021512702` で **build ✅ / verify ✅**。`debian:13` 実機検証込み |
| **ドキュメント** | 🟢 | `CLAUDE.md` / `PLAN.md` / `README.md` / `docs/ONBOARDING.md` / 本書 / `diary/` の作業書 |

### 未実装・これから着手する項目

* **`main` へのマージ** — 作業は全部 `develop`。`main` は `01832ef` のまま、`develop` が先行している (差分は `git log --oneline main..develop` で確認)。マージは人間の判断待ち。
* **リリースの公開** — `v*` タグ未作成のため、**Releases ページにはまだ成果物がない**。README の「Option B: prebuilt binary」は現時点で空振りする。
* **`ci.yml` の実行** — `main` への push / PR でのみ起動する設定のため、**このリポジトリでは一度も走っていない** (`未検証`)。
* **Metal バックエンド** — 検証環境が Linux のため `未検証`。
* **SAM 3 本体モデル (テキストプロンプト / PCS 経路)** — 同梱は EdgeTAM のみ。別途重みの取得が必要 (`未検証`)。

### 生成物 (ビルド成果物) の現状

| パス | 内容 | git 追跡 |
|------|------|----------|
| `build/examples/` | `sam3_seg` / `sam3_benchmark` / `sam3_profile_edgetam` / `sam3_quantize` | 除外 (`build/`) |
| `build-release/` | `release-build` タスクの中間生成物 | 除外 |
| `dist/sam3-linux-x86_64-*.tar.gz` | ローカル生成の可搬バンドル (約 77MB) | 除外 (`dist/`) |
| `dist/sam3-linux-x86_64-bundle/` | 上の展開済みディレクトリ | 除外 |
| `output/mask.png` | `demo-cpu` の出力 | 除外 (`output/`) |
| `.pixi/` (約 1.5GB) | pixi が解決した環境実体 | 除外 |

> **注意**: ローカルの `package_release.sh` が作るディレクトリ名は `sam3-linux-x86_64-bundle`、
> CI が公開するアーカイブの中身は `sam3-linux-x86_64` で**名前が異なる**。README は CI 側 (公開される方) に合わせてある。

### 重要なファイル／ディレクトリ

```text
/home/<user>/Project/sam3.cpp/
├── CLAUDE.md                 # 開発規約。ggml グラフ分離ルール(最重要)を含む。作業前に必読
├── PLAN.md                   # Phase 0〜10 の実装計画・重み形式・検証基準
├── README.md                 # 利用者向け。Quick Start / ベンチマーク / Model Zoo
├── README-release.md         # 配布バンドルに同梱される README
├── sam3.cpp / sam3.h         # ライブラリ本体と公開 API
├── pixi.toml / pixi.lock     # ビルド・実行環境の定義
├── pyproject.toml / uv.lock  # Python (重み変換・数値検証) 用
├── docs/
│   ├── ONBOARDING.md         # 制約・タスク境界・禁止事項 (何を知るべきか)
│   └── SETUP_GUIDE.md        # 本書 (何をどの順でやるか)
├── diary/
│   └── workdoc_Sep06-2026_vendor_pixi_release.md   # vendor化/pixi/リリース作業の作業書
├── ggml/                     # vendored ggml (pin 331b9cba)。直接編集しない
├── models/                   # edgetam_{f16,q8_0,q4_0}.ggml (同梱)
├── data/                     # test_image.jpg / test_video.mp4 (同梱)
├── examples/                 # sam3_seg / benchmark / profile / quantize / GUI 2種
├── scripts/                  # package_release.sh ほか
├── tests/                    # Python 数値比較スクリプト + C++ テスト
└── temp/                     # 作業用。グローバル gitignore 済み。コミットしない
```

---

## 3. 前提条件の確認

**セットアップの前に、まず自分がどこにいるかを確認する。**

### 3.1 システム情報の確認

```bash
cat /etc/os-release | grep -E "^(NAME|VERSION)="
uname -r
nproc
free -h
pwd
ldd --version | sed -n 1p    # ビルドしたバイナリの最小 glibc を決める値
```

**開発機の実測値** (`確認済み` 2026-09-06):

```text
NAME="Ubuntu"
VERSION="24.04.4 LTS (Noble Numbat)"
6.8.0-138-generic
12                       # cores
15Gi total               # memory
/home/<user>/Project/sam3.cpp
ldd (Ubuntu GLIBC 2.39-0ubuntu8.8) 2.39
```

**デプロイ先環境** (利用者提示の実測値。開発機とは別物なので混同しない):

| 項目 | 値 |
|------|-----|
| OS / Kernel | Debian GNU/Linux 13 (trixie) / Linux 6.18.35 |
| CPU | AMD EPYC 9V74 (論理 5 CPU、KVM)、AVX2 / AVX-512 / FMA / VNNI |
| glibc | **2.41** |
| ツールチェーン | GCC 14.2 / Clang 17 (Clang 用 OpenMP なし) / CMake 3.31.6 / Ninja 1.12.1 |
| 未導入 | Eigen3 / fmt / spdlog / GoogleTest / Catch2 |

### 3.2 必須ツールの存在確認

```bash
# 必須
which pixi  && pixi --version
which git   && git --version

# 推奨 (リリース検証に docker、CI 確認に gh)
which docker && docker --version && docker info >/dev/null 2>&1 && echo "docker daemon OK"
which gh     && gh --version | sed -n 1p

# Python 側だけで使う (重み変換・数値検証)
which uv && uv --version
```

**開発機の実測値** (`確認済み`):

```text
pixi 0.76.2
docker 29.4.0  (daemon OK)
uv 0.9.7
```

> **cmake / ninja / コンパイラ / ffmpeg は個別に入れなくてよい。** pixi が供給する。
> ホストに古い `ffmpeg` (開発機では n3.4.13) が入っていても、pixi 環境内の ffmpeg (>=6.0) が使われる。
> pixi を使わない場合のみ、CMake >= 3.31 / Ninja / C++14 コンパイラ / ffmpeg CLI を自前で用意する。

### 3.3 Git ブランチ・コミット確認

```bash
git branch --show-current
git log --oneline -1
git status --porcelain
git log --oneline main..$(git branch --show-current)   # main との差分
```

**期待される状態** (`確認済み` 2026-09-06):

```text
develop
fdd1495 docs: add docs/ONBOARDING.md, an LLM onboarding summary
?? .serena/        # ツールのローカル状態。コミットしない
```

> `main` ではなく **`develop` で作業する**。`main` への直接 push はしない (`docs/ONBOARDING.md` 第 4 章)。

---

## 4. 環境セットアップ手順

### 4.1 システムパッケージ

**通常は不要。** pixi がビルドツールチェーンを供給する。
pixi を使わずに素の cmake でビルドする場合のみ、以下に相当するものを用意する。

```bash
# 例: Debian/Ubuntu で pixi を使わない場合
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential cmake ninja-build ffmpeg
```

**注意事項:**

* `ffmpeg` は**動画機能の実行時のみ**必要 (`sam3_benchmark`、`sam3_video`)。画像だけなら不要。
* SDL2 / OpenGL は GUI 例のためだけに使う。無くてもライブラリと CLI は完全にビルドできる。

### 4.2 pixi 環境の構築（最重要）

```bash
cd /path/to/sam3.cpp

# 環境の解決と全ターゲットのビルド (configure に依存しているので単独で完結する)
pixi run build
```

`pixi.toml` が供給する依存:

* `cmake >= 3.31` — ビルド
* `ninja >= 1.11` — ビルド
* `cxx-compiler >= 1.7` — conda-forge のツールチェーン。**古い sysroot を持つため、生成物の最小 glibc が下がる** (ローカルビルドで GLIBC 2.27 要求)
* `ffmpeg >= 6.0` — 動画デコード (実行時)
* `patchelf >= 0.17` — `package_release.sh` が RPATH を除去するのに使う

**初回は `.pixi/` の解決に時間がかかり、約 1.5GB を消費する** (`確認済み`)。

### 4.3 pixi を使わない場合

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# テストも含める場合
cmake -B build -DSAM3_BUILD_TESTS=ON
```

### 4.4 Python 環境（重み変換・数値検証のときだけ）

```bash
uv sync
uv run python convert_edgetam_to_ggml.py --help
```

> **素の `python` / `pip` を使わない。** 必ず `uv run python` / `uv pip install` (`CLAUDE.md`)。

### 4.5 ターゲット環境の検証用（リリースに触るときだけ）

```bash
docker pull debian:13     # デプロイ先と同じ glibc 2.41 環境
```

---

## 5. 動作確認

**ここを通らないまま実装に進まないこと。** 推論パスの破壊はクラッシュせず数値だけ壊れるため、
「既知の正解値と一致するか」が唯一の早期検知手段になる。

### 5.1 画像セグメンテーション（最重要の基準値）

```bash
pixi run demo-cpu
```

**期待される出力** (`確認済み` 2026-09-06、開発機と `debian:13` の双方で一致):

```text
Detections: 1
Best:       score=0.511 iou=0.511 box=(91.0, 217.0, 347.0, 377.0)
Mask:       output/mask.png (640x480)
```

> **`score=0.511` と box の 4 値が基準値。** 推論パスを変更したら必ずこれと突き合わせる。
> 値がずれたら、その変更は数値を壊している。

### 5.2 動画トラッキング（ffmpeg 経路）

```bash
pixi run benchmark-cpu
```

**期待される出力** (`確認済み`):

```text
SUMMARY: 3 runs, 3 OK, 0 FAIL
```

> `sam3_benchmark` は**個別 run が失敗しても終了コード 0 を返す**。
> 成否は必ず `SUMMARY` 行の `0 FAIL` で判定する (CI も同じ方法で assert している)。

### 5.3 ステージ別プロファイル

```bash
pixi run profile-cpu
```

エンコーダのステージごとのレイテンシ内訳が出る。最適化の前後比較に使う。

### 5.4 可搬バンドルの生成と検査（リリースに触るときだけ）

```bash
pixi run release-build

cd dist/sam3-linux-x86_64-bundle
for f in bin/*; do
  echo "--- $f"
  ldd "$f"
  objdump -T "$f" | sed -n 's/.*GLIBC_\([0-9.]*\).*/\1/p' | sort -uV | tail -1
done
```

**期待される結果** (`確認済み`):

* 最大 GLIBC 要求: ローカル (conda ツールチェーン) で **2.27**、CI (ubuntu-24.04) で **2.38**。いずれもデプロイ先の 2.41 以下
* `ldd` に **libgomp と libstdc++ が現れない**こと

### 5.5 ターゲット環境での実行確認

```bash
docker run --rm -v "$PWD/dist/sam3-linux-x86_64-bundle:/bundle" -w /bundle debian:13 \
  bash -euo pipefail -c '
    ldd --version | sed -n 1p
    ./bin/sam3_seg --model models/edgetam_q8_0.ggml \
                   --image data/test_image.jpg --x 315 --y 250 \
                   --out /tmp/mask.png --cpu
    test -s /tmp/mask.png && echo "OK: $(stat -c %s /tmp/mask.png) bytes"
  '
```

**期待される結果** (`確認済み`): `ldd (Debian GLIBC 2.41-...) 2.41` と `score=0.511`、mask 3,627 bytes。

> `head -1` ではなく `sed -n 1p` を使っている理由はセクション 6 の問題 5 を参照。

### 5.6 CI の状態確認

```bash
gh run list --workflow=release.yml --limit 3
gh run view <run-id> --json status,conclusion,jobs \
  --jq '"\(.status) \(.conclusion)", (.jobs[] | "  \(.name): \(.conclusion)")'
```

**期待される状態** (`確認済み`): 最新 run `34021512702` が
`build: success` / `verify-linux: success` / `release: skipped` (タグ ref でないため skipped が正常)。

---

## 6. トラブルシューティング

### 問題1: `sam3_image` / `sam3_video` がビルドされない

**原因**: SDL2 または OpenGL が見つからない。`examples/CMakeLists.txt:17-35` は**両方揃ったときだけ** GUI 例を追加する。
**対処**: ヘッドレス環境では**これが正常**。`Skipping GUI examples` は異常ではない。GUI が必要なら SDL2 と OpenGL の開発パッケージを入れる。

### 問題2: 動画系が「ffmpeg がない」で落ちる / デコードが全滅する

**原因**: PATH 上の ffmpeg が古い、または不在。
**対処**: pixi 環境内の ffmpeg を使っているか確認する。

```bash
pixi run bash -c 'which ffmpeg && ffmpeg -version | sed -n 1p'
```

> 既知の落とし穴: **ffmpeg 8 で `-vsync` オプションが廃止**され、動画デコードが全滅した実績がある
> (`sam3.cpp` から削除済み、新旧両対応)。動画だけ壊れたらまず ffmpeg のバージョン差を疑う。

### 問題3: 推論結果が壊れている（クラッシュはしない）

**原因**: **ggml のグラフ分離ルール違反**の可能性が高い。state tensor (`state.neck_trk[*]` 等) を
グラフのオペランドに渡すと、`ggml_build_forward_expand` が依存ツリー全体を辿り、
グラフアロケータが下流でまだ必要なバッファを上書きする。
**対処**: `CLAUDE.md` の「ggml graph isolation (CRITICAL)」章を読み、
新しい入力テンソルを作って CPU 経由でコピーする正しい書き方に直す。
`score=0.511` の基準値 (5.1) と突き合わせて回帰を特定する。

### 問題4: バンドルがデプロイ先で動かない

**対処**: 症状別に切り分ける。

```bash
# GLIBC 要求が高すぎる
objdump -T bin/sam3_seg | sed -n 's/.*GLIBC_\([0-9.]*\).*/\1/p' | sort -uV | tail -1

# libgomp / libstdc++ を引いている
ldd bin/sam3_seg | grep -E "libgomp|libstdc"

# 命令セットが合っていない (Illegal instruction)
# -> GGML_NATIVE=OFF と AVX2/FMA 固定でビルドされているか確認
```

`release.yml` の `Audit portability` ステップが CI でこれらを強制しているので、
ローカルで再現しない場合は CI のログを見る。

### 問題5: `... | head -1` が exit 141 で落ちる

**原因**: SIGPIPE。`ldd --version` のように**複数回に分けて出力する**コマンドを `head -1` に繋ぐと、
`head` が先に終了してパイプを閉じ、残りの書き込みが EPIPE で死ぬ。
`set -o pipefail` があるとこれが 141 としてスクリプト全体を落とす。**タイミング依存で再現しないことがある。**
**対処**: `sed -n 1p` を使う (入力を最後まで読むので競合しない)。実際に CI をこれで落とした実績がある。

### 問題6: `gh workflow run` が `not found on the default branch` で失敗する

**原因**: `workflow_dispatch` は**デフォルトブランチ (`main`) にワークフローが存在しないと使えない**。
現状 `main` は古く、`workflow_dispatch` 付きの `release.yml` を持っていない。
**対処**: `develop` への push で build + verify は自動的に走る (公開はされない)。これで代替する。

---

## 7. 次のステップ

**新しいセッションでは、以下を Step 0 から順に実施する。** 飛ばさないこと。

### Step 0. 立ち位置の把握（コードを読む前）

```bash
git branch --show-current && git log --oneline -3
git status --porcelain
git log --oneline main..$(git branch --show-current)
gh run list --workflow=release.yml --limit 3
```

確認すること: **どのブランチにいるか / 未コミットの変更はあるか / `main` とどれだけ差があるか / CI は緑か**。

### Step 1. ドキュメントをこの順で読む

| 順 | ファイル | 読む理由 |
|----|----------|----------|
| 1 | `CLAUDE.md` | **ggml グラフ分離ルール**を含む開発規約。これを知らずにコードを触ると静かに壊す |
| 2 | `docs/ONBOARDING.md` | 制約・タスク境界・任せてよい/いけない作業・既知のドキュメント乖離 |
| 3 | 本書 (`docs/SETUP_GUIDE.md`) | 手順と基準値 |
| 4 | `diary/workdoc_Sep06-2026_vendor_pixi_release.md` | 直近の大きな作業 (vendor化/pixi/リリース) の目的と経緯 |
| 5 | `PLAN.md` の該当 Phase | 実装に入るときだけ。全部読む必要はない |
| 6 | `README.md` / `README-release.md` | 利用者に何を約束しているか |

### Step 2. 生成物の現状を確認

```bash
ls -la models/ data/                  # 同梱物 (git 追跡下)
ls -la build/examples/ 2>/dev/null    # ビルド済みバイナリ (未ビルドなら空)
ls -la dist/ output/ 2>/dev/null      # ローカル生成物 (git 除外)
du -sh .pixi 2>/dev/null              # pixi 環境の有無
```

**判断**: `build/examples/` が空なら Step 3 へ。埋まっていても、前セッションから
ソースが変わっている可能性があるので Step 3 は実行する (差分ビルドで済む)。

### Step 3. 環境構築とビルド

```bash
pixi run build
```

### Step 4. 基準値の再現（必須ゲート）

```bash
pixi run demo-cpu       # -> score=0.511, box=(91.0, 217.0, 347.0, 377.0)
pixi run benchmark-cpu  # -> SUMMARY: 3 runs, 3 OK, 0 FAIL
```

**ここが一致しない状態で実装を始めない。** 一致しないなら、まず原因の切り分け (セクション 6) を行う。

### Step 5. 作業内容を決める

現時点で残っている項目 (セクション 2 の「未実装」より):

1. **`main` へのマージ** — `develop` の先行コミットをどう扱うか。**人間の判断が必要。**
2. **リリースの公開** — `v*` タグを打つと CI が build → verify → 公開まで実行する。**不可逆かつ人間の判断が必要。**
3. **`ci.yml` の実行確認** — `main` 側でしか起動しないため未検証。マージ時に初めて走る。
4. **Metal バックエンドの検証** — macOS 実機が必要。
5. **SAM 3 本体モデル (PCS / テキストプロンプト経路) の検証** — 重みの入手が必要。

### Step 6. 実装するときのルール

* 推論パスを触ったら **Step 4 の基準値と必ず突き合わせる**。
* ステージを追加するなら **専用の `ggml_context` / `ggml_cgraph` / `ggml_gallocr`** を持たせ、
  ステージ間は CPU 側の `std::vector<float>` で受け渡す。
* `ggml/` を編集しない。`models/*.ggml` を差し替えない。
* 実行して終了コードと出力を確認していないものを「検証済み」と書かない。

### Step 7. リリースに触るとき

```bash
pixi run release-build                    # ローカルバンドル生成
# セクション 5.4 で GLIBC と ldd を検査
# セクション 5.5 で debian:13 実行確認
git push origin develop                   # CI が build + verify を実行 (公開はされない)
gh run list --workflow=release.yml --limit 1
# 緑を確認してから、人間の指示のもとで:
# git tag vX.Y.Z && git push origin vX.Y.Z
```

### Step 8. 便利なコマンド集

```bash
pixi task list                            # 定義済みタスク一覧
pixi run bash -c 'which ffmpeg cmake'     # pixi 環境内のツール解決先
git log --oneline main..develop           # main との差分
gh run list --workflow=release.yml        # CI 履歴
./build/examples/sam3_seg --help
./build/examples/sam3_benchmark --help
```

---

## 8. 環境セットアップ完了チェックリスト

実装・検証作業に進む前に、すべて確認すること。

* [ ] システム情報を確認した (OS / コア数 / メモリ / **ホストの glibc**)
* [ ] 必須ツールが揃っている (`pixi`、`git`。リリース検証なら `docker`、CI 確認なら `gh`)
* [ ] Git ブランチが **`develop`** で、`main` との差分を把握している
* [ ] 未コミットの変更を把握している (`.serena/` は untracked のままでよい)
* [ ] `pixi run build` が成功した
* [ ] GUI 例がスキップされた場合、それが**環境要因の正常動作**だと理解している
* [ ] `pixi run demo-cpu` が **`score=0.511`** と box 4 値を再現した
* [ ] `pixi run benchmark-cpu` が **`0 FAIL`** を返した
* [ ] `CLAUDE.md` の **ggml グラフ分離ルール**を読んだ
* [ ] `docs/ONBOARDING.md` の**タスク境界 (第 4 章)** を読んだ
* [ ] 直近の CI (`gh run list`) の状態を確認した
* [ ] (リリースに触る場合) `debian:13` での実行確認が通った

---

## 9. 更新履歴

* `2026-09-06 初版作成` — テンプレートを sam3.cpp 向けに具体化。開発機とデプロイ先の実測値、基準値 (`score=0.511` / `0 FAIL`)、CI run `34021512702` の結果、既知の落とし穴 (SIGPIPE / ffmpeg 8 の `-vsync` / GUI スキップ / `workflow_dispatch` の制約) を記載。

---

## このドキュメントについて

本ガイドは、新しいセッションや新規参加メンバーが短時間で同一の実行環境を再現し、
**現状の生成物とドキュメントを確認したうえで**既存タスクを引き継げるようにすることを目的としている。

このプロジェクト特有の事情として、**推論パスの破壊はクラッシュではなく数値の劣化として現れる**。
そのため本書ではセクション 5.1 の基準値 (`score=0.511`) を必須ゲートとして扱っている。
環境構築で問題が起きた場合はセクション 6 に加えて、`diary/` の作業書とコミットログを参照すること。
