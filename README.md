# ナンバープレート消しツール (license-plate-eraser)

写真・動画に写っている車のナンバープレートを AI で自動検出して、モザイク・白塗り・AI 修復のどれかで消すデスクトップアプリです。
ファイルをドラッグ＆ドロップするだけで、まとめて処理できます。

## 主な機能

- **自動検出**: YOLO ベースのナンバープレート検出モデルで、プレートの位置を自動で見つけます
- **3 種類の消し方**
  - **モザイク**（推奨・高速）
  - **白塗り**
  - **AI 修復**（高品質・低速）: [LaMa](https://github.com/advimman/lama) で、プレート部分を周りの背景に合わせて描き直します
- **動画対応**: フレームごとに検出し、検出が一瞬途切れても直前の位置を引き継いで、モザイクがちらつかないようにしています。音声も残ります
- **一括処理**: 複数ファイルやフォルダをまとめてドロップできます
- **詳細設定**: 検出感度とモザイクの粗さを調整できます
- **GPU 対応**: Apple Silicon (MPS) / CUDA があれば自動で使い、なければ CPU で動きます

### 対応フォーマット

| 種類 | 拡張子 |
| --- | --- |
| 画像 | `.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tif` `.tiff` |
| 動画 | `.mp4` `.mov` `.m4v` `.avi` `.mkv` `.webm` |

### 出力先

処理したファイルは `~/Downloads` に `元のファイル名_masked` という名前で保存されます。動画は `.mp4` で出力されます。
元のファイルは変更しません。

## 使い方（ソースから実行）

Python 3.10 で動作確認しています。macOS での利用を前提にしています。

```bash
git clone https://github.com/hamadareo/license-plate-eraser.git
cd license-plate-eraser
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./run.command
```

起動したら、ウィンドウに写真・動画をドラッグ＆ドロップするか「ファイルを選択…」を押してください。

> [!NOTE]
> 「AI 修復」を初めて使うときは、LaMa モデルのダウンロードに数分かかることがあります。

## アプリのビルド

### ローカル（macOS）

```bash
./venv/bin/pyinstaller "ナンバープレート消しツール.spec" --noconfirm
```

`dist/ナンバープレート消しツール.app` ができます。

### GitHub Actions

`.github/workflows/build.yml` を手動実行 (workflow_dispatch) すると、macOS / Windows / Linux 向けのビルドが Artifacts として作られます。
Windows / Linux 版も作れますが、Finder で開くなど一部の機能は macOS 前提です。

## 注意事項

- 自動検出は完璧ではありません。角度・距離・ブレ・逆光などで**消し漏れが出ることがあります**。SNS などに公開する前に、必ず結果を目で確認してください
- このツールを使ったことによる損害について、作者は責任を負いません（詳しくは [LICENSE](LICENSE) をご覧ください）

## ライセンス

このソフトウェアは **GNU Affero General Public License v3.0 (AGPL-3.0)** で公開しています。全文は [LICENSE](LICENSE) をご覧ください。

Copyright (C) 2026 hamadareo

### 利用しているサードパーティ製のソフトウェア・モデル

| 名前 | 用途 | ライセンス |
| --- | --- | --- |
| [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) (`license-plate-finetune-v1s.pt`) | ナンバープレート検出モデル（`models/` に同梱） | AGPL-3.0 |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO の推論 | AGPL-3.0 |
| [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) / [LaMa](https://github.com/advimman/lama) | AI 修復（モデルは初回実行時にダウンロード） | Apache-2.0 |
| [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) | GUI | GPL-3.0 |
| [PyTorch](https://pytorch.org/) | 推論 | BSD-3-Clause |
| [OpenCV](https://opencv.org/) | 画像・動画の処理 | Apache-2.0 |
| [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) (FFmpeg) | 動画の音声を付け直す | BSD-2-Clause（FFmpeg 本体は LGPL/GPL） |

`models/license-plate-finetune-v1s.pt` は、上記の Hugging Face リポジトリで配布されているファイルを改変せずに同梱しています（SHA-256: `95e50c25ab7066dd0ca5aec18fa80349676db08697780d1149576461174d2381`）。
