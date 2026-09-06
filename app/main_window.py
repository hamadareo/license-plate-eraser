from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import inpainter
from .masker import MODE_FILL, MODE_INPAINT, MODE_MOSAIC
from .media import DOWNLOADS_DIR, is_supported
from .worker import BatchWorker, LamaPreloadThread, ModelLoaderThread


class DropArea(QListWidget):
    files_dropped = pyqtSignal(list)
    file_activated = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.setStyleSheet(
            """
            QListWidget {
                border: 2px dashed #9aa0a6;
                border-radius: 10px;
                padding: 6px;
                font-size: 13px;
            }
            """
        )
        self._placeholder: QListWidgetItem | None = None
        self._show_placeholder()
        self.itemDoubleClicked.connect(self._on_item_double_clicked)

    def _show_placeholder(self):
        # QListWidget.clear() destroys the underlying C++ item object, so a
        # previously-stored placeholder reference would dangle after that --
        # always build a fresh one instead of reusing self._placeholder.
        self._placeholder = QListWidgetItem(
            "ここに写真・動画をドラッグ＆ドロップ\n（複数可・フォルダごとも可）\n\n完了した項目はダブルクリックでFinder表示"
        )
        self._placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
        self._placeholder.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.addItem(self._placeholder)

    def _on_item_double_clicked(self, item: QListWidgetItem):
        out_path = item.data(Qt.ItemDataRole.UserRole)
        if out_path:
            self.file_activated.emit(out_path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = []
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_dir():
                paths.extend(sorted(x for x in p.rglob("*") if x.is_file()))
            elif p.is_file():
                paths.append(p)
        files = [str(p) for p in paths if is_supported(p)]
        if files:
            self.files_dropped.emit(files)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ナンバープレート消しツール")
        self.resize(720, 760)
        self.setMinimumSize(600, 560)

        self.detector = None
        self.worker: BatchWorker | None = None
        self.pending_files: list[str] = []
        self.item_rows: dict[str, QListWidgetItem] = {}
        self.lama_ready = False
        self.lama_loading = False
        self.cancel_requested = False
        self._current_batch_files: list[str] = []

        self._mode_labels = {
            MODE_MOSAIC: "モザイク",
            MODE_FILL: "白塗り",
            MODE_INPAINT: "AI修復",
        }

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("ナンバープレート消しツール")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        root.addWidget(title)

        self.status_label = QLabel("モデルを読み込み中…")
        self.status_label.setStyleSheet("color: #444;")
        root.addWidget(self.status_label)

        self.drop_area = DropArea()
        self.drop_area.files_dropped.connect(self.on_files_added)
        self.drop_area.file_activated.connect(self.reveal_in_finder)
        root.addWidget(self.drop_area, stretch=1)

        hint = QLabel("ヒント: 完了した項目はダブルクリックでFinderに表示できます")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(hint)

        btn_row = QHBoxLayout()
        self.select_btn = QPushButton("ファイルを選択…")
        self.select_btn.clicked.connect(self.on_select_files)
        btn_row.addWidget(self.select_btn)

        self.open_downloads_btn = QPushButton("Downloadsフォルダを開く")
        self.open_downloads_btn.clicked.connect(self.on_open_downloads)
        btn_row.addWidget(self.open_downloads_btn)
        root.addLayout(btn_row)

        btn_row2 = QHBoxLayout()
        self.clear_btn = QPushButton("完了済みをリストから消す")
        self.clear_btn.clicked.connect(self.on_clear_finished)
        btn_row2.addWidget(self.clear_btn)

        btn_row2.addStretch(1)

        self.cancel_btn = QPushButton("キャンセル")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.on_cancel_clicked)
        btn_row2.addWidget(self.cancel_btn)
        root.addLayout(btn_row2)

        # -- masking mode --------------------------------------------------
        mode_box = QGroupBox("消し方")
        mode_row = QHBoxLayout(mode_box)
        self.mode_group = QButtonGroup(self)
        self.rb_mosaic = QRadioButton("モザイク（推奨・高速）")
        self.rb_fill = QRadioButton("白塗り")
        self.rb_inpaint = QRadioButton("AI修復で消す（高品質・低速）")
        self.rb_mosaic.setChecked(True)
        for i, rb in enumerate([self.rb_mosaic, self.rb_fill, self.rb_inpaint]):
            self.mode_group.addButton(rb, i)
            mode_row.addWidget(rb)
        self.rb_inpaint.toggled.connect(self.on_inpaint_toggled)
        root.addWidget(mode_box)

        # -- fine tuning ------------------------------------------------------
        detail_box = QGroupBox("詳細設定")
        detail_form = QFormLayout(detail_box)
        detail_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(25)
        self.conf_label = QLabel("0.25")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(f"{v / 100:.2f}")
        )
        conf_row = QHBoxLayout()
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(self.conf_label)
        detail_form.addRow("検出感度（低いほど見逃しにくいが誤検出も増える）", conf_row)

        self.mosaic_slider = QSlider(Qt.Orientation.Horizontal)
        self.mosaic_slider.setRange(3, 15)
        self.mosaic_slider.setValue(6)
        self.mosaic_label = QLabel("6")
        self.mosaic_slider.valueChanged.connect(
            lambda v: self.mosaic_label.setText(str(v))
        )
        mosaic_row = QHBoxLayout()
        mosaic_row.addWidget(self.mosaic_slider)
        mosaic_row.addWidget(self.mosaic_label)
        detail_form.addRow("モザイクの粗さ（数値が小さいほど粗く、隠す効果が高い）", mosaic_row)

        root.addWidget(detail_box)
        self.rb_mosaic.toggled.connect(
            lambda checked: self.mosaic_slider.setEnabled(checked) or self.mosaic_label.setEnabled(checked)
        )

        # -- progress -----------------------------------------------------
        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        progress_row.addWidget(self.progress, stretch=1)

        self.batch_label = QLabel("")
        self.batch_label.setStyleSheet("color: #444;")
        progress_row.addWidget(self.batch_label)
        root.addLayout(progress_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        self.log.setStyleSheet("font-family: Menlo, monospace; font-size: 11px; color: #555;")
        root.addWidget(self.log)

        self.drop_area.setEnabled(False)
        self.select_btn.setEnabled(False)

        self.loader = ModelLoaderThread()
        self.loader.loaded.connect(self.on_model_loaded)
        self.loader.failed.connect(self.on_model_failed)
        self.loader.start()

    # -- model loading -----------------------------------------------
    def on_model_loaded(self, detector):
        self.detector = detector
        self.status_label.setText(
            f"準備完了（{detector.device} で実行）。写真・動画をドロップしてください。"
        )
        self.drop_area.setEnabled(True)
        self.select_btn.setEnabled(True)

    def on_model_failed(self, msg: str):
        self.status_label.setText("モデルの読み込みに失敗しました")
        QMessageBox.critical(self, "エラー", f"検出モデルの読み込みに失敗しました:\n{msg}")

    def on_inpaint_toggled(self, checked: bool):
        if not checked or self.lama_ready or self.lama_loading:
            return
        self.lama_loading = True
        self.status_label.setText(
            "AI修復モデルを準備中…（初回はモデルのダウンロードで数分かかることがあります）"
        )
        self.drop_area.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.lama_thread = LamaPreloadThread()
        self.lama_thread.loaded.connect(self.on_lama_loaded)
        self.lama_thread.failed.connect(self.on_lama_failed)
        self.lama_thread.start()

    def on_lama_loaded(self):
        self.lama_ready = True
        self.lama_loading = False
        self.status_label.setText("AI修復モデルの準備が完了しました。")
        self.drop_area.setEnabled(True)
        self.select_btn.setEnabled(True)
        self._start_worker_if_idle()

    def on_lama_failed(self, msg: str):
        self.lama_loading = False
        self.drop_area.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.status_label.setText("AI修復モデルの読み込みに失敗しました")
        QMessageBox.critical(self, "エラー", f"AI修復モデルの読み込みに失敗しました:\n{msg}")

    # -- file intake ---------------------------------------------------
    def on_select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "写真・動画を選択",
            "",
            "画像・動画 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.mp4 *.mov *.m4v *.avi *.mkv *.webm)",
        )
        if files:
            self.on_files_added(files)

    def on_files_added(self, files: list[str]):
        if self._placeholder_present():
            self.drop_area.clear()

        for f in files:
            item = QListWidgetItem(f"待機中  {Path(f).name}")
            self.drop_area.addItem(item)
            self.item_rows[f] = item

        self.pending_files.extend(files)
        self.log.append(f"{len(files)} 件のファイルを追加しました。")
        self._start_worker_if_idle()

    def _placeholder_present(self) -> bool:
        return self.drop_area.count() == 1 and self.drop_area.item(0) is self.drop_area._placeholder

    # -- processing ------------------------------------------------------
    def _start_worker_if_idle(self):
        if self.worker is not None and self.worker.isRunning():
            return
        if not self.pending_files or self.detector is None:
            return

        mode = {0: MODE_MOSAIC, 1: MODE_FILL, 2: MODE_INPAINT}[self.mode_group.checkedId()]
        if mode == MODE_INPAINT and not self.lama_ready:
            # Still downloading/loading the AI model; wait for it to finish.
            return

        files = self.pending_files
        self.pending_files = []
        self._current_batch_files = files
        self._current_mode = mode

        margin = 0.0
        mosaic_cells = self.mosaic_slider.value()
        self.detector.conf = self.conf_slider.value() / 100

        self.worker = BatchWorker(self.detector, files, mode, margin, mosaic_cells)
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_progress.connect(self.on_file_progress)
        self.worker.file_done.connect(self.on_file_done)
        self.worker.file_error.connect(self.on_file_error)
        self.worker.batch_progress.connect(self.on_batch_progress)
        self.worker.all_done.connect(self.on_all_done)
        self.progress.setValue(0)
        self.batch_label.setText(f"0/{len(files)} 件完了")
        self.cancel_btn.setEnabled(True)
        self.worker.start()

    def on_file_started(self, path: str):
        item = self.item_rows.get(path)
        if item:
            item.setText(f"処理中…  {Path(path).name}")
        self.status_label.setText(f"処理中: {Path(path).name}")
        self.progress.setValue(0)

    def on_file_progress(self, path: str, pct: int):
        self.progress.setValue(pct)

    def on_file_done(self, path: str, out_path: str):
        item = self.item_rows.get(path)
        mode_label = self._mode_labels.get(getattr(self, "_current_mode", None), "")
        if item:
            item.setText(f"✅ 完了 [{mode_label}]  {Path(path).name}  →  {Path(out_path).name}")
            item.setForeground(Qt.GlobalColor.darkGreen)
            item.setData(Qt.ItemDataRole.UserRole, out_path)
        self.log.append(f"完了: {Path(path).name} → {out_path}")
        self.progress.setValue(100)

    def on_file_error(self, path: str, msg: str):
        item = self.item_rows.get(path)
        if item:
            item.setText(f"⚠️ エラー  {Path(path).name}")
            item.setForeground(Qt.GlobalColor.darkRed)
        self.log.append(f"エラー: {Path(path).name}: {msg}")

    def on_batch_progress(self, finished: int, total: int):
        self.batch_label.setText(f"{finished}/{total} 件完了")

    def on_cancel_clicked(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_requested = True
            self.cancel_btn.setEnabled(False)
            self.status_label.setText("キャンセル中…現在のファイルの処理が終わり次第停止します。")

    def on_all_done(self, was_cancelled: bool):
        self.cancel_btn.setEnabled(False)
        if was_cancelled or self.cancel_requested:
            self.cancel_requested = False
            # Files in this batch that never got to start stay marked "待機中";
            # relabel them so it's clear they were skipped, not silently lost.
            for f in self._current_batch_files:
                item = self.item_rows.get(f)
                if item and item.text().startswith("待機中"):
                    item.setText(f"⏹ キャンセル済み  {Path(f).name}")
                    item.setForeground(Qt.GlobalColor.gray)
            self.pending_files = []
            self.batch_label.setText("")
            self.status_label.setText("キャンセルしました。写真・動画をドロップしてください。")
            return

        self.status_label.setText("すべての処理が完了しました。写真・動画をドロップしてください。")
        if self.pending_files:
            self._start_worker_if_idle()

    def on_clear_finished(self):
        for path in list(self.item_rows.keys()):
            item = self.item_rows[path]
            if item.text().startswith("✅") or item.text().startswith("⚠️") or item.text().startswith("⏹"):
                row = self.drop_area.row(item)
                self.drop_area.takeItem(row)
                del self.item_rows[path]
        if self.drop_area.count() == 0:
            self.drop_area._show_placeholder()

    def on_open_downloads(self):
        subprocess.run(["open", str(DOWNLOADS_DIR)])

    def reveal_in_finder(self, out_path: str):
        subprocess.run(["open", "-R", out_path])


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
