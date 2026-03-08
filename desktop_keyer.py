import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
from mss import mss

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QColor, QGuiApplication, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


@dataclass
class CaptureConfig:
    source_monitor_index: int
    target_monitor_index: int
    mode: str  # "luma" or "chroma"
    luma_threshold: int
    chroma_threshold: int
    softness: int
    chroma_r: int
    chroma_g: int
    chroma_b: int
    fps: int
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    enabled: bool = True


class OverlayWindow(QWidget):
    def __init__(self, monitor_rect: dict):
        super().__init__(None)
        self.current_pixmap = None
        self.overlay_enabled = True
        self.ftb_enabled = False
        self.ftb_opacity = 255

        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self.setGeometry(
            monitor_rect["left"],
            monitor_rect["top"],
            monitor_rect["width"],
            monitor_rect["height"],
        )
        self.showFullScreen()
        self.show()

    def set_monitor_rect(self, monitor_rect: dict):
        self.setGeometry(
            monitor_rect["left"],
            monitor_rect["top"],
            monitor_rect["width"],
            monitor_rect["height"],
        )
        self.show()

    def set_overlay_enabled(self, enabled: bool):
        self.overlay_enabled = enabled
        self.update()

    def set_ftb_enabled(self, enabled: bool):
        self.ftb_enabled = enabled
        self.update()

    def set_ftb_opacity(self, opacity: int):
        self.ftb_opacity = max(0, min(255, opacity))
        self.update()

    def set_frame(self, image: QImage):
        self.current_pixmap = QPixmap.fromImage(image)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        if self.ftb_enabled:
            color = QColor(0, 0, 0, self.ftb_opacity)
            painter.fillRect(self.rect(), color)
            return

        if self.overlay_enabled and self.current_pixmap is not None:
            painter.drawPixmap(self.rect(), self.current_pixmap)

        painter.end()


class CaptureWorker(QThread):
    frame_ready = Signal(QImage)
    status_message = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self.config = None

    def update_config(self, config: CaptureConfig):
        self.config = config

    def stop(self):
        self._running = False
        self.wait(1500)

    def _make_alpha_luma(self, bgr: np.ndarray, threshold: int, softness: int) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        ramp = max(1, softness)
        alpha = ((gray - threshold) * 255 / ramp).clip(0, 255).astype(np.uint8)
        return alpha

    def _make_alpha_chroma(
        self,
        bgr: np.ndarray,
        key_bgr: tuple[int, int, int],
        threshold: int,
        softness: int,
    ) -> np.ndarray:
        frame = bgr.astype(np.int16)
        key = np.array(key_bgr, dtype=np.int16).reshape((1, 1, 3))
        dist = np.sqrt(np.sum((frame - key) ** 2, axis=2))
        ramp = max(1, softness)
        alpha = ((dist - threshold) * 255 / ramp).clip(0, 255).astype(np.uint8)
        return alpha

    def run(self):
        self._running = True

        with mss() as sct:
            while self._running:
                if self.config is None or not self.config.enabled:
                    time.sleep(0.03)
                    continue

                try:
                    monitors = sct.monitors
                    if self.config.source_monitor_index >= len(monitors):
                        self.status_message.emit("Invalid source monitor selection.")
                        time.sleep(0.25)
                        continue

                    src = monitors[self.config.source_monitor_index]

                    capture_rect = {
                        "left": src["left"] + self.config.crop_x,
                        "top": src["top"] + self.config.crop_y,
                        "width": max(1, self.config.crop_w),
                        "height": max(1, self.config.crop_h),
                    }

                    raw = np.array(sct.grab(capture_rect), dtype=np.uint8)
                    bgr = raw[:, :, :3]

                    if self.config.mode == "luma":
                        alpha = self._make_alpha_luma(
                            bgr=bgr,
                            threshold=self.config.luma_threshold,
                            softness=self.config.softness,
                        )
                    else:
                        alpha = self._make_alpha_chroma(
                            bgr=bgr,
                            key_bgr=(
                                self.config.chroma_b,
                                self.config.chroma_g,
                                self.config.chroma_r,
                            ),
                            threshold=self.config.chroma_threshold,
                            softness=self.config.softness,
                        )

                    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
                    rgba[:, :, 3] = alpha

                    h, w, _ = rgba.shape
                    image = QImage(
                        rgba.data,
                        w,
                        h,
                        4 * w,
                        QImage.Format.Format_RGBA8888,
                    ).copy()

                    self.frame_ready.emit(image)

                    target_fps = max(1, self.config.fps)
                    time.sleep(1.0 / target_fps)

                except Exception as exc:
                    self.status_message.emit(f"Capture error: {exc}")
                    time.sleep(0.25)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Desktop Keyer")
        self.resize(760, 560)

        self.overlay_window = None
        self.worker = CaptureWorker()
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.status_message.connect(self.set_status)

        self.monitor_data = []
        self._build_ui()
        self.refresh_monitors()
        self._setup_shortcuts()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        source_group = QGroupBox("Routing")
        source_form = QFormLayout(source_group)

        self.source_monitor_combo = QComboBox()
        self.target_monitor_combo = QComboBox()

        self.refresh_button = QPushButton("Refresh monitors")
        self.refresh_button.clicked.connect(self.refresh_monitors)

        self.use_source_bounds_button = QPushButton("Use full source monitor")
        self.use_source_bounds_button.clicked.connect(self.apply_source_monitor_bounds)

        source_form.addRow("Source monitor", self.source_monitor_combo)
        source_form.addRow("Target monitor", self.target_monitor_combo)
        source_form.addRow(self.refresh_button, self.use_source_bounds_button)

        key_group = QGroupBox("Keying")
        key_form = QFormLayout(key_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["luma", "chroma"])

        self.luma_threshold_spin = QSpinBox()
        self.luma_threshold_spin.setRange(0, 255)
        self.luma_threshold_spin.setValue(20)

        self.chroma_threshold_spin = QSpinBox()
        self.chroma_threshold_spin.setRange(0, 441)
        self.chroma_threshold_spin.setValue(40)

        self.softness_spin = QSpinBox()
        self.softness_spin.setRange(1, 255)
        self.softness_spin.setValue(30)

        self.chroma_color_button = QPushButton("#00FF00")
        self.chroma_color_button.clicked.connect(self.pick_chroma_color)
        self.chroma_color = QColor(0, 255, 0)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)

        key_form.addRow("Mode", self.mode_combo)
        key_form.addRow("Luma threshold", self.luma_threshold_spin)
        key_form.addRow("Chroma threshold", self.chroma_threshold_spin)
        key_form.addRow("Softness", self.softness_spin)
        key_form.addRow("Chroma color", self.chroma_color_button)
        key_form.addRow("FPS", self.fps_spin)

        crop_group = QGroupBox("Capture Crop (relative to source monitor)")
        crop_layout = QGridLayout(crop_group)

        self.crop_x_spin = QSpinBox()
        self.crop_y_spin = QSpinBox()
        self.crop_w_spin = QSpinBox()
        self.crop_h_spin = QSpinBox()

        for spin in (self.crop_x_spin, self.crop_y_spin, self.crop_w_spin, self.crop_h_spin):
            spin.setRange(0, 10000)

        crop_layout.addWidget(QLabel("X"), 0, 0)
        crop_layout.addWidget(self.crop_x_spin, 0, 1)
        crop_layout.addWidget(QLabel("Y"), 0, 2)
        crop_layout.addWidget(self.crop_y_spin, 0, 3)
        crop_layout.addWidget(QLabel("W"), 1, 0)
        crop_layout.addWidget(self.crop_w_spin, 1, 1)
        crop_layout.addWidget(QLabel("H"), 1, 2)
        crop_layout.addWidget(self.crop_h_spin, 1, 3)

        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout(control_group)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_capture)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_capture)

        self.overlay_toggle_button = QPushButton("Overlay ON")
        self.overlay_toggle_button.clicked.connect(self.toggle_overlay)

        self.ftb_toggle_button = QPushButton("FTB OFF")
        self.ftb_toggle_button.clicked.connect(self.toggle_ftb)

        self.always_on_checkbox = QCheckBox("Start with overlay enabled")
        self.always_on_checkbox.setChecked(True)

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.overlay_toggle_button)
        control_layout.addWidget(self.ftb_toggle_button)
        control_layout.addWidget(self.always_on_checkbox)

        self.status_label = QLabel("Ready.")

        layout.addWidget(source_group)
        layout.addWidget(key_group)
        layout.addWidget(crop_group)
        layout.addWidget(control_group)
        layout.addWidget(self.status_label)

    def _setup_shortcuts(self):
        toggle_overlay_shortcut = QShortcut(QKeySequence("F9"), self)
        toggle_overlay_shortcut.activated.connect(self.toggle_overlay)

        toggle_ftb_shortcut = QShortcut(QKeySequence("F10"), self)
        toggle_ftb_shortcut.activated.connect(self.toggle_ftb)

    def set_status(self, text: str):
        self.status_label.setText(text)

    def refresh_monitors(self):
        self.monitor_data.clear()
        self.source_monitor_combo.clear()
        self.target_monitor_combo.clear()

        with mss() as sct:
            monitors = sct.monitors[1:]

        for idx, mon in enumerate(monitors, start=1):
            label = (
                f"{idx}: {mon['width']}x{mon['height']} "
                f"@ ({mon['left']}, {mon['top']})"
            )
            self.monitor_data.append(mon)
            self.source_monitor_combo.addItem(label, idx)
            self.target_monitor_combo.addItem(label, idx)

        if self.monitor_data:
            self.source_monitor_combo.setCurrentIndex(0)
            self.target_monitor_combo.setCurrentIndex(min(1, len(self.monitor_data) - 1))
            self.apply_source_monitor_bounds()

        self.set_status("Monitors refreshed.")

    def apply_source_monitor_bounds(self):
        source_index = self.source_monitor_combo.currentData()
        if source_index is None:
            return

        mon = self.monitor_data[source_index - 1]
        self.crop_x_spin.setValue(0)
        self.crop_y_spin.setValue(0)
        self.crop_w_spin.setValue(mon["width"])
        self.crop_h_spin.setValue(mon["height"])

    def pick_chroma_color(self):
        color = QColorDialog.getColor(self.chroma_color, self, "Choose chroma key color")
        if color.isValid():
            self.chroma_color = color
            self.chroma_color_button.setText(color.name().upper())
            self.chroma_color_button.setStyleSheet(
                f"background-color: {color.name()}; color: black;"
            )

    def build_config(self) -> CaptureConfig:
        return CaptureConfig(
            source_monitor_index=self.source_monitor_combo.currentData(),
            target_monitor_index=self.target_monitor_combo.currentData(),
            mode=self.mode_combo.currentText(),
            luma_threshold=self.luma_threshold_spin.value(),
            chroma_threshold=self.chroma_threshold_spin.value(),
            softness=self.softness_spin.value(),
            chroma_r=self.chroma_color.red(),
            chroma_g=self.chroma_color.green(),
            chroma_b=self.chroma_color.blue(),
            fps=self.fps_spin.value(),
            crop_x=self.crop_x_spin.value(),
            crop_y=self.crop_y_spin.value(),
            crop_w=self.crop_w_spin.value(),
            crop_h=self.crop_h_spin.value(),
            enabled=True,
        )

    def ensure_overlay_window(self, target_monitor_index: int):
        mon = self.monitor_data[target_monitor_index - 1]
        if self.overlay_window is None:
            self.overlay_window = OverlayWindow(mon)
        else:
            self.overlay_window.set_monitor_rect(mon)

        self.overlay_window.set_overlay_enabled(self.always_on_checkbox.isChecked())

    def start_capture(self):
        if not self.monitor_data:
            self.set_status("No monitors found.")
            return

        config = self.build_config()
        self.ensure_overlay_window(config.target_monitor_index)

        self.overlay_window.set_overlay_enabled(self.always_on_checkbox.isChecked())

        self.worker.update_config(config)
        if not self.worker.isRunning():
            self.worker.start()

        self.set_status("Capture started.")

    def stop_capture(self):
        if self.worker.isRunning():
            self.worker.stop()

        if self.overlay_window is not None:
            self.overlay_window.set_overlay_enabled(False)
            self.overlay_toggle_button.setText("Overlay OFF")

        self.set_status("Capture stopped.")

    def toggle_overlay(self):
        if self.overlay_window is None:
            self.start_capture()

        if self.overlay_window is None:
            return

        new_state = not self.overlay_window.overlay_enabled
        self.overlay_window.set_overlay_enabled(new_state)
        self.overlay_toggle_button.setText("Overlay ON" if new_state else "Overlay OFF")
        self.set_status(f"Overlay {'enabled' if new_state else 'disabled'}.")

    def toggle_ftb(self):
        if self.overlay_window is None:
            self.start_capture()

        if self.overlay_window is None:
            return

        new_state = not self.overlay_window.ftb_enabled
        self.overlay_window.set_ftb_enabled(new_state)
        self.ftb_toggle_button.setText("FTB ON" if new_state else "FTB OFF")
        self.set_status(f"Fade to black {'enabled' if new_state else 'disabled'}.")

    @Slot(QImage)
    def on_frame_ready(self, image: QImage):
        if self.overlay_window is not None:
            self.overlay_window.set_frame(image)

    def closeEvent(self, event):
        try:
            self.stop_capture()
        finally:
            super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()

    # Keep a visible app window for control while the overlay window sits on top of the target display.
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()