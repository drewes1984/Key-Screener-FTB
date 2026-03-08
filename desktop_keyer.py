import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from mss import mss
from PySide6.QtCore import QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPixmap, QShortcut
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
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


SOURCE_MONITOR = "monitor"
SOURCE_NDI = "ndi"
SCENES_FILE = Path(__file__).with_name("scenes.json")


@dataclass
class CaptureConfig:
    source_type: str
    source_monitor_index: int
    ndi_source_name: str
    target_monitor_index: int
    mode: str
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
    overlay_x: int
    overlay_y: int
    overlay_w: int
    overlay_h: int
    overlay_opacity: int
    keep_aspect: bool
    flip_h: bool
    flip_v: bool
    enable_output_ndi: bool
    output_ndi_name: str
    enabled: bool = True


class NDIManager:
    def __init__(self):
        self.available = False
        self.message = (
            "NDI is unavailable in this Python runtime. Use the screen source "
            "option, or run the app with the Python 3.13 build that has cyndilib."
        )
        self.finder = None
        self.receiver = None
        self.receiver_source_name = ""
        self.recv_frame = None
        self.sender = None
        self.sender_name = ""
        self._cyndi = {}
        self._detect_runtime()

    def _detect_runtime(self):
        if importlib.util.find_spec("cyndilib") is None:
            return

        try:
            from fractions import Fraction

            from cyndilib.finder import Finder
            from cyndilib.receiver import ReceiveFrameType, Receiver
            from cyndilib.sender import Sender
            from cyndilib.video_frame import VideoRecvFrame
            from cyndilib.wrapper.ndi_recv import RecvBandwidth, RecvColorFormat
            from cyndilib.wrapper.ndi_structs import FourCC

            self._cyndi = {
                "Fraction": Fraction,
                "Finder": Finder,
                "ReceiveFrameType": ReceiveFrameType,
                "Receiver": Receiver,
                "Sender": Sender,
                "VideoRecvFrame": VideoRecvFrame,
                "RecvBandwidth": RecvBandwidth,
                "RecvColorFormat": RecvColorFormat,
                "FourCC": FourCC,
            }
            self.finder = Finder()
            self.finder.open()
            self.finder.update_sources()
            self.available = True
            self.message = (
                "NDI ready. Refresh sources to discover network senders, or switch "
                "back to a monitor source at any time."
            )
        except Exception as exc:
            self.available = False
            self.message = (
                f"NDI runtime found but could not start: {exc}. You can still use "
                "a real monitor source."
            )

    def shutdown(self):
        try:
            if self.receiver is not None:
                self.receiver.disconnect()
        except Exception:
            pass
        try:
            if self.sender is not None:
                self.sender.close()
        except Exception:
            pass
        try:
            if self.finder is not None:
                self.finder.close()
        except Exception:
            pass

    def list_sources(self):
        if not self.available or self.finder is None:
            return []

        try:
            self.finder.update_sources()
            return list(self.finder.get_source_names())
        except Exception as exc:
            self.message = (
                f"NDI source discovery failed: {exc}. You can still use a monitor source."
            )
            return []

    def _get_source(self, source_name: str):
        if not source_name:
            return None
        self.finder.update_sources()
        source = self.finder.get_source(source_name)
        if source is not None and getattr(source, "valid", True):
            return source
        for item in self.finder.iter_sources():
            if item.name == source_name:
                return item
        return None

    def _ensure_receiver(self, source_name: str):
        if not self.available:
            return False, self.message
        if not source_name:
            return False, "Choose an NDI source first, or switch to monitor source."

        source = self._get_source(source_name)
        if source is None:
            return False, f"NDI source '{source_name}' was not found on the network."

        if self.receiver is not None and self.receiver_source_name == source_name:
            return True, "NDI receiver connected."

        try:
            if self.receiver is not None:
                self.receiver.disconnect()
        except Exception:
            pass

        try:
            Receiver = self._cyndi["Receiver"]
            VideoRecvFrame = self._cyndi["VideoRecvFrame"]
            RecvColorFormat = self._cyndi["RecvColorFormat"]
            RecvBandwidth = self._cyndi["RecvBandwidth"]

            self.receiver = Receiver(
                color_format=RecvColorFormat.BGRX_BGRA,
                bandwidth=RecvBandwidth.highest,
                allow_video_fields=False,
                recv_name="Screen Keyer Receiver",
            )
            self.recv_frame = VideoRecvFrame()
            self.receiver.set_video_frame(self.recv_frame)
            self.receiver.set_source(source)
            self.receiver.connect_to(source)
            self.receiver_source_name = source_name
            return True, f"Connected to NDI source '{source_name}'."
        except Exception as exc:
            self.receiver = None
            self.recv_frame = None
            self.receiver_source_name = ""
            return False, f"Could not connect to NDI source '{source_name}': {exc}"

    def receive_frame(self, source_name: str):
        ok, message = self._ensure_receiver(source_name)
        if not ok:
            return None, message

        try:
            recv_type = self._cyndi["ReceiveFrameType"].recv_video
            result = self.receiver.receive(recv_type, 0)
            if int(result) & int(recv_type) == 0:
                return None, f"Waiting for video frames from '{source_name}'."

            width, height = self.recv_frame.get_resolution()
            if width <= 0 or height <= 0:
                return None, f"NDI source '{source_name}' has no video yet."

            data = np.empty(self.recv_frame.get_data_size(), dtype=np.uint8)
            self.recv_frame.fill_p_data(data)
            fourcc = self.recv_frame.get_fourcc()
            frame = data.reshape((height, width, 4))
            fourcc_name = getattr(fourcc, "name", str(fourcc))
            if fourcc_name in {"BGRA", "BGRX"}:
                bgr = frame[:, :, :3].copy()
            elif fourcc_name in {"RGBA", "RGBX"}:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                return None, f"Unsupported NDI pixel format: {fourcc_name}."

            return bgr, f"Receiving NDI source '{source_name}'."
        except Exception as exc:
            return None, f"NDI receive failed for '{source_name}': {exc}"

    def _ensure_sender(self, stream_name: str, width: int, height: int):
        if not self.available:
            return False, self.message

        try:
            if self.sender is None or self.sender_name != stream_name:
                if self.sender is not None:
                    self.sender.close()
                Sender = self._cyndi["Sender"]
                self.sender = Sender(stream_name)
                self.sender.open()
                self.sender_name = stream_name

            video_frame = self.sender.video_frame
            video_frame.set_fourcc(self._cyndi["FourCC"].BGRA)
            video_frame.set_resolution(width, height)
            video_frame.set_frame_rate(self._cyndi["Fraction"](30000, 1001))
            return True, f"Sending NDI output '{stream_name}'."
        except Exception as exc:
            self.sender = None
            self.sender_name = ""
            return False, f"NDI output setup failed for '{stream_name}': {exc}"

    def send_frame(self, rgba_frame: np.ndarray, stream_name: str):
        height, width, _ = rgba_frame.shape
        ok, message = self._ensure_sender(stream_name, width, height)
        if not ok:
            return False, message

        try:
            bgra_frame = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGRA)
            self.sender.write_video(bgra_frame.reshape(-1))
            return True, f"Sending NDI output '{stream_name}'."
        except Exception as exc:
            return False, f"NDI send failed for '{stream_name}': {exc}"


class OverlayWindow(QWidget):
    def __init__(self, monitor_rect: dict):
        super().__init__(None)
        self.current_pixmap = None
        self.overlay_enabled = True
        self.overlay_rect = None
        self.keep_aspect = True
        self.overlay_opacity = 255
        self.overlay_black_enabled = False
        self.screen_black_enabled = False
        self.overlay_black_opacity = 0
        self.overlay_black_target = 0
        self.screen_black_opacity = 0
        self.screen_black_target = 0
        self.transition_ms = 300
        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(16)
        self._animation_timer.timeout.connect(self._advance_animation)

        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self.set_monitor_rect(monitor_rect)
        self.showFullScreen()
        self.show()

    def set_monitor_rect(self, monitor_rect: dict):
        self.setGeometry(
            monitor_rect["left"],
            monitor_rect["top"],
            monitor_rect["width"],
            monitor_rect["height"],
        )
        if self.overlay_rect is None:
            self.overlay_rect = self.rect()
        self.show()
        self.update()

    def set_overlay_enabled(self, enabled: bool):
        self.overlay_enabled = enabled
        self.update()

    def set_overlay_geometry(self, x: int, y: int, w: int, h: int, keep_aspect: bool):
        self.overlay_rect = self.rect().adjusted(0, 0, 0, 0)
        self.overlay_rect.setX(max(0, x))
        self.overlay_rect.setY(max(0, y))
        self.overlay_rect.setWidth(max(1, min(w, self.width())))
        self.overlay_rect.setHeight(max(1, min(h, self.height())))
        self.keep_aspect = keep_aspect
        self.update()

    def set_overlay_opacity(self, opacity: int):
        self.overlay_opacity = max(0, min(255, opacity))
        self.update()

    def set_transition_duration(self, duration_ms: int):
        self.transition_ms = max(0, duration_ms)

    def set_overlay_black_enabled(self, enabled: bool):
        self.overlay_black_enabled = enabled
        self.overlay_black_target = 255 if enabled else 0
        self._start_animation_if_needed()

    def set_screen_black_enabled(self, enabled: bool):
        self.screen_black_enabled = enabled
        self.screen_black_target = 255 if enabled else 0
        self._start_animation_if_needed()

    def _start_animation_if_needed(self):
        if self.transition_ms <= 0:
            self.overlay_black_opacity = self.overlay_black_target
            self.screen_black_opacity = self.screen_black_target
            self.update()
            return

        if not self._animation_timer.isActive():
            self._animation_timer.start()

    def _advance_channel(self, current: int, target: int) -> int:
        if current == target:
            return current

        step_count = max(1, self.transition_ms // self._animation_timer.interval())
        step = max(1, int(np.ceil(255 / step_count)))
        if current < target:
            return min(target, current + step)
        return max(target, current - step)

    def _advance_animation(self):
        self.overlay_black_opacity = self._advance_channel(
            self.overlay_black_opacity,
            self.overlay_black_target,
        )
        self.screen_black_opacity = self._advance_channel(
            self.screen_black_opacity,
            self.screen_black_target,
        )

        if (
            self.overlay_black_opacity == self.overlay_black_target
            and self.screen_black_opacity == self.screen_black_target
        ):
            self._animation_timer.stop()

        self.update()

    def set_frame(self, image: QImage):
        self.current_pixmap = QPixmap.fromImage(image)
        self.update()

    def _fitted_target_rect(self):
        if self.overlay_rect is None:
            return self.rect()

        target = self.overlay_rect.intersected(self.rect())
        if not self.keep_aspect or self.current_pixmap is None or self.current_pixmap.isNull():
            return target

        pix_size = self.current_pixmap.size()
        pix_w = max(1, pix_size.width())
        pix_h = max(1, pix_size.height())
        scale = min(target.width() / pix_w, target.height() / pix_h)
        draw_w = max(1, int(pix_w * scale))
        draw_h = max(1, int(pix_h * scale))
        draw_x = target.x() + (target.width() - draw_w) // 2
        draw_y = target.y() + (target.height() - draw_h) // 2
        return target.adjusted(draw_x - target.x(), draw_y - target.y(), draw_w - target.width(), draw_h - target.height())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        target_rect = self._fitted_target_rect()
        if self.overlay_enabled and self.current_pixmap is not None:
            painter.setOpacity(self.overlay_opacity / 255.0)
            painter.drawPixmap(target_rect, self.current_pixmap)
            painter.setOpacity(1.0)

        if self.overlay_black_opacity > 0:
            painter.fillRect(target_rect, QColor(0, 0, 0, self.overlay_black_opacity))

        if self.screen_black_opacity > 0:
            painter.fillRect(self.rect(), QColor(0, 0, 0, self.screen_black_opacity))

        painter.end()


class CaptureWorker(QThread):
    frame_ready = Signal(QImage)
    status_message = Signal(str)

    def __init__(self, ndi_manager: NDIManager):
        super().__init__()
        self._running = False
        self._last_status = ""
        self.config = None
        self.ndi_manager = ndi_manager

    def update_config(self, config: CaptureConfig):
        self.config = config

    def stop(self):
        self._running = False
        self.wait(1500)

    def _emit_status_once(self, message: str):
        if message and message != self._last_status:
            self._last_status = message
            self.status_message.emit(message)

    def _make_alpha_luma(self, bgr: np.ndarray, threshold: int, softness: int) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        ramp = max(1, softness)
        return ((gray - threshold) * 255 / ramp).clip(0, 255).astype(np.uint8)

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
        return ((dist - threshold) * 255 / ramp).clip(0, 255).astype(np.uint8)

    def _capture_monitor_frame(self, sct, config: CaptureConfig):
        monitors = sct.monitors
        if config.source_monitor_index >= len(monitors):
            self._emit_status_once("Invalid source monitor selection.")
            return None

        src = monitors[config.source_monitor_index]
        capture_rect = {
            "left": src["left"] + config.crop_x,
            "top": src["top"] + config.crop_y,
            "width": max(1, config.crop_w),
            "height": max(1, config.crop_h),
        }
        raw = np.array(sct.grab(capture_rect), dtype=np.uint8)
        return raw[:, :, :3]

    def _capture_ndi_frame(self, config: CaptureConfig):
        frame, message = self.ndi_manager.receive_frame(config.ndi_source_name)
        self._emit_status_once(message)
        return frame

    def _prepare_rgba(self, bgr: np.ndarray, config: CaptureConfig) -> np.ndarray:
        if config.flip_h:
            bgr = cv2.flip(bgr, 1)
        if config.flip_v:
            bgr = cv2.flip(bgr, 0)

        if config.mode == "luma":
            alpha = self._make_alpha_luma(bgr, config.luma_threshold, config.softness)
        else:
            alpha = self._make_alpha_chroma(
                bgr=bgr,
                key_bgr=(config.chroma_b, config.chroma_g, config.chroma_r),
                threshold=config.chroma_threshold,
                softness=config.softness,
            )

        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = alpha
        return rgba

    def run(self):
        self._running = True
        with mss() as sct:
            while self._running:
                config = self.config
                if config is None or not config.enabled:
                    time.sleep(0.03)
                    continue

                try:
                    if config.source_type == SOURCE_NDI:
                        bgr = self._capture_ndi_frame(config)
                    else:
                        bgr = self._capture_monitor_frame(sct, config)

                    if bgr is None:
                        time.sleep(0.1)
                        continue

                    rgba = self._prepare_rgba(bgr, config)
                    if config.enable_output_ndi:
                        sent, message = self.ndi_manager.send_frame(rgba, config.output_ndi_name)
                        if not sent:
                            self._emit_status_once(message)

                    h, w, _ = rgba.shape
                    image = QImage(
                        rgba.data,
                        w,
                        h,
                        4 * w,
                        QImage.Format.Format_RGBA8888,
                    ).copy()
                    self.frame_ready.emit(image)

                    self._emit_status_once("Live.")
                    time.sleep(1.0 / max(1, config.fps))
                except Exception as exc:
                    self._emit_status_once(f"Capture error: {exc}")
                    time.sleep(0.25)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen Keyer")
        self.resize(980, 760)

        self.monitor_data = []
        self.scenes = {}
        self.overlay_window = None
        self.chroma_color = QColor(0, 255, 0)
        self.ndi_manager = NDIManager()
        self.worker = CaptureWorker(self.ndi_manager)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.status_message.connect(self.set_status)

        self._build_ui()
        self.load_scenes_from_disk()
        self.refresh_sources()
        self._setup_shortcuts()
        self._update_source_mode_ui()
        self._update_ndi_capability_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        scene_group = QGroupBox("Scene")
        scene_layout = QHBoxLayout(scene_group)
        self.scene_name_edit = QLineEdit("Main")
        self.scene_combo = QComboBox()
        self.save_scene_button = QPushButton("Save scene")
        self.load_scene_button = QPushButton("Load scene")
        self.delete_scene_button = QPushButton("Delete scene")
        self.save_scene_button.clicked.connect(self.save_scene)
        self.load_scene_button.clicked.connect(self.load_selected_scene)
        self.delete_scene_button.clicked.connect(self.delete_selected_scene)
        scene_layout.addWidget(QLabel("Name"))
        scene_layout.addWidget(self.scene_name_edit)
        scene_layout.addWidget(QLabel("Saved"))
        scene_layout.addWidget(self.scene_combo)
        scene_layout.addWidget(self.save_scene_button)
        scene_layout.addWidget(self.load_scene_button)
        scene_layout.addWidget(self.delete_scene_button)

        source_group = QGroupBox("Source")
        source_form = QFormLayout(source_group)
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems([SOURCE_MONITOR, SOURCE_NDI])
        self.source_type_combo.currentIndexChanged.connect(self._update_source_mode_ui)
        self.source_monitor_combo = QComboBox()
        self.ndi_source_combo = QComboBox()
        self.refresh_sources_button = QPushButton("Refresh sources")
        self.refresh_sources_button.clicked.connect(self.refresh_sources)
        self.ndi_status_label = QLabel("")
        self.ndi_status_label.setWordWrap(True)
        source_form.addRow("Source type", self.source_type_combo)
        source_form.addRow("Monitor source", self.source_monitor_combo)
        source_form.addRow("NDI source", self.ndi_source_combo)
        source_form.addRow(self.refresh_sources_button)
        source_form.addRow("NDI status", self.ndi_status_label)

        target_group = QGroupBox("Target Screen")
        target_form = QFormLayout(target_group)
        self.target_monitor_combo = QComboBox()
        self.fill_target_button = QPushButton("Fill target screen")
        self.fill_target_button.clicked.connect(self.apply_target_bounds)
        self.overlay_x_spin = QSpinBox()
        self.overlay_y_spin = QSpinBox()
        self.overlay_w_spin = QSpinBox()
        self.overlay_h_spin = QSpinBox()
        for spin in (self.overlay_x_spin, self.overlay_y_spin, self.overlay_w_spin, self.overlay_h_spin):
            spin.setRange(0, 10000)
        self.keep_aspect_checkbox = QCheckBox("Keep aspect")
        self.keep_aspect_checkbox.setChecked(True)
        self.overlay_opacity_spin = QSpinBox()
        self.overlay_opacity_spin.setRange(0, 255)
        self.overlay_opacity_spin.setValue(255)
        self.flip_h_checkbox = QCheckBox("Flip horizontal")
        self.flip_v_checkbox = QCheckBox("Flip vertical")
        target_form.addRow("Target monitor", self.target_monitor_combo)
        target_form.addRow(self.fill_target_button)
        target_form.addRow("Overlay X", self.overlay_x_spin)
        target_form.addRow("Overlay Y", self.overlay_y_spin)
        target_form.addRow("Overlay W", self.overlay_w_spin)
        target_form.addRow("Overlay H", self.overlay_h_spin)
        target_form.addRow("Overlay opacity", self.overlay_opacity_spin)
        target_form.addRow(self.keep_aspect_checkbox)
        target_form.addRow(self.flip_h_checkbox)
        target_form.addRow(self.flip_v_checkbox)

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
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)
        key_form.addRow("Mode", self.mode_combo)
        key_form.addRow("Luma threshold", self.luma_threshold_spin)
        key_form.addRow("Chroma threshold", self.chroma_threshold_spin)
        key_form.addRow("Softness", self.softness_spin)
        key_form.addRow("Chroma color", self.chroma_color_button)
        key_form.addRow("FPS", self.fps_spin)

        crop_group = QGroupBox("Capture Crop")
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

        output_group = QGroupBox("Output")
        output_form = QFormLayout(output_group)
        self.enable_output_ndi_checkbox = QCheckBox("Send processed output as NDI")
        self.output_ndi_name_edit = QLineEdit("Screen Keyer Out")
        self.transition_spin = QSpinBox()
        self.transition_spin.setRange(0, 5000)
        self.transition_spin.setValue(300)
        self.transition_spin.setSuffix(" ms")
        output_form.addRow(self.enable_output_ndi_checkbox)
        output_form.addRow("NDI stream name", self.output_ndi_name_edit)
        output_form.addRow("Dissolve time", self.transition_spin)

        controls_group = QGroupBox("Live Controls")
        controls_layout = QHBoxLayout(controls_group)
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.overlay_toggle_button = QPushButton("Overlay ON")
        self.overlay_black_button = QPushButton("Overlay Black OFF")
        self.screen_black_button = QPushButton("Screen Black OFF")
        self.start_enabled_checkbox = QCheckBox("Start with overlay enabled")
        self.start_enabled_checkbox.setChecked(True)
        self.start_button.clicked.connect(self.start_capture)
        self.stop_button.clicked.connect(self.stop_capture)
        self.overlay_toggle_button.clicked.connect(self.toggle_overlay)
        self.overlay_black_button.clicked.connect(self.toggle_overlay_black)
        self.screen_black_button.clicked.connect(self.toggle_screen_black)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.overlay_toggle_button)
        controls_layout.addWidget(self.overlay_black_button)
        controls_layout.addWidget(self.screen_black_button)
        controls_layout.addWidget(self.start_enabled_checkbox)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)

        top_grid = QGridLayout()
        top_grid.addWidget(scene_group, 0, 0, 1, 2)
        top_grid.addWidget(source_group, 1, 0)
        top_grid.addWidget(target_group, 1, 1)
        top_grid.addWidget(key_group, 2, 0)
        top_grid.addWidget(output_group, 2, 1)

        layout.addLayout(top_grid)
        layout.addWidget(crop_group)
        layout.addWidget(controls_group)
        layout.addWidget(self.status_label)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("F9"), self).activated.connect(self.toggle_overlay)
        QShortcut(QKeySequence("F10"), self).activated.connect(self.toggle_overlay_black)
        QShortcut(QKeySequence("F11"), self).activated.connect(self.toggle_screen_black)

    def _update_source_mode_ui(self):
        source_type = self.source_type_combo.currentText()
        is_monitor = source_type == SOURCE_MONITOR
        self.source_monitor_combo.setEnabled(is_monitor)
        self.ndi_source_combo.setEnabled(not is_monitor and self.ndi_source_combo.count() > 0)

    def _update_ndi_capability_ui(self):
        self.ndi_status_label.setText(self.ndi_manager.message)
        self.enable_output_ndi_checkbox.setEnabled(self.ndi_manager.available)
        if not self.ndi_manager.available:
            self.enable_output_ndi_checkbox.setChecked(False)

    def set_status(self, text: str):
        self.status_label.setText(text)

    def refresh_sources(self):
        self.refresh_monitors()
        self.refresh_ndi_sources()
        self._update_ndi_capability_ui()
        self._update_source_mode_ui()

    def refresh_monitors(self):
        self.monitor_data.clear()
        self.source_monitor_combo.clear()
        self.target_monitor_combo.clear()

        with mss() as sct:
            monitors = sct.monitors[1:]

        for idx, mon in enumerate(monitors, start=1):
            label = f"{idx}: {mon['width']}x{mon['height']} @ ({mon['left']}, {mon['top']})"
            self.monitor_data.append(mon)
            self.source_monitor_combo.addItem(label, idx)
            self.target_monitor_combo.addItem(label, idx)

        if self.monitor_data:
            self.source_monitor_combo.setCurrentIndex(0)
            self.target_monitor_combo.setCurrentIndex(min(1, len(self.monitor_data) - 1))
            self.apply_source_monitor_bounds()
            self.apply_target_bounds()

    def refresh_ndi_sources(self):
        self.ndi_source_combo.clear()
        sources = self.ndi_manager.list_sources()
        if not sources:
            self.ndi_source_combo.addItem("No NDI sources available")
            return

        for source_name in sources:
            self.ndi_source_combo.addItem(source_name)

    def apply_source_monitor_bounds(self):
        source_index = self.source_monitor_combo.currentData()
        if source_index is None or source_index - 1 >= len(self.monitor_data):
            return

        mon = self.monitor_data[source_index - 1]
        self.crop_x_spin.setValue(0)
        self.crop_y_spin.setValue(0)
        self.crop_w_spin.setValue(mon["width"])
        self.crop_h_spin.setValue(mon["height"])

    def apply_target_bounds(self):
        target_index = self.target_monitor_combo.currentData()
        if target_index is None or target_index - 1 >= len(self.monitor_data):
            return

        mon = self.monitor_data[target_index - 1]
        self.overlay_x_spin.setValue(0)
        self.overlay_y_spin.setValue(0)
        self.overlay_w_spin.setValue(mon["width"])
        self.overlay_h_spin.setValue(mon["height"])

    def pick_chroma_color(self):
        color = QColorDialog.getColor(self.chroma_color, self, "Choose chroma key color")
        if color.isValid():
            self.chroma_color = color
            self.chroma_color_button.setText(color.name().upper())
            self.chroma_color_button.setStyleSheet(
                f"background-color: {color.name()}; color: black;"
            )

    def build_config(self) -> CaptureConfig:
        ndi_name = self.ndi_source_combo.currentText()
        if ndi_name == "No NDI sources available":
            ndi_name = ""

        return CaptureConfig(
            source_type=self.source_type_combo.currentText(),
            source_monitor_index=self.source_monitor_combo.currentData() or 1,
            ndi_source_name=ndi_name,
            target_monitor_index=self.target_monitor_combo.currentData() or 1,
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
            overlay_x=self.overlay_x_spin.value(),
            overlay_y=self.overlay_y_spin.value(),
            overlay_w=self.overlay_w_spin.value(),
            overlay_h=self.overlay_h_spin.value(),
            overlay_opacity=self.overlay_opacity_spin.value(),
            keep_aspect=self.keep_aspect_checkbox.isChecked(),
            flip_h=self.flip_h_checkbox.isChecked(),
            flip_v=self.flip_v_checkbox.isChecked(),
            enable_output_ndi=self.enable_output_ndi_checkbox.isChecked(),
            output_ndi_name=self.output_ndi_name_edit.text().strip() or "Screen Keyer Out",
            enabled=True,
        )

    def ensure_overlay_window(self, target_monitor_index: int):
        if not self.monitor_data or target_monitor_index - 1 >= len(self.monitor_data):
            return

        mon = self.monitor_data[target_monitor_index - 1]
        if self.overlay_window is None:
            self.overlay_window = OverlayWindow(mon)
        else:
            self.overlay_window.set_monitor_rect(mon)

        self.apply_overlay_settings_to_window()

    def apply_overlay_settings_to_window(self):
        if self.overlay_window is None:
            return

        self.overlay_window.set_overlay_enabled(self.start_enabled_checkbox.isChecked())
        self.overlay_window.set_overlay_geometry(
            self.overlay_x_spin.value(),
            self.overlay_y_spin.value(),
            self.overlay_w_spin.value(),
            self.overlay_h_spin.value(),
            self.keep_aspect_checkbox.isChecked(),
        )
        self.overlay_window.set_overlay_opacity(self.overlay_opacity_spin.value())
        self.overlay_window.set_transition_duration(self.transition_spin.value())

    def start_capture(self):
        if not self.monitor_data:
            self.set_status("No monitors found.")
            return

        config = self.build_config()
        if config.source_type == SOURCE_NDI and not self.ndi_manager.available:
            self.set_status(self.ndi_manager.message)
            return

        self.ensure_overlay_window(config.target_monitor_index)
        self.worker.update_config(config)
        if not self.worker.isRunning():
            self.worker.start()

        self.set_status(
            "Capture started. F9 toggles overlay, F10 blacks the overlay, F11 blacks the full target screen."
        )

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

    def toggle_overlay_black(self):
        if self.overlay_window is None:
            self.start_capture()
        if self.overlay_window is None:
            return

        new_state = not self.overlay_window.overlay_black_enabled
        self.overlay_window.set_transition_duration(self.transition_spin.value())
        self.overlay_window.set_overlay_black_enabled(new_state)
        self.overlay_black_button.setText(
            "Overlay Black ON" if new_state else "Overlay Black OFF"
        )
        self.set_status(
            f"Overlay black {'enabled' if new_state else 'disabled'} with {self.transition_spin.value()} ms dissolve."
        )

    def toggle_screen_black(self):
        if self.overlay_window is None:
            self.start_capture()
        if self.overlay_window is None:
            return

        new_state = not self.overlay_window.screen_black_enabled
        self.overlay_window.set_transition_duration(self.transition_spin.value())
        self.overlay_window.set_screen_black_enabled(new_state)
        self.screen_black_button.setText(
            "Screen Black ON" if new_state else "Screen Black OFF"
        )
        self.set_status(
            f"Target screen black {'enabled' if new_state else 'disabled'} with {self.transition_spin.value()} ms dissolve."
        )

    def current_scene_payload(self):
        payload = asdict(self.build_config())
        payload.update(
            {
                "scene_name": self.scene_name_edit.text().strip() or "Main",
                "start_enabled": self.start_enabled_checkbox.isChecked(),
            }
        )
        return payload

    def save_scene(self):
        name = self.scene_name_edit.text().strip() or "Main"
        self.scenes[name] = self.current_scene_payload()
        self.write_scenes_to_disk()
        self.refresh_scene_combo(name)
        self.set_status(f"Saved scene '{name}'.")

    def load_selected_scene(self):
        name = self.scene_combo.currentText().strip()
        if not name:
            return
        self.apply_scene(name)

    def delete_selected_scene(self):
        name = self.scene_combo.currentText().strip()
        if not name or name not in self.scenes:
            return
        del self.scenes[name]
        self.write_scenes_to_disk()
        self.refresh_scene_combo()
        self.set_status(f"Deleted scene '{name}'.")

    def apply_scene(self, name: str):
        data = self.scenes.get(name)
        if not data:
            QMessageBox.warning(self, "Scene missing", f"Scene '{name}' was not found.")
            return

        self.scene_name_edit.setText(name)
        self.source_type_combo.setCurrentText(data.get("source_type", SOURCE_MONITOR))
        self._set_combo_data(self.source_monitor_combo, data.get("source_monitor_index", 1))
        self._set_combo_text(self.ndi_source_combo, data.get("ndi_source_name", ""))
        self._set_combo_data(self.target_monitor_combo, data.get("target_monitor_index", 1))
        self._set_combo_text(self.mode_combo, data.get("mode", "luma"))
        self.luma_threshold_spin.setValue(data.get("luma_threshold", 20))
        self.chroma_threshold_spin.setValue(data.get("chroma_threshold", 40))
        self.softness_spin.setValue(data.get("softness", 30))
        self.chroma_color = QColor(
            data.get("chroma_r", 0),
            data.get("chroma_g", 255),
            data.get("chroma_b", 0),
        )
        self.chroma_color_button.setText(self.chroma_color.name().upper())
        self.chroma_color_button.setStyleSheet(
            f"background-color: {self.chroma_color.name()}; color: black;"
        )
        self.fps_spin.setValue(data.get("fps", 30))
        self.crop_x_spin.setValue(data.get("crop_x", 0))
        self.crop_y_spin.setValue(data.get("crop_y", 0))
        self.crop_w_spin.setValue(data.get("crop_w", self.crop_w_spin.value()))
        self.crop_h_spin.setValue(data.get("crop_h", self.crop_h_spin.value()))
        self.overlay_x_spin.setValue(data.get("overlay_x", 0))
        self.overlay_y_spin.setValue(data.get("overlay_y", 0))
        self.overlay_w_spin.setValue(data.get("overlay_w", self.overlay_w_spin.value()))
        self.overlay_h_spin.setValue(data.get("overlay_h", self.overlay_h_spin.value()))
        self.overlay_opacity_spin.setValue(data.get("overlay_opacity", 255))
        self.keep_aspect_checkbox.setChecked(data.get("keep_aspect", True))
        self.flip_h_checkbox.setChecked(data.get("flip_h", False))
        self.flip_v_checkbox.setChecked(data.get("flip_v", False))
        self.enable_output_ndi_checkbox.setChecked(data.get("enable_output_ndi", False))
        self.output_ndi_name_edit.setText(data.get("output_ndi_name", "Screen Keyer Out"))
        self.start_enabled_checkbox.setChecked(data.get("start_enabled", True))
        self._update_source_mode_ui()
        self.apply_overlay_settings_to_window()
        self.set_status(f"Loaded scene '{name}'.")

    def _set_combo_data(self, combo: QComboBox, value):
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return

    def _set_combo_text(self, combo: QComboBox, value: str):
        for index in range(combo.count()):
            if combo.itemText(index) == value:
                combo.setCurrentIndex(index)
                return

    def load_scenes_from_disk(self):
        if not SCENES_FILE.exists():
            self.refresh_scene_combo()
            return

        try:
            self.scenes = json.loads(SCENES_FILE.read_text(encoding="utf-8"))
        except Exception:
            self.scenes = {}
        self.refresh_scene_combo()

    def write_scenes_to_disk(self):
        SCENES_FILE.write_text(json.dumps(self.scenes, indent=2), encoding="utf-8")

    def refresh_scene_combo(self, select_name: str = ""):
        self.scene_combo.clear()
        for name in sorted(self.scenes):
            self.scene_combo.addItem(name)
        if select_name:
            self._set_combo_text(self.scene_combo, select_name)

    @Slot(QImage)
    def on_frame_ready(self, image: QImage):
        if self.overlay_window is not None:
            self.apply_overlay_settings_to_window()
            self.overlay_window.set_frame(image)

    def closeEvent(self, event):
        try:
            self.stop_capture()
            self.ndi_manager.shutdown()
        finally:
            super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
