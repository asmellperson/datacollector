r"""
Desktop data collection app for camera frames, ROI crops, and detector crops.

Run:
    python src\data_collector_app.py

Package later with PyInstaller:
    powershell -ExecutionPolicy Bypass -File scripts\build.ps1
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SOURCE_DIR.parent
EXECUTABLE_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else PROJECT_DIR
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", PROJECT_DIR))


def first_existing_path(*paths: Path) -> Path:
    return next((path for path in paths if path.exists()), paths[0])


DEFAULT_CAMERAS_FILE = first_existing_path(
    EXECUTABLE_DIR / "config" / "cameras.json",
    BUNDLE_DIR / "config" / "cameras.json",
    PROJECT_DIR / "config" / "cameras.json",
)
DEFAULT_MODEL = str(
    first_existing_path(
        EXECUTABLE_DIR / "models" / "yolov8n.pt",
        BUNDLE_DIR / "models" / "yolov8n.pt",
        PROJECT_DIR / "models" / "yolov8n.pt",
    )
)
DEFAULT_OUTPUT = str(EXECUTABLE_DIR / "output")


MODE_RAW_INTERVAL = "固定间隔保存原始帧"
MODE_DETECT_RAW = "检测目标后保存原始帧"
MODE_ROI_INTERVAL = "固定ROI裁剪保存"
MODE_DETECT_CROP = "检测目标框裁剪保存"


@dataclass
class CameraConfig:
    name: str
    source: str


@dataclass
class DetectResult:
    boxes_xyxy: np.ndarray
    confs: np.ndarray
    class_ids: np.ndarray


@dataclass
class LabelClass:
    cls_id: int
    name: str


@dataclass
class AnnBox:
    cls_id: int
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class CollectorSettings:
    mode: str
    output_dir: Path
    model_path: str
    class_id: int
    conf: float
    save_interval: float
    infer_interval: float
    expand_ratio: float
    rtsp_tcp: bool
    roi_xyxy: Optional[tuple[int, int, int, int]]


def safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    return value.strip("._-") or "item"


def now_stem() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def safe_imwrite(path: Path, image: np.ndarray, jpeg_quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    buf.tofile(str(path))


def clip_xyxy(box: tuple[int, int, int, int], width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expand_box(
    box: np.ndarray,
    width: int,
    height: int,
    expand_ratio: float,
) -> Optional[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = x2 - x1
    bh = y2 - y1
    pad_w = bw * expand_ratio
    pad_h = bh * expand_ratio
    return clip_xyxy((int(x1 - pad_w), int(y1 - pad_h), int(x2 + pad_w), int(y2 + pad_h)), width, height)


def camera_config_candidates() -> list[Path]:
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.extend(
            [
                exe_dir / "config" / "cameras.json",
                exe_dir / "cameras.json",
                BUNDLE_DIR / "config" / "cameras.json",
                BUNDLE_DIR / "dataset" / "cameras.json",
            ]
        )
    candidates.extend(
        [
            Path.cwd() / "config" / "cameras.json",
            Path.cwd() / "cameras.json",
            Path.cwd() / "dataset" / "cameras.json",
            DEFAULT_CAMERAS_FILE,
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def load_cameras(path: Optional[Path] = None) -> list[CameraConfig]:
    config_path = path
    if config_path is None:
        config_path = next((candidate for candidate in camera_config_candidates() if candidate.exists()), None)
    if config_path is None or not config_path.exists():
        return []
    with open(config_path, "r", encoding="utf-8") as fh:
        data: Any = json.load(fh)
    cameras: list[CameraConfig] = []
    for idx, item in enumerate(data):
        if isinstance(item, str):
            if "=" in item:
                name, source = item.split("=", 1)
            else:
                name, source = f"cam{idx + 1}", item
        elif isinstance(item, dict):
            name = str(item.get("name") or f"cam{idx + 1}")
            source = str(item.get("source") or "")
        else:
            continue
        source = source.strip()
        if source:
            cameras.append(CameraConfig(safe_name(name), source))
    return cameras


def writable_cameras_file() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "config" / "cameras.json"
    existing = next((candidate for candidate in camera_config_candidates() if candidate.exists()), None)
    return existing or DEFAULT_CAMERAS_FILE


def save_cameras(cameras: list[CameraConfig], path: Optional[Path] = None) -> Path:
    config_path = path or writable_cameras_file()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"name": camera.name, "source": camera.source} for camera in cameras]
    config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config_path


def read_image(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def parse_label_classes(text: str) -> list[LabelClass]:
    classes: list[LabelClass] = []
    seen: set[int] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\s,=:]+", line, maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            cls_id = int(parts[0])
        except ValueError:
            continue
        name = parts[1].strip()
        if name and cls_id not in seen:
            classes.append(LabelClass(cls_id, name))
            seen.add(cls_id)
    classes.sort(key=lambda item: item.cls_id)
    return classes


def classes_to_text(classes: list[LabelClass]) -> str:
    return "\n".join(f"{item.cls_id} {item.name}" for item in sorted(classes, key=lambda x: x.cls_id))


def default_label_classes() -> list[LabelClass]:
    return [LabelClass(0, "人体"), LabelClass(1, "手机")]


class CameraReader:
    def __init__(self) -> None:
        self.source = ""
        self.name = ""
        self.rtsp_tcp = True
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_id = 0
        self.error = ""
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self, camera: CameraConfig, rtsp_tcp: bool = True) -> None:
        self.stop()
        self.source = camera.source
        self.name = camera.name
        self.rtsp_tcp = rtsp_tcp
        self.latest_frame = None
        self.latest_id = 0
        self.error = ""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name=f"reader-{camera.name}", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.thread = None

    def snapshot(self) -> tuple[int, str, Optional[np.ndarray]]:
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            return self.latest_id, self.error, frame

    def _run(self) -> None:
        while not self.stop_event.is_set():
            if self.source.lower().startswith("rtsp://") and self.rtsp_tcp:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self._set_error("打开摄像头失败")
                cap.release()
                time.sleep(2)
                continue
            self._set_error("")

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self._set_error("读取帧失败，正在重连")
                    break
                with self.lock:
                    self.latest_frame = frame
                    self.latest_id += 1
                    self.error = ""
            cap.release()
            time.sleep(1)

    def _set_error(self, error: str) -> None:
        with self.lock:
            self.error = error


class CollectorTask:
    def __init__(self, camera: CameraConfig, settings: CollectorSettings, log_queue: queue.Queue[str]) -> None:
        self.camera = camera
        self.settings = settings
        self.log_queue = log_queue
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.saved_count = 0
        self.last_save_ts = 0.0
        self.last_infer_ts = 0.0
        self.last_detect: Optional[DetectResult] = None
        self.model = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name=f"collector-{self.camera.name}", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

    def _run(self) -> None:
        self._log(f"开始采集：{self.camera.name}")
        while not self.stop_event.is_set():
            if self.camera.source.lower().startswith("rtsp://") and self.settings.rtsp_tcp:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(self.camera.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self._log(f"{self.camera.name} 打开失败，2秒后重试")
                cap.release()
                time.sleep(2)
                continue

            frame_id = 0
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self._log(f"{self.camera.name} 读取失败，正在重连")
                    break
                frame_id += 1
                self._collect_frame(frame_id, frame)
            cap.release()
            time.sleep(1)
        self._log(f"停止采集：{self.camera.name}，保存 {self.saved_count} 张")

    def _collect_frame(self, frame_id: int, frame: np.ndarray) -> None:
        now = time.time()
        if now - self.last_save_ts < self.settings.save_interval:
            return

        mode = self.settings.mode
        if mode == MODE_RAW_INTERVAL:
            self._save_image_and_meta("raw_frames", frame, {"mode": mode, "frame_id": frame_id})
            self.last_save_ts = now
            return

        if mode == MODE_ROI_INTERVAL:
            roi = self.settings.roi_xyxy
            if roi is None:
                return
            x1, y1, x2, y2 = roi
            clipped = clip_xyxy((x1, y1, x2, y2), frame.shape[1], frame.shape[0])
            if clipped is None:
                return
            x1, y1, x2, y2 = clipped
            crop = frame[y1:y2, x1:x2]
            self._save_image_and_meta("roi_crops", crop, {"mode": mode, "frame_id": frame_id, "roi_xyxy": list(clipped)})
            self.last_save_ts = now
            return

        result = self._detect_if_needed(frame_id, frame)
        if result is None or len(result.boxes_xyxy) == 0:
            return

        if mode == MODE_DETECT_RAW:
            best = int(np.argmax(result.confs))
            self._save_image_and_meta(
                "detected_raw_frames",
                frame,
                {
                    "mode": mode,
                    "frame_id": frame_id,
                    "class_id": int(result.class_ids[best]),
                    "conf": float(result.confs[best]),
                    "box_xyxy": [float(v) for v in result.boxes_xyxy[best]],
                },
            )
            self.last_save_ts = now
            return

        if mode == MODE_DETECT_CROP:
            height, width = frame.shape[:2]
            saved_any = False
            for idx, (box, conf, cls_id) in enumerate(zip(result.boxes_xyxy, result.confs, result.class_ids)):
                crop_box = expand_box(box, width, height, self.settings.expand_ratio)
                if crop_box is None:
                    continue
                x1, y1, x2, y2 = crop_box
                crop = frame[y1:y2, x1:x2]
                self._save_image_and_meta(
                    "detector_crops",
                    crop,
                    {
                        "mode": mode,
                        "frame_id": frame_id,
                        "det_index": idx,
                        "class_id": int(cls_id),
                        "conf": float(conf),
                        "box_xyxy": [float(v) for v in box],
                        "crop_xyxy": list(crop_box),
                    },
                )
                saved_any = True
            if saved_any:
                self.last_save_ts = now

    def _detect_if_needed(self, frame_id: int, frame: np.ndarray) -> Optional[DetectResult]:
        now = time.time()
        if self.last_detect is not None and now - self.last_infer_ts < self.settings.infer_interval:
            return self.last_detect
        if self.model is None:
            from ultralytics import YOLO

            self.model = YOLO(self.settings.model_path)
            self._log(f"{self.camera.name} 模型已加载")

        results = self.model.predict(
            source=frame,
            classes=[self.settings.class_id],
            conf=self.settings.conf,
            verbose=False,
        )
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            result = DetectResult(
                boxes_xyxy=np.empty((0, 4), dtype=np.float32),
                confs=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
            )
        else:
            result = DetectResult(
                boxes_xyxy=boxes.xyxy.cpu().numpy(),
                confs=boxes.conf.cpu().numpy(),
                class_ids=boxes.cls.cpu().numpy().astype(np.int32),
            )
        self.last_detect = result
        self.last_infer_ts = now
        return result

    def _save_image_and_meta(self, bucket: str, image: np.ndarray, meta: dict[str, Any]) -> None:
        if image is None or image.size == 0:
            return
        run_day = datetime.now().strftime("%Y%m%d")
        root = self.settings.output_dir / "runs" / run_day / self.camera.name
        stem = f"{safe_name(self.camera.name)}_{now_stem()}_{self.saved_count + 1:06d}"
        image_path = root / bucket / f"{stem}.jpg"
        safe_imwrite(image_path, image)

        full_meta = {
            "camera": self.camera.name,
            "bucket": bucket,
            "image_path": str(image_path),
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "image_shape": list(image.shape[:2]),
            **meta,
        }
        meta_path = root / "meta" / f"{stem}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(full_meta, fh, ensure_ascii=False, indent=2)

        self.saved_count += 1
        self._log(f"{self.camera.name} 保存：{image_path}")

    def _log(self, message: str) -> None:
        self.log_queue.put(f"{datetime.now().strftime('%H:%M:%S')}  {message}")


class CameraDialog(tk.Toplevel):
    def __init__(self, master: tk.Misc, existing_names: set[str]) -> None:
        super().__init__(master)
        self.title("添加摄像头")
        self.geometry("560x210")
        self.resizable(False, False)
        self.existing_names = existing_names
        self.result: Optional[CameraConfig] = None

        self.name_var = tk.StringVar()
        self.source_var = tk.StringVar()

        body = ttk.Frame(self, padding=16)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)

        ttk.Label(body, text="摄像头名称").grid(row=0, column=0, sticky="w", pady=8)
        ttk.Entry(body, textvariable=self.name_var).grid(row=0, column=1, sticky="ew", padx=(12, 0), pady=8)
        ttk.Label(body, text="RTSP地址").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(body, textvariable=self.source_var).grid(row=1, column=1, sticky="ew", padx=(12, 0), pady=8)

        buttons = ttk.Frame(body)
        buttons.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(18, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ttk.Button(buttons, text="确定", style="Primary.TButton", command=self.on_ok).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(buttons, text="取消", command=self.destroy).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.transient(master)
        self.grab_set()
        self.wait_visibility()
        self.focus()

    def on_ok(self) -> None:
        name = safe_name(self.name_var.get())
        source = self.source_var.get().strip()
        if not name:
            messagebox.showerror("错误", "请输入摄像头名称", parent=self)
            return
        if name in self.existing_names:
            messagebox.showerror("错误", "摄像头名称已存在", parent=self)
            return
        if not source:
            messagebox.showerror("错误", "请输入RTSP地址", parent=self)
            return
        self.result = CameraConfig(name, source)
        self.destroy()


class AnnotatorWindow(tk.Toplevel):
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.title("数据标注")
        self.geometry("1200x780")
        self.minsize(980, 640)

        self.images_dir = Path(DEFAULT_OUTPUT)
        self.labels_dir = Path(DEFAULT_OUTPUT) / "labels"
        self.image_paths: list[Path] = []
        self.index = 0
        self.image: Optional[np.ndarray] = None
        self.boxes: list[AnnBox] = []
        self.label_classes = default_label_classes()
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.draw_start: Optional[tuple[int, int]] = None
        self.temp_box_canvas: Optional[tuple[int, int, int, int]] = None

        self.images_var = tk.StringVar(value="")
        self.labels_var = tk.StringVar(value="")
        self.class_var = tk.StringVar(value=self.class_display_values()[0])
        self.status_var = tk.StringVar(value="请选择图片目录开始标注")

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.ann_canvas = tk.Canvas(left, bg="#111111", highlightthickness=0)
        self.ann_canvas.grid(row=0, column=0, sticky="nsew")
        self.ann_canvas.bind("<ButtonPress-1>", self.on_press)
        self.ann_canvas.bind("<B1-Motion>", self.on_drag)
        self.ann_canvas.bind("<ButtonRelease-1>", self.on_release)
        self.ann_canvas.bind("<Configure>", lambda _event: self.draw())

        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky="ns")
        right.columnconfigure(0, weight=1)

        row = 0
        ttk.Label(right, text="图片目录").grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Entry(right, textvariable=self.images_var, width=42).grid(row=row, column=0, sticky="ew", pady=3)
        row += 1
        ttk.Button(right, text="选择图片目录", command=self.choose_images_dir).grid(row=row, column=0, sticky="ew", pady=3)
        row += 1

        ttk.Label(right, text="标签目录").grid(row=row, column=0, sticky="w", pady=(8, 0))
        row += 1
        ttk.Entry(right, textvariable=self.labels_var, width=42).grid(row=row, column=0, sticky="ew", pady=3)
        row += 1
        ttk.Button(right, text="选择标签目录", command=self.choose_labels_dir).grid(row=row, column=0, sticky="ew", pady=3)
        row += 1

        ttk.Separator(right).grid(row=row, column=0, sticky="ew", pady=8)
        row += 1

        ttk.Label(right, text="类别配置：每行一个，格式为 id 名称").grid(row=row, column=0, sticky="w")
        row += 1
        self.classes_text = tk.Text(right, width=42, height=7)
        self.classes_text.grid(row=row, column=0, sticky="ew", pady=3)
        self.classes_text.insert("1.0", classes_to_text(self.label_classes))
        row += 1

        class_buttons = ttk.Frame(right)
        class_buttons.grid(row=row, column=0, sticky="ew", pady=3)
        class_buttons.columnconfigure(0, weight=1)
        class_buttons.columnconfigure(1, weight=1)
        ttk.Button(class_buttons, text="应用类别", command=self.apply_classes).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(class_buttons, text="保存类别文件", command=self.save_classes_file).grid(row=0, column=1, sticky="ew", padx=(3, 0))
        row += 1

        ttk.Label(right, text="当前类别").grid(row=row, column=0, sticky="w", pady=(8, 0))
        row += 1
        self.class_combo = ttk.Combobox(right, textvariable=self.class_var, values=self.class_display_values(), state="readonly")
        self.class_combo.grid(row=row, column=0, sticky="ew", pady=3)
        row += 1

        ttk.Separator(right).grid(row=row, column=0, sticky="ew", pady=8)
        row += 1

        nav = ttk.Frame(right)
        nav.grid(row=row, column=0, sticky="ew")
        nav.columnconfigure(0, weight=1)
        nav.columnconfigure(1, weight=1)
        ttk.Button(nav, text="上一张", command=self.prev_image).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(nav, text="下一张", command=self.next_image).grid(row=0, column=1, sticky="ew", padx=(3, 0))
        row += 1

        actions = ttk.Frame(right)
        actions.grid(row=row, column=0, sticky="ew", pady=5)
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        ttk.Button(actions, text="保存标签", command=self.save_label).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(actions, text="删除上一个框", command=self.delete_last).grid(row=0, column=1, sticky="ew", padx=(3, 0))
        row += 1

        ttk.Button(right, text="清空当前框", command=self.clear_boxes).grid(row=row, column=0, sticky="ew", pady=3)
        row += 1

        help_text = "鼠标拖拽：新增标注框 | 切换图片前会自动保存"
        ttk.Label(right, text=help_text).grid(row=row, column=0, sticky="w", pady=(10, 3))
        row += 1
        ttk.Label(right, textvariable=self.status_var, foreground="#0a5", wraplength=310).grid(row=row, column=0, sticky="ew")

    def class_display_values(self) -> list[str]:
        return [f"{item.cls_id}: {item.name}" for item in self.label_classes] or ["0: 目标"]

    def current_cls_id(self) -> int:
        text = self.class_var.get().split(":", 1)[0].strip()
        try:
            return int(text)
        except ValueError:
            return 0

    def apply_classes(self) -> None:
        classes = parse_label_classes(self.classes_text.get("1.0", "end"))
        if not classes:
            messagebox.showerror("类别配置错误", "请使用类似格式：0 人体")
            return
        self.label_classes = classes
        values = self.class_display_values()
        self.class_combo.configure(values=values)
        self.class_var.set(values[0])
        self.draw()

    def choose_images_dir(self) -> None:
        path = filedialog.askdirectory(title="选择图片目录")
        if not path:
            return
        self.images_dir = Path(path)
        self.images_var.set(str(self.images_dir))
        if not self.labels_var.get():
            self.labels_dir = self.images_dir.parent / "labels"
            self.labels_var.set(str(self.labels_dir))
        self.reload_images()

    def choose_labels_dir(self) -> None:
        path = filedialog.askdirectory(title="选择标签目录")
        if not path:
            return
        self.labels_dir = Path(path)
        self.labels_var.set(str(self.labels_dir))
        self.load_current_image()

    def reload_images(self) -> None:
        self.image_paths = sorted(
            path for path in self.images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.IMAGE_EXTS
        )
        self.index = 0
        if not self.image_paths:
            self.image = None
            self.boxes = []
            self.status_var.set("没有找到图片")
            self.draw()
            return
        self.load_current_image()

    def load_current_image(self) -> None:
        if not self.image_paths:
            return
        img_path = self.image_paths[self.index]
        image = read_image(img_path)
        if image is None:
            self.status_var.set(f"读取图片失败：{img_path.name}")
            return
        self.image = image
        self.boxes = self.load_label_file(img_path, image.shape[1], image.shape[0])
        self.status_var.set(f"{self.index + 1}/{len(self.image_paths)}  {img_path.name}  标注框={len(self.boxes)}")
        self.draw()

    def label_path_for(self, image_path: Path) -> Path:
        return Path(self.labels_var.get() or self.labels_dir) / f"{image_path.stem}.txt"

    def load_label_file(self, image_path: Path, width: int, height: int) -> list[AnnBox]:
        label_path = self.label_path_for(image_path)
        boxes: list[AnnBox] = []
        if not label_path.exists():
            return boxes
        for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                cx, cy, bw, bh = [float(v) for v in parts[1:5]]
            except ValueError:
                continue
            x1 = int((cx - bw / 2) * width)
            y1 = int((cy - bh / 2) * height)
            x2 = int((cx + bw / 2) * width)
            y2 = int((cy + bh / 2) * height)
            clipped = clip_xyxy((x1, y1, x2, y2), width, height)
            if clipped is not None:
                boxes.append(AnnBox(cls_id, *clipped))
        return boxes

    def save_label(self) -> None:
        if self.image is None or not self.image_paths:
            return
        img_path = self.image_paths[self.index]
        label_path = self.label_path_for(img_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = self.image.shape[:2]
        lines: list[str] = []
        for box in self.boxes:
            x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
            cx = ((x1 + x2) / 2) / width
            cy = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            lines.append(f"{box.cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.status_var.set(f"已保存：{label_path}  标注框={len(lines)}")

    def save_classes_file(self) -> None:
        labels_dir = Path(self.labels_var.get() or self.labels_dir)
        labels_dir.mkdir(parents=True, exist_ok=True)
        path = labels_dir / "classes.txt"
        path.write_text(classes_to_text(self.label_classes) + "\n", encoding="utf-8")
        self.status_var.set(f"已保存：{path}")

    def prev_image(self) -> None:
        if not self.image_paths:
            return
        self.save_label()
        self.index = max(0, self.index - 1)
        self.load_current_image()

    def next_image(self) -> None:
        if not self.image_paths:
            return
        self.save_label()
        self.index = min(len(self.image_paths) - 1, self.index + 1)
        self.load_current_image()

    def delete_last(self) -> None:
        if self.boxes:
            self.boxes.pop()
            self.draw()

    def clear_boxes(self) -> None:
        self.boxes = []
        self.draw()

    def on_press(self, event: tk.Event) -> None:
        if self.image is None:
            return
        self.draw_start = (int(event.x), int(event.y))
        self.temp_box_canvas = (int(event.x), int(event.y), int(event.x), int(event.y))

    def on_drag(self, event: tk.Event) -> None:
        if self.draw_start is None:
            return
        x1, y1 = self.draw_start
        self.temp_box_canvas = (x1, y1, int(event.x), int(event.y))
        self.draw()

    def on_release(self, event: tk.Event) -> None:
        if self.image is None or self.draw_start is None:
            return
        x1, y1 = self.draw_start
        self.temp_box_canvas = (x1, y1, int(event.x), int(event.y))
        frame_box = self.canvas_box_to_image_box(self.temp_box_canvas)
        if frame_box is not None:
            bx1, by1, bx2, by2 = frame_box
            if (bx2 - bx1) >= 4 and (by2 - by1) >= 4:
                self.boxes.append(AnnBox(self.current_cls_id(), bx1, by1, bx2, by2))
        self.draw_start = None
        self.temp_box_canvas = None
        self.draw()

    def canvas_box_to_image_box(self, canvas_box: tuple[int, int, int, int]) -> Optional[tuple[int, int, int, int]]:
        if self.image is None:
            return None
        height, width = self.image.shape[:2]
        ox, oy = self.display_offset
        scale = self.display_scale
        x1, y1, x2, y2 = canvas_box
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        ix1 = int((left - ox) / scale)
        iy1 = int((top - oy) / scale)
        ix2 = int((right - ox) / scale)
        iy2 = int((bottom - oy) / scale)
        return clip_xyxy((ix1, iy1, ix2, iy2), width, height)

    def image_box_to_canvas_box(self, box: AnnBox) -> tuple[int, int, int, int]:
        ox, oy = self.display_offset
        scale = self.display_scale
        return (
            int(ox + box.x1 * scale),
            int(oy + box.y1 * scale),
            int(ox + box.x2 * scale),
            int(oy + box.y2 * scale),
        )

    def class_name(self, cls_id: int) -> str:
        for item in self.label_classes:
            if item.cls_id == cls_id:
                return item.name
        return str(cls_id)

    def class_color(self, cls_id: int) -> str:
        name = self.class_name(cls_id).lower()
        if cls_id == 0 or name in {"人体", "人", "person"}:
            return "#1e88e5"
        if cls_id == 1 or name in {"手机", "phone", "cell_phone", "cellphone"}:
            return "#e53935"
        palette = [
            "#43a047",
            "#fb8c00",
            "#8e24aa",
            "#00acc1",
            "#fdd835",
            "#6d4c41",
            "#3949ab",
            "#c2185b",
        ]
        return palette[abs(cls_id) % len(palette)]

    def draw(self) -> None:
        self.ann_canvas.delete("all")
        if self.image is None:
            self.ann_canvas.create_text(30, 30, text="请选择图片目录", fill="#dddddd", anchor="nw")
            return

        canvas_w = max(1, self.ann_canvas.winfo_width())
        canvas_h = max(1, self.ann_canvas.winfo_height())
        height, width = self.image.shape[:2]
        scale = min(canvas_w / width, canvas_h / height)
        disp_w = max(1, int(width * scale))
        disp_h = max(1, int(height * scale))
        ox = (canvas_w - disp_w) // 2
        oy = (canvas_h - disp_h) // 2
        self.display_scale = scale
        self.display_offset = (ox, oy)

        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image=image)
        self.ann_canvas.create_image(ox, oy, image=self.photo, anchor="nw")

        for idx, box in enumerate(self.boxes, start=1):
            x1, y1, x2, y2 = self.image_box_to_canvas_box(box)
            color = self.class_color(box.cls_id)
            self.ann_canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            self.ann_canvas.create_text(
                x1 + 4,
                max(y1 - 10, 10),
                text=f"{idx} {box.cls_id}:{self.class_name(box.cls_id)}",
                fill=color,
                anchor="w",
            )

        if self.temp_box_canvas is not None:
            x1, y1, x2, y2 = self.temp_box_canvas
            self.ann_canvas.create_rectangle(x1, y1, x2, y2, outline="#ffeb3b", width=2)


class DataCollectorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("数据采集与标注")
        self.geometry("1280x820")
        self.minsize(1080, 720)

        self.cameras = load_cameras()
        self.reader = CameraReader()
        self.collecting = False
        self.collector_tasks: dict[str, CollectorTask] = {}
        self.model = None
        self.last_save_ts = 0.0
        self.last_infer_ts = 0.0
        self.last_detect: Optional[DetectResult] = None
        self.last_detect_frame_id = -1
        self.saved_count = 0
        self.current_frame_id = -1
        self.current_frame: Optional[np.ndarray] = None
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.roi_canvas: Optional[tuple[int, int, int, int]] = None
        self.roi_start: Optional[tuple[int, int]] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        self._setup_style()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(30, self.ui_loop)

    def _setup_style(self) -> None:
        self.configure(bg="#f5f7fb")
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", font=("Microsoft YaHei UI", 10), background="#f5f7fb", foreground="#172033")
        style.configure("TFrame", background="#f5f7fb")
        style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("Section.TLabelframe", background="#ffffff", bordercolor="#d9dee8", relief="solid")
        style.configure("Section.TLabelframe.Label", background="#ffffff", foreground="#101828", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("TLabel", background="#ffffff", foreground="#172033")
        style.configure("Muted.TLabel", background="#ffffff", foreground="#667085")
        style.configure("Status.TLabel", background="#ffffff", foreground="#067647")
        style.configure("TEntry", fieldbackground="#ffffff", bordercolor="#d0d5dd", lightcolor="#d0d5dd", darkcolor="#d0d5dd")
        style.configure("TCombobox", fieldbackground="#ffffff", bordercolor="#d0d5dd", arrowcolor="#475467")
        style.configure("TCheckbutton", background="#ffffff", foreground="#172033")
        style.configure("TButton", padding=(12, 7), background="#ffffff", foreground="#344054", bordercolor="#d0d5dd")
        style.map("TButton", background=[("active", "#f2f4f7")])
        style.configure("Primary.TButton", padding=(12, 8), background="#0b73d9", foreground="#ffffff", bordercolor="#0b73d9")
        style.map("Primary.TButton", background=[("active", "#075fb4"), ("pressed", "#064f99")], foreground=[("active", "#ffffff")])
        style.configure("Danger.TButton", padding=(12, 8), background="#ffffff", foreground="#344054", bordercolor="#d0d5dd")
        style.configure("Tool.TButton", padding=(8, 5), background="#ffffff", foreground="#475467", bordercolor="#ffffff")

    def _make_section(self, parent: ttk.Frame, title: str, row: int) -> ttk.LabelFrame:
        section = ttk.LabelFrame(parent, text=title, style="Section.TLabelframe", padding=(14, 12))
        section.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        section.columnconfigure(1, weight=1)
        return section

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(self, text="实时预览", style="Section.TLabelframe", padding=(10, 10))
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)
        preview_frame.rowconfigure(1, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        tool_bar = ttk.Frame(preview_frame, style="Card.TFrame")
        tool_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tool_bar.columnconfigure(0, weight=1)
        ttk.Label(tool_bar, text="摄像头画面", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(tool_bar, text="截图", style="Tool.TButton", command=lambda: None).grid(row=0, column=1, padx=2)
        ttk.Button(tool_bar, text="网格", style="Tool.TButton", command=lambda: None).grid(row=0, column=2, padx=2)
        ttk.Button(tool_bar, text="全屏", style="Tool.TButton", command=lambda: None).grid(row=0, column=3, padx=2)

        self.canvas = tk.Canvas(preview_frame, bg="#202325", highlightthickness=0, bd=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self.on_roi_press)
        self.canvas.bind("<B1-Motion>", self.on_roi_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_roi_release)

        self.preview_footer = ttk.Frame(preview_frame, style="Card.TFrame")
        self.preview_footer.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for col in range(5):
            self.preview_footer.columnconfigure(col, weight=1)
        ttk.Label(self.preview_footer, text="● 未启动", style="Muted.TLabel").grid(row=0, column=0, sticky="w", padx=8)
        ttk.Label(self.preview_footer, text="帧率: --", style="Muted.TLabel").grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(self.preview_footer, text="分辨率: --x--", style="Muted.TLabel").grid(row=0, column=2, sticky="w", padx=8)
        ttk.Label(self.preview_footer, text="帧: 0", style="Muted.TLabel").grid(row=0, column=3, sticky="w", padx=8)
        ttk.Label(self.preview_footer, text="连接状态: 未连接", style="Muted.TLabel").grid(row=0, column=4, sticky="e", padx=8)

        panel = ttk.Frame(self, padding=0)
        panel.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)
        panel.columnconfigure(0, weight=1)

        self.camera_var = tk.StringVar(value=self.cameras[0].name if self.cameras else "")
        self.mode_var = tk.StringVar(value=MODE_RAW_INTERVAL)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.output_var = tk.StringVar(value=DEFAULT_OUTPUT)
        self.class_id_var = tk.StringVar(value="67")
        self.conf_var = tk.StringVar(value="0.15")
        self.interval_var = tk.StringVar(value="2.0")
        self.infer_interval_var = tk.StringVar(value="0.2")
        self.expand_var = tk.StringVar(value="0.05")
        self.rtsp_tcp_var = tk.BooleanVar(value=True)
        self.preview_boxes_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="未启动")

        row = 0
        device = self._make_section(panel, "设备连接", row)
        row += 1
        ttk.Label(device, text="摄像头").grid(row=0, column=0, sticky="w", pady=5)
        self.camera_combo = ttk.Combobox(device, textvariable=self.camera_var, values=[c.name for c in self.cameras], width=28, state="readonly")
        self.camera_combo.grid(row=0, column=1, sticky="ew", padx=(10, 8), pady=5)
        ttk.Button(device, text="打开摄像头", style="Primary.TButton", command=self.open_camera).grid(row=0, column=2, sticky="ew", pady=5)
        ttk.Label(device, text="连接状态").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Label(device, text="● 未连接", foreground="#f04438", background="#ffffff").grid(row=1, column=1, columnspan=2, sticky="w", padx=(10, 0), pady=5)
        camera_buttons = ttk.Frame(device, style="Card.TFrame")
        camera_buttons.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        camera_buttons.columnconfigure(0, weight=1)
        camera_buttons.columnconfigure(1, weight=1)
        camera_buttons.columnconfigure(2, weight=1)
        ttk.Button(camera_buttons, text="添加摄像头", command=self.add_camera).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(camera_buttons, text="删除当前", command=self.delete_current_camera).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(camera_buttons, text="保存配置", command=self.save_camera_config).grid(row=0, column=2, sticky="ew", padx=(4, 0))

        config = self._make_section(panel, "采集配置", row)
        row += 1
        ttk.Label(config, text="采集模式").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Combobox(
            config,
            textvariable=self.mode_var,
            values=[MODE_RAW_INTERVAL, MODE_DETECT_RAW, MODE_ROI_INTERVAL, MODE_DETECT_CROP],
            width=28,
            state="readonly",
        ).grid(row=0, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=5)
        ttk.Label(config, text="模型").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(config, textvariable=self.model_var, width=32).grid(row=1, column=1, sticky="ew", padx=(10, 8), pady=5)
        ttk.Button(config, text="选择模型", command=self.pick_model).grid(row=1, column=2, sticky="ew", pady=5)
        ttk.Label(config, text="输出目录").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(config, textvariable=self.output_var, width=32).grid(row=2, column=1, sticky="ew", padx=(10, 8), pady=5)
        ttk.Button(config, text="选择输出目录", command=self.pick_output).grid(row=2, column=2, sticky="ew", pady=5)

        params = self._make_section(panel, "检测参数", row)
        row += 1

        fields = [
            ("类别ID", self.class_id_var),
            ("检测置信度", self.conf_var),
            ("保存间隔秒", self.interval_var),
            ("检测间隔秒", self.infer_interval_var),
            ("裁剪扩边比例", self.expand_var),
        ]
        for idx, (label, var) in enumerate(fields):
            grid_row = idx // 2
            grid_col = (idx % 2) * 2
            ttk.Label(params, text=label).grid(row=grid_row, column=grid_col, sticky="w", pady=5)
            ttk.Entry(params, textvariable=var, width=12).grid(row=grid_row, column=grid_col + 1, sticky="ew", padx=(10, 18), pady=5)

        options = self._make_section(panel, "选项", row)
        row += 1
        ttk.Checkbutton(options, text="RTSP使用TCP", variable=self.rtsp_tcp_var).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Checkbutton(options, text="预览检测框", variable=self.preview_boxes_var).grid(row=0, column=1, sticky="w", pady=5)
        ttk.Button(options, text="清除ROI", command=self.clear_roi).grid(row=0, column=2, sticky="ew", pady=5)
        ttk.Button(options, text="打开标注", style="Primary.TButton", command=self.open_annotator).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        multi = self._make_section(panel, "同时采集摄像头", row)
        row += 1
        self.camera_listbox = tk.Listbox(
            multi,
            selectmode="multiple",
            height=4,
            exportselection=False,
            bg="#ffffff",
            fg="#172033",
            relief="solid",
            bd=1,
            highlightthickness=0,
            font=("Microsoft YaHei UI", 10),
        )
        self.camera_listbox.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.refresh_camera_widgets()
        self.camera_listbox.selection_set(0, "end")

        control_section = self._make_section(panel, "控制", row)
        row += 1
        controls = ttk.Frame(control_section, style="Card.TFrame")
        controls.grid(row=0, column=0, columnspan=3, sticky="ew")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(controls, text="开始采集选中摄像头", style="Primary.TButton", command=self.start_collect)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(controls, text="停止", style="Danger.TButton", command=self.stop_collect).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        status = self._make_section(panel, "运行状态", row)
        status.columnconfigure(0, weight=1)
        ttk.Label(status, text="状态").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Label(status, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(status, text="日志").grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 4))
        self.log_text = tk.Text(
            status,
            width=46,
            height=6,
            state="disabled",
            bg="#ffffff",
            fg="#172033",
            relief="solid",
            bd=1,
            highlightthickness=0,
            font=("Consolas", 9),
        )
        self.log_text.grid(row=2, column=0, columnspan=3, sticky="nsew")

        self.after(100, self.draw_placeholder)

    def draw_placeholder(self) -> None:
        if self.current_frame is not None:
            return
        self.canvas.delete("all")
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        self.canvas.create_rectangle(0, 0, w, h, fill="#202325", outline="")
        for x in range(0, w, 24):
            self.canvas.create_line(x, 0, x, h, fill="#2a2e31")
        for y in range(0, h, 24):
            self.canvas.create_line(0, y, w, y, fill="#2a2e31")
        self.canvas.create_text(w // 2, h // 2 - 18, text="摄像头预览", fill="#9aa0a6", font=("Microsoft YaHei UI", 20, "bold"))
        self.canvas.create_text(w // 2, h // 2 + 18, text="等待连接...", fill="#9aa0a6", font=("Microsoft YaHei UI", 12))

    def open_camera(self) -> None:
        camera = self.selected_camera()
        if camera is None:
            messagebox.showerror("错误", "没有可用摄像头，请检查 config/cameras.json")
            return
        self.reader.start(camera, bool(self.rtsp_tcp_var.get()))
        self.status_var.set(f"已打开 {camera.name}")
        self.log(f"打开摄像头: {camera.name}")

    def refresh_camera_widgets(self) -> None:
        names = [camera.name for camera in self.cameras]
        if hasattr(self, "camera_combo"):
            self.camera_combo.configure(values=names)
        if self.camera_var.get() not in names:
            self.camera_var.set(names[0] if names else "")
        if hasattr(self, "camera_listbox"):
            selected_names = set(self.selected_collection_camera_names())
            self.camera_listbox.delete(0, "end")
            for idx, name in enumerate(names):
                self.camera_listbox.insert("end", name)
                if name in selected_names:
                    self.camera_listbox.selection_set(idx)

    def selected_collection_camera_names(self) -> list[str]:
        if not hasattr(self, "camera_listbox"):
            return [camera.name for camera in self.cameras]
        return [self.camera_listbox.get(idx) for idx in self.camera_listbox.curselection()]

    def selected_collection_cameras(self) -> list[CameraConfig]:
        names = set(self.selected_collection_camera_names())
        if not names:
            return []
        return [camera for camera in self.cameras if camera.name in names]

    def add_camera(self) -> None:
        dialog = CameraDialog(self, {camera.name for camera in self.cameras})
        self.wait_window(dialog)
        if dialog.result is None:
            return
        selected_before = set(self.selected_collection_camera_names())
        self.cameras.append(dialog.result)
        self.refresh_camera_widgets()
        for idx, camera in enumerate(self.cameras):
            if camera.name in selected_before or camera.name == dialog.result.name:
                self.camera_listbox.selection_set(idx)
        path = save_cameras(self.cameras)
        self.log(f"已添加摄像头：{dialog.result.name}")
        self.log(f"配置已保存：{path}")

    def delete_current_camera(self) -> None:
        camera = self.selected_camera()
        if camera is None:
            return
        if camera.name in self.collector_tasks:
            messagebox.showerror("错误", "该摄像头正在采集，请先停止采集")
            return
        if not messagebox.askyesno("确认删除", f"确定删除摄像头 {camera.name} 吗？"):
            return
        self.cameras = [item for item in self.cameras if item.name != camera.name]
        self.refresh_camera_widgets()
        path = save_cameras(self.cameras)
        self.log(f"已删除摄像头：{camera.name}")
        self.log(f"配置已保存：{path}")

    def save_camera_config(self) -> None:
        path = save_cameras(self.cameras)
        self.log(f"配置已保存：{path}")
        messagebox.showinfo("保存成功", f"摄像头配置已保存：\n{path}")

    def selected_camera(self) -> Optional[CameraConfig]:
        name = self.camera_var.get()
        for camera in self.cameras:
            if camera.name == name:
                return camera
        return None

    def pick_model(self) -> None:
        path = filedialog.askopenfilename(title="选择 YOLO 模型", filetypes=[("YOLO模型", "*.pt *.engine"), ("所有文件", "*.*")])
        if path:
            self.model_var.set(path)

    def pick_output(self) -> None:
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_var.set(path)

    def start_collect(self) -> None:
        cameras = self.selected_collection_cameras()
        if not cameras:
            messagebox.showerror("错误", "请至少选择一个要采集的摄像头")
            return
        settings = self.current_collector_settings()
        if settings.mode == MODE_ROI_INTERVAL and settings.roi_xyxy is None:
            messagebox.showerror("错误", "ROI裁剪模式需要先在预览画面上画ROI")
            return

        if self.current_frame is None and self.selected_camera() is not None:
            self.open_camera()

        for camera in cameras:
            if camera.name in self.collector_tasks:
                continue
            task = CollectorTask(camera, settings, self.log_queue)
            self.collector_tasks[camera.name] = task
            task.start()

        self.collecting = True
        self.start_button.configure(text="采集中...")
        self.status_var.set(f"采集中：{len(self.collector_tasks)} 路")
        self.log(f"开始多路采集：{', '.join(camera.name for camera in cameras)}")

    def stop_collect(self) -> None:
        for task in list(self.collector_tasks.values()):
            task.stop()
        total_saved = sum(task.saved_count for task in self.collector_tasks.values())
        self.collector_tasks.clear()
        self.collecting = False
        self.start_button.configure(text="开始采集选中摄像头")
        self.status_var.set(f"已停止，保存 {total_saved} 张")
        self.log("停止全部采集")

    def current_collector_settings(self) -> CollectorSettings:
        return CollectorSettings(
            mode=self.mode_var.get(),
            output_dir=Path(self.output_var.get()),
            model_path=self.model_var.get(),
            class_id=int(self.float_value(self.class_id_var, 67, 0)),
            conf=self.float_value(self.conf_var, 0.15, 0.0),
            save_interval=self.float_value(self.interval_var, 2.0, 0.0),
            infer_interval=self.float_value(self.infer_interval_var, 0.2, 0.01),
            expand_ratio=self.float_value(self.expand_var, 0.05, 0.0),
            rtsp_tcp=bool(self.rtsp_tcp_var.get()),
            roi_xyxy=self.canvas_roi_to_frame_roi(),
        )

    def clear_roi(self) -> None:
        self.roi_canvas = None
        self.log("已清除ROI")

    def open_annotator(self) -> None:
        AnnotatorWindow(self)

    def on_roi_press(self, event: tk.Event) -> None:
        self.roi_start = (int(event.x), int(event.y))
        self.roi_canvas = (int(event.x), int(event.y), int(event.x), int(event.y))

    def on_roi_drag(self, event: tk.Event) -> None:
        if self.roi_start is None:
            return
        x1, y1 = self.roi_start
        self.roi_canvas = (x1, y1, int(event.x), int(event.y))

    def on_roi_release(self, event: tk.Event) -> None:
        if self.roi_start is None:
            return
        x1, y1 = self.roi_start
        x2, y2 = int(event.x), int(event.y)
        if abs(x2 - x1) < 8 or abs(y2 - y1) < 8:
            self.roi_canvas = None
        else:
            self.roi_canvas = (x1, y1, x2, y2)
            self.log(f"ROI: {self.canvas_roi_to_frame_roi()}")
        self.roi_start = None

    def ui_loop(self) -> None:
        frame_id, error, frame = self.reader.snapshot()
        if frame is not None:
            self.current_frame_id = frame_id
            self.current_frame = frame
            self.draw_frame(frame)
        elif error:
            self.status_var.set(error)

        self.flush_logs()
        self.after(30, self.ui_loop)

    def draw_frame(self, frame: np.ndarray) -> None:
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        height, width = frame.shape[:2]
        scale = min(canvas_w / width, canvas_h / height)
        disp_w = max(1, int(width * scale))
        disp_h = max(1, int(height * scale))
        offset_x = (canvas_w - disp_w) // 2
        offset_y = (canvas_h - disp_h) // 2
        self.display_scale = scale
        self.display_offset = (offset_x, offset_y)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, image=self.photo, anchor="nw")

        if self.preview_boxes_var.get() and self.last_detect is not None and self.last_detect_frame_id == self.current_frame_id:
            for box, conf, cls_id in zip(self.last_detect.boxes_xyxy, self.last_detect.confs, self.last_detect.class_ids):
                x1, y1, x2, y2 = self.frame_box_to_canvas(box)
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="#ff4fd8", width=2)
                self.canvas.create_text(x1 + 4, max(y1 - 10, 10), text=f"{int(cls_id)} {float(conf):.2f}", fill="#ff4fd8", anchor="w")

        if self.roi_canvas is not None:
            x1, y1, x2, y2 = self.roi_canvas
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00e676", width=2)

    def frame_box_to_canvas(self, box: np.ndarray) -> tuple[int, int, int, int]:
        ox, oy = self.display_offset
        s = self.display_scale
        x1, y1, x2, y2 = [float(v) for v in box]
        return int(ox + x1 * s), int(oy + y1 * s), int(ox + x2 * s), int(oy + y2 * s)

    def canvas_roi_to_frame_roi(self) -> Optional[tuple[int, int, int, int]]:
        if self.current_frame is None or self.roi_canvas is None:
            return None
        height, width = self.current_frame.shape[:2]
        ox, oy = self.display_offset
        s = self.display_scale
        x1, y1, x2, y2 = self.roi_canvas
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        fx1 = int((left - ox) / s)
        fy1 = int((top - oy) / s)
        fx2 = int((right - ox) / s)
        fy2 = int((bottom - oy) / s)
        return clip_xyxy((fx1, fy1, fx2, fy2), width, height)

    def collect(self, frame_id: int, frame: np.ndarray) -> None:
        now = time.time()
        mode = self.mode_var.get()
        interval = self.float_value(self.interval_var, 2.0, 0.0)
        if now - self.last_save_ts < interval:
            return

        camera = self.selected_camera()
        camera_name = camera.name if camera else "camera"

        if mode == MODE_RAW_INTERVAL:
            self.save_image_and_meta(camera_name, "raw_frames", frame, {"mode": mode, "frame_id": frame_id})
            self.last_save_ts = now
            return

        if mode == MODE_ROI_INTERVAL:
            roi = self.canvas_roi_to_frame_roi()
            if roi is None:
                self.status_var.set("请先在预览画面上画ROI")
                return
            x1, y1, x2, y2 = roi
            crop = frame[y1:y2, x1:x2]
            self.save_image_and_meta(camera_name, "roi_crops", crop, {"mode": mode, "frame_id": frame_id, "roi_xyxy": list(roi)})
            self.last_save_ts = now
            return

        result = self.detect_if_needed(frame_id, frame)
        if result is None or len(result.boxes_xyxy) == 0:
            return

        if mode == MODE_DETECT_RAW:
            best = int(np.argmax(result.confs))
            self.save_image_and_meta(
                camera_name,
                "detected_raw_frames",
                frame,
                {
                    "mode": mode,
                    "frame_id": frame_id,
                    "class_id": int(result.class_ids[best]),
                    "conf": float(result.confs[best]),
                    "box_xyxy": [float(v) for v in result.boxes_xyxy[best]],
                },
            )
            self.last_save_ts = now
            return

        if mode == MODE_DETECT_CROP:
            height, width = frame.shape[:2]
            expand_ratio = self.float_value(self.expand_var, 0.05, 0.0)
            saved_any = False
            for idx, (box, conf, cls_id) in enumerate(zip(result.boxes_xyxy, result.confs, result.class_ids)):
                crop_box = expand_box(box, width, height, expand_ratio)
                if crop_box is None:
                    continue
                x1, y1, x2, y2 = crop_box
                crop = frame[y1:y2, x1:x2]
                self.save_image_and_meta(
                    camera_name,
                    "detector_crops",
                    crop,
                    {
                        "mode": mode,
                        "frame_id": frame_id,
                        "det_index": idx,
                        "class_id": int(cls_id),
                        "conf": float(conf),
                        "box_xyxy": [float(v) for v in box],
                        "crop_xyxy": list(crop_box),
                    },
                )
                saved_any = True
            if saved_any:
                self.last_save_ts = now

    def detect_if_needed(self, frame_id: int, frame: np.ndarray) -> Optional[DetectResult]:
        now = time.time()
        infer_interval = self.float_value(self.infer_interval_var, 0.2, 0.01)
        if self.last_detect is not None and self.last_detect_frame_id == frame_id:
            return self.last_detect
        if now - self.last_infer_ts < infer_interval:
            return self.last_detect

        if self.model is None:
            self.status_var.set("正在加载模型...")
            self.update_idletasks()
            from ultralytics import YOLO

            self.model = YOLO(self.model_var.get())
            self.log(f"模型已加载: {self.model_var.get()}")

        class_id = int(self.float_value(self.class_id_var, 67, 0))
        conf = self.float_value(self.conf_var, 0.15, 0.0)
        results = self.model.predict(source=frame, classes=[class_id], conf=conf, verbose=False)
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            result = DetectResult(
                boxes_xyxy=np.empty((0, 4), dtype=np.float32),
                confs=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
            )
        else:
            result = DetectResult(
                boxes_xyxy=boxes.xyxy.cpu().numpy(),
                confs=boxes.conf.cpu().numpy(),
                class_ids=boxes.cls.cpu().numpy().astype(np.int32),
            )
        self.last_detect = result
        self.last_detect_frame_id = frame_id
        self.last_infer_ts = now
        return result

    def save_image_and_meta(self, camera_name: str, bucket: str, image: np.ndarray, meta: dict[str, Any]) -> None:
        if image is None or image.size == 0:
            return
        run_day = datetime.now().strftime("%Y%m%d")
        root = Path(self.output_var.get()) / "runs" / run_day / camera_name
        stem = f"{safe_name(camera_name)}_{now_stem()}_{self.saved_count + 1:06d}"
        image_path = root / bucket / f"{stem}.jpg"
        safe_imwrite(image_path, image)

        full_meta = {
            "camera": camera_name,
            "bucket": bucket,
            "image_path": str(image_path),
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "image_shape": list(image.shape[:2]),
            **meta,
        }
        meta_path = root / "meta" / f"{stem}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(full_meta, fh, ensure_ascii=False, indent=2)

        self.saved_count += 1
        self.status_var.set(f"已保存 {self.saved_count} 张")
        self.log(f"保存: {image_path}")

    def float_value(self, var: tk.StringVar, default: float, min_value: float) -> float:
        try:
            value = float(var.get())
        except ValueError:
            return default
        return max(min_value, value)

    def log(self, message: str) -> None:
        self.log_queue.put(f"{datetime.now().strftime('%H:%M:%S')}  {message}")

    def flush_logs(self) -> None:
        changed = False
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.configure(state="normal")
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
            changed = True
        if changed:
            self.log_text.update_idletasks()

    def on_close(self) -> None:
        self.collecting = False
        for task in list(self.collector_tasks.values()):
            task.stop()
        self.collector_tasks.clear()
        self.reader.stop()
        self.destroy()


def main() -> int:
    app = DataCollectorApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
