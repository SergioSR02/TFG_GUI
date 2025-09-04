"""
GUI PySide6 – Esqueleto integrable para tu pipeline de visión
----------------------------------------------------------------
Ventana principal con:
 - Área de vídeo (QLabel) que recibe QImage desde un PipelineWorker en segundo plano.
 - Dock izquierdo (Controles):
     • Modo: Detección (YOLOv8+ByteTrack) / Seguimiento local (Multi‑KCF)
     • Focus ON/OFF
     • Segmentación ON/OFF + botón Parámetros (3 presets)
     • Selector de modelo: COCO / Militar (lee config/models.yaml)
     • Botón Pantallazo/ROI
     • Mostrar/Ocultar Info/Coordenadas/Teclas (equivalente a tus paneles h/i/o)
     • Botón Exit
 - Dock derecho (Estado):
     • FPS, backend, dispositivo, modelo actual, ruta de pesos
     • Log de eventos

Integración prevista (no intrusiva):
 - Sustituye los métodos "process_frame_*" por tus llamadas reales a
   detector_yolov8 / tracker_mot_bytetrack / tracker_sot_multikcf / roi_* / ui_overlay, etc.
 - Mantiene las rutas de pesos en config/models.yaml (con fallback a las rutas que nos diste).
 - ROISelectDialog devuelve una bbox (x, y, w, h) para inicializar tu Multi‑KCF.

Requisitos:
  pip install PySide6 opencv-python pyyaml numpy

Ejecutar:
  python gui_app.py
"""
from __future__ import annotations

import sys
import time
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml
import cv2

from PySide6.QtCore import (Qt, QThread, Signal, Slot, QSize, QRect, QPoint,
                            QObject, QEvent)
from PySide6.QtGui import (QImage, QPixmap, QAction, QPainter, QPen,
                           QCloseEvent)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QDockWidget, QPushButton, QComboBox, QCheckBox, QGroupBox, QRadioButton,
    QButtonGroup, QFormLayout, QDialog, QDialogButtonBox, QStatusBar,
    QFileDialog, QTextEdit, QSpinBox, QMessageBox
)

from PySide6.QtWidgets import QTabWidget
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QAbstractItemView
from PySide6.QtGui import QTextOption
from PySide6.QtCore import Qt

from core import detector_yolov8 as det
from core import tracker_mot_bytetrack as mot
from core import tracker_sot_multikcf as sot
from core import roi_segmentation as seg
from core.bounding_box import BBox

import supervision as sv
import csv
import datetime as dt

from collections import deque
import math


# -------------------------------------------------------------
# Configuración (lee config/models.yaml con fallback a rutas dadas)
# -------------------------------------------------------------
DEFAULT_CFG = {
    "coco_weights": "C:/Users/sergi/OneDrive - Universidad Politécnica de Madrid/Escritorio/GIA/TFG/Seguimiento/Comparacion imagenes/Sergio_4/yolov8n.pt",
    "military_weights": "C:/Users/sergi/OneDrive - Universidad Politécnica de Madrid/Escritorio/GIA/TFG/Seguimiento/Comparacion imagenes/Sergio_4/modelos/best.pt",
    "device": "cuda",  # o "cpu"
    "half": True,
    "imgsz": 640,
    "conf": 0.25,
    "iou": 0.45,
}


def load_models_yaml() -> dict:
    cfg_path = Path("config/models.yaml")
    if cfg_path.exists():
        try:
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            return {**DEFAULT_CFG, **(data or {})}
        except Exception as e:
            print(f"[config] Error leyendo YAML: {e}. Se usan defaults.")
    return dict(DEFAULT_CFG)

def _now_ns(): 
    import time
    return time.perf_counter_ns()

def _ms(dt_ns: int) -> float:
    return dt_ns / 1e6
# -------------------------------------------------------------
# Utilidades de imagen: cv2 Mat (BGR) -> QImage
# -------------------------------------------------------------
def cv_to_qimage(frame: np.ndarray) -> QImage:
    if frame is None:
        return QImage()
    if len(frame.shape) == 2:
        h, w = frame.shape
        qimg = QImage(frame.data, w, h, w, QImage.Format_Grayscale8)
        return qimg.copy()
    h, w, ch = frame.shape
    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return qimg.copy()


# -------------------------------------------------------------
# Diálogo de selección de ROI (rectángulo con ratón)
# -------------------------------------------------------------
class ROISelectDialog(QDialog):
    def __init__(self, frame_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seleccionar ROI")
        self._orig = frame_bgr.copy()
        self._pix = QPixmap.fromImage(cv_to_qimage(self._orig))
        self._label = QLabel()
        self._label.setMinimumSize(640, 360)
        self._label.setScaledContents(True)
        self._label.setMouseTracking(True)      
        self._label.setCursor(Qt.CrossCursor)  
        self._label.installEventFilter(self)
        self._update_pix()
        self._dragging = False
        self._p0: Optional[QPoint] = None
        self._rect = QRect()

        btns = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self._bb = QDialogButtonBox(btns)
        self._bb.accepted.connect(self.accept)
        self._bb.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(self._label)
        lay.addWidget(self._bb)

    def eventFilter(self, obj, ev):
        if obj is self._label:
            if ev.type() == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                self._dragging = True
                self._p0 = ev.position().toPoint()
                self._rect = QRect(self._p0, self._p0)
                self._update_pix()
                return True
            if ev.type() == QEvent.MouseMove and self._dragging:
                p1 = ev.position().toPoint()
                self._rect = QRect(self._p0, p1).normalized()
                self._update_pix()
                return True
            if ev.type() == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton:
                self._dragging = False
                return True
        return super().eventFilter(obj, ev)


    def _update_pix(self):
        # Imagen base desde el frame original
        base = QPixmap.fromImage(cv_to_qimage(self._orig))

        # Tamaño visible real del QLabel (sin bordes)
        vis = self._label.contentsRect().size()
        w = max(1, vis.width())
        h = max(1, vis.height())

        # Fallback por si aún no hay layout: usa como mínimo el tamaño del pixmap base
        if w <= 1 or h <= 1:
            w = max(w, base.width() or 640)
            h = max(h, base.height() or 360)

        disp = base.scaled(QSize(w, h), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        # Dibuja el rectángulo actual (normalizado) sobre 'disp'
        pix = disp.copy()
        if not self._rect.isNull():
            p = QPainter(pix)
            p.setPen(QPen(Qt.green, 2))
            p.drawRect(self._rect.normalized())
            p.end()

        self._label.setPixmap(pix)


    def roi_xywh(self) -> Optional[Tuple[int, int, int, int]]:
        if self._rect.isNull():
            return None

        r = self._rect.normalized()

        disp = self._label.pixmap()
        if disp is None or disp.isNull():
            return None

        # Tamaño visible (sin bordes)
        lw = self._label.contentsRect().width()
        lh = self._label.contentsRect().height()
        if lw <= 0 or lh <= 0:
            lw, lh = disp.width(), disp.height()
            if lw <= 0 or lh <= 0:
                return None

        h, w = self._orig.shape[:2]
        sx = w / float(lw)
        sy = h / float(lh)

        x  = int(round(r.x()      * sx))
        y  = int(round(r.y()      * sy))
        ww = int(round(r.width()  * sx))
        hh = int(round(r.height() * sy))

        # Clamping y tamaño mínimo útil
        x  = max(0, min(w - 1, x))
        y  = max(0, min(h - 1, y))
        ww = max(3, min(w - x, ww))
        hh = max(3, min(h - y, hh))

        return (x, y, ww, hh)


    def keyPressEvent(self, ev):
        key = ev.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            # Solo aceptar si hay ROI dibujado
            if not self._rect.isNull():
                self.accept()
            else:
                # Ignora Enter si no hay rectángulo
                ev.ignore()
                return
        elif key == Qt.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(ev)

    def showEvent(self, ev):
        super().showEvent(ev)
        # Al mostrarse, el label ya tiene tamaño real → re-render definitivo
        self._update_pix()

# -------------------------------------------------------------
# Diálogo de parámetros de segmentación (3 presets)
# -------------------------------------------------------------
class SegParamsDialog(QDialog):
    def __init__(self, current_preset: int = 1, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Segmentación asistida – Parámetros")
        self.group = QButtonGroup(self)
        self.r1 = QRadioButton("Preset 1 – FloodFill")
        self.r2 = QRadioButton("Preset 2 – FloodFill + Morfología")
        self.r3 = QRadioButton("Preset 3 – HSV")
        for i, r in enumerate((self.r1, self.r2, self.r3), start=1):
            self.group.addButton(r, i)
        if current_preset == 1:
            self.r1.setChecked(True)
        elif current_preset == 2:
            self.r2.setChecked(True)
        else:
            self.r3.setChecked(True)

        lay = QVBoxLayout(self)
        lay.addWidget(self.r1)
        lay.addWidget(self.r2)
        lay.addWidget(self.r3)
        btns = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        bb = QDialogButtonBox(btns)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def selected_preset(self) -> int:
        bid = self.group.checkedId()
        return 1 if bid <= 0 else bid


# -------------------------------------------------------------
# Trabajador de cámara (captura) y pipeline (procesamiento)
# -------------------------------------------------------------
class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray)
    opened = Signal(int, int, float)  # w, h, fps
    error = Signal(str)

    def __init__(self, source: int | str = 0):
        super().__init__()
        self._source = source
        self._running = True
        self._cap: Optional[cv2.VideoCapture] = None

    def stop(self):
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            self.error.emit("No se pudo abrir la fuente de vídeo")
            return
        self._cap = cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.opened.emit(w, h, float(fps))
        while self._running:
            ok, frame = cap.read()
            if not ok:
                break
            self.frame_ready.emit(frame)
        cap.release()

    def grab_last_frame(self) -> Optional[np.ndarray]:
        # Método simple: abrir una captura rápida (no perfecto pero ok para pantallazo)
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None


@dataclass
class PipelineSettings:
    mode: str = "det"
    focus_on: bool = False
    seg_on: bool = False
    seg_preset: int = 1
    model_name: str = "coco"
    kcf_redetect_interval: int = 0
    seg_backend: str = "basic"   # "basic"  / "sam"


class PipelineWorker(QThread):
    image_ready = Signal(QImage)
    stats_ready = Signal(dict)
    perf_ready  = Signal(dict)
    error       = Signal(str)  

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.settings = PipelineSettings()
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._running = True
        self._last_time = time.time()
        self._fps = 0.0
        self._model_paths = {
            "coco": cfg.get("coco_weights", "yolov8n.pt"),
            "militar": cfg.get("military_weights", "modelos/best.pt"),
        }
        # TODO: enganchar aquí tus objetos reales (detector/trackers)
        self._stub_color = (20, 20, 20)
        self._det_model = None
        self._mot = None
        self._loaded_model_name = None
        self._last_dets: Optional[sv.Detections] = None
        self._last_sot_boxes = []          # [(x,y,w,h), ...]
        self._last_frame_shape = None
        self._objects_rows = []        
        self._t0 = time.perf_counter()         
        self._focus_box = None             # (x,y,w,h) seleccionado
        self._sot = None
        self._pending_init_sot = None      # (frame, [xywh...])
        self._current_boxes = []          # lista de BBox actuales (detección o KCF)
        self._frame_count = 0

        self._seg_click = None           # ya lo usas para FloodFill
        self._sam = None                 # modelo FastSAM (lazy)
        self._sam_ready = False
        self._sam_points = []            # lista de (x,y, label) con label 1=pos, 0=neg
        self._sam_last_mask = None
        self._sam_last_frame_idx = -999
        self._sam_every_n = 5
        self._grabcut_mask = None        # cache para ROI estático
        self._last_roi_rect = None       # (x,y,w,h) si hay ROI
        self._last_canvas = None

        self._t_cap_ns = None           
        self._last_perf = None          
        self._prev_ids_set = set()      

    # ---- API pública ----
    def enqueue(self, frame: np.ndarray):
        if not self._running:
            return
        try:
            if self._queue.full():
                _ = self._queue.get_nowait()
            self._t_cap_ns = _now_ns()
            self._queue.put_nowait(frame)
        except queue.Full:
            pass

    def update_settings(self, s: PipelineSettings):
        self.settings = s
    
    def snapshot_frame(self):
        lf = getattr(self, "_last_frame", None)
        return None if lf is None else lf.copy()

    def snapshot_canvas(self) -> Optional[np.ndarray]:
        lc = getattr(self, "_last_canvas", None)
        return None if lc is None else lc.copy()

    def stop(self):
        self._running = False

    # ---- Procesamiento principal ----
    def run(self):
        while self._running:
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                processed = self._process_frame(frame)
            except Exception as e:
                self.error.emit(f"Error en pipeline: {e}")
                processed = frame

            if self._last_perf is not None:
                self.perf_ready.emit(self._last_perf)
                self._last_perf = None

            self._emit_image(processed)
            self._update_fps()

    def _update_fps(self):
        now = time.time()
        dt = now - self._last_time
        if dt > 0:
            self._fps = 1.0 / dt
        self._last_time = now

        H, W = (0, 0)
        if self._last_frame_shape is not None:
            H, W = self._last_frame_shape
        self.stats_ready.emit({
            "fps": round(self._fps, 1),
            "mode": self.settings.mode,
            "focus": self.settings.focus_on,
            "seg": f"{self.settings.seg_on} (p{self.settings.seg_preset})",
            "model": self.settings.model_name,
            "weights": self._model_paths.get(self.settings.model_name, ""),
            "coords_text": getattr(self, "_coords_text", ""),
            "keys_text": "",
            "resolution": f"{W}x{H}",
            "optical_center": (W // 2, H // 2),
            "n_objects": len(self._objects_rows or []),
            "objects_rows": self._objects_rows or []
        })

    def _emit_image(self, frame_bgr: np.ndarray):
        self._last_canvas = frame_bgr.copy()           # ← guarda el “pantallazo” real
        qimg = cv_to_qimage(frame_bgr)
        self.image_ready.emit(qimg)

    # ---- Aquí enganchas tu pipeline real ----
    def _process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        self._last_frame_shape = frame_bgr.shape[:2]
        self._last_frame = frame_bgr.copy()
        if self.settings.mode == "det":
            return self._process_detection(frame_bgr)
        else:
            return self._process_tracking_local(frame_bgr)

    def _process_detection(self, frame: np.ndarray) -> np.ndarray:
        canvas = frame.copy()   
        perf = {"mode": "det", "num_rois": 0, "model": self.settings.model_name}
        # (re)carga modelo si cambió
        if self._loaded_model_name != self.settings.model_name or self._det_model is None:            
            self._det_model = det.inicializar_modelo(self.settings.model_name)
            self._mot = mot.inicializar_seguimiento()
            self._loaded_model_name = self.settings.model_name

        cfg_det = det.DetectionConfig(
            imgsz=self.cfg.get("imgsz", 640),
            conf=self.cfg.get("conf", 0.25),
            iou=self.cfg.get("iou", 0.45),
            device=self.cfg.get("device", "cuda"),
            half=self.cfg.get("half", True),
        )

        # YOLO
        t_yi = _now_ns()
        dets = det.detectar_objetos(self._det_model, canvas, cfg_det)
        t_yo = _now_ns()
        self._last_dets = dets
        self._current_boxes = []

        # ByteTrack
        if self._mot is None:
            self._mot = mot.inicializar_seguimiento()
        t_mi = _now_ns()
        tracks = mot.actualizar_seguimiento(self._mot, dets)
        t_mo = _now_ns()

        # ids (para estabilidad)
        ids = []
        if tracks and getattr(tracks, "tracker_id", None) is not None:
            ids = [int(i) for i in tracks.tracker_id if i is not None]
        cur = set(ids)
        ids_new  = len(cur - self._prev_ids_set)
        ids_lost = len(self._prev_ids_set - cur)
        self._prev_ids_set = cur

        # dibujar + poblar _current_boxes
        coords_lines = []
        if tracks and getattr(tracks, "xyxy", None) is not None:
            xyxy = tracks.xyxy.astype(int)
            ids = getattr(tracks, "tracker_id", None)
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                self._current_boxes.append(BBox.from_xyxy(x1, y1, x2, y2))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 60), 2)
                label = f"ID {int(ids[i])}" if ids is not None and ids[i] is not None else "obj"
                cv2.putText(canvas, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 60), 2)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                coords_lines.append(f"{label}: ({cx},{cy})")

        rows = []
        if tracks and getattr(tracks, "xyxy", None) is not None:
            xyxy = tracks.xyxy.astype(int)
            ids  = getattr(tracks, "tracker_id", None)
            # intenta sacar class_id del resultado del MOT o, si no, del último YOLO
            cls  = getattr(tracks, "class_id", None)
            if cls is None and self._last_dets is not None:
                cls = getattr(self._last_dets, "class_id", None)

            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                tid = int(ids[i]) if ids is not None and ids[i] is not None else i
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                bbox = (x1, y1, x2 - x1, y2 - y1)
                cname = "-"
                if cls is not None and i < len(cls):
                    try: cname = self._class_name(int(cls[i]))
                    except Exception: cname = str(cls[i])
                rows.append({"id": tid, "class": cname, "bbox": bbox, "center": (cx, cy)})
        self._objects_rows = rows


        # Segmentación y foco (mide ΔSEG si está ON)
        if self.settings.seg_on:
            t_segi = _now_ns()
            canvas = self._apply_segmentation(canvas)
            t_sego = _now_ns()
            perf["tsegi"], perf["tsego"] = t_segi, t_sego
        else:
            canvas = self._apply_segmentation(canvas)

        canvas = self._apply_focus(canvas)

        # Guardar marcas en el perf dict
        perf.update({
            "tyi": t_yi, "tyo": t_yo,
            "tmi": t_mi, "tmo": t_mo,
            "ids_new": ids_new, "ids_lost": ids_lost,
            "imgsz": self.cfg.get("imgsz", 640),
            "N": getattr(self.settings, "kcf_redetect_interval", 0),
            "t_cap": self._t_cap_ns
        })
        self._last_perf = perf

        self._coords_text = "Detecciones:\n" + ("\n".join(coords_lines) if coords_lines else "—")
        return canvas


    def _process_tracking_local(self, frame: np.ndarray) -> np.ndarray:
        canvas = frame.copy()
        self._frame_count += 1

        perf = {"mode": "sot", "model": self.settings.model_name,
                "imgsz": self.cfg.get("imgsz", 640),
                "N": getattr(self.settings, "kcf_redetect_interval", 0),
                "t_cap": self._t_cap_ns}

        # init pendiente desde GUI
        if self._pending_init_sot is not None:
            init_frame, rois = self._pending_init_sot
            self._sot = sot.MultiKCFTracker(init_frame, rois)
            self._pending_init_sot = None
            self._last_sot_boxes = rois

        if self._sot is None:
            cv2.putText(canvas, "Usa 'Pantallazo / Seleccionar ROI' (pestaña General)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            self._coords_text = "SOT: sin ROI"
            self._current_boxes = []
            self._last_perf = perf   
            return canvas            
        
        # re-detección cada N frames (0 = sin redetección; valor viene de la GUI)
        interval = int(getattr(self.settings, "kcf_redetect_interval", 0))
        if interval > 0 and (self._frame_count % interval == 0) and self._det_model is not None:
            cfg_det = det.DetectionConfig(
                imgsz=self.cfg.get("imgsz", 640),
                conf=self.cfg.get("conf", 0.25),
                iou=self.cfg.get("iou", 0.45),
                device=self.cfg.get("device", "cuda"),
                half=self.cfg.get("half", True),
            )
            t_yi = _now_ns()
            dets = det.detectar_objetos(self._det_model, frame, cfg_det)
            t_yo = _now_ns()
            perf.update({"tyi": t_yi, "tyo": t_yo})
            if dets is not None and hasattr(dets, "xyxy") and len(dets) > 0:
                rois = [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in dets.xyxy.astype(int)]
                self._sot = sot.MultiKCFTracker(frame, rois)

        # actualizar KCF
        self._current_boxes = []
        t_si = _now_ns()
        boxes = self._sot.update(canvas)
        rows = []
        if boxes:
            for i, (x, y, w, h) in enumerate(boxes):
                cx, cy = x + w // 2, y + h // 2
                rows.append({"id": i + 1, "class": "-", "bbox": (x, y, w, h), "center": (cx, cy)})
        self._objects_rows = rows

        t_so = _now_ns()
        perf.update({"tsi": t_si, "tso": t_so, "num_rois": len(boxes or [])})

        # ΔSEG si aplica
        if self.settings.seg_on:
            t_segi = _now_ns()
            canvas = self._apply_segmentation(canvas)
            t_sego = _now_ns()
            perf["tsegi"], perf["tsego"] = t_segi, t_sego
        else:
            canvas = self._apply_segmentation(canvas)

        canvas = self._apply_focus(canvas)

        self._last_perf = perf

        coords_lines = []
        if boxes:
            for (x, y, w, h) in boxes:
                self._current_boxes.append(BBox.from_xyxy(x, y, w, h))
                cv2.rectangle(canvas, (x, y), (x+w, y+h), (60, 220, 60), 2)
                cx, cy = x + w // 2, y + h // 2
                coords_lines.append(f"ROI: ({cx},{cy})")

        self._coords_text = "SOT boxes:\n" + ("\n".join(coords_lines) if coords_lines else "—")

        return canvas


    def _apply_focus(self, img: np.ndarray) -> np.ndarray:
        if not self.settings.focus_on or self._focus_box is None:
            return img

        # si hay cajas actuales, re-alinea el foco a la de mayor IoU
        if self._current_boxes:
            ref = BBox(*self._focus_box) if isinstance(self._focus_box, tuple) else self._focus_box
            best = max(self._current_boxes, key=lambda b: ref.iou(b))
            self._focus_box = best.xywh if isinstance(self._focus_box, tuple) else best

        x, y, w, h = (self._focus_box.xywh if hasattr(self._focus_box, "xywh") else self._focus_box)
        H, W = img.shape[:2]
        pad = int(0.15 * max(w, h))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            return img
        return cv2.resize(crop, (W, H), interpolation=cv2.INTER_CUBIC)

    def _apply_segmentation(self, img: np.ndarray) -> np.ndarray:
        if not self.settings.seg_on:
            return img

        backend = getattr(self.settings, "seg_backend", "basic")
        mask = None

        if backend == "basic":
            # === tus presets FloodFill (1/2) y HSV (3) ===
            preset = self.settings.seg_preset
            if preset in (1, 2):
                if self._seg_click is None:
                    return img  # esperamos a clic seed
                x, y = self._seg_click
                mask = seg.segment_floodfill(img, x, y, loDiff=20, upDiff=20,
                                            kernel_size=5, use_morph=(preset == 2))
            else:  # 3 = HSV
                mask = seg.segment_color_threshold(img, kernel_size=5)

        elif backend == "sam":
            try:
                import torch
                from ultralytics import FastSAM
            except Exception as e:
                print(f"[seg] FastSAM no disponible: {e}")
                return img

            # carga perezosa una vez
            if self._sam is None:
                self._sam = FastSAM("FastSAM-s.pt")
                # mueve a CUDA si hay
                try:
                    if torch.cuda.is_available():
                        self._sam.model.to("cuda")
                        if hasattr(self._sam.model, "half"):
                            self._sam.model.half()  # fp16
                except Exception:
                    pass
                self._sam_ready = False

            # prompts: clicks o bbox
            points, labels, b = None, None, None
            if self._sam_ready and self._sam_points:
                points = [[p[0], p[1]] for p in self._sam_points]
                labels = [p[2] for p in self._sam_points]
                # limita a 10 clics para no ralentizar
                if len(points) > 10:
                    points, labels = points[-10:], labels[-10:]
            else:
                if self._focus_box is not None:
                    (x, y, w, h) = (self._focus_box.xywh if hasattr(self._focus_box, "xywh") else self._focus_box)
                    b = [x, y, x + w, y + h]
                elif self._last_dets is not None and len(self._last_dets) > 0:
                    (x1, y1, x2, y2) = self._last_dets.xyxy.astype(int)[0]
                    b = [int(x1), int(y1), int(x2), int(y2)]

            # si no hay prompt, reutiliza máscara si existe; si no, nada
            if points is None and b is None:
                if self._sam_last_mask is not None:
                    mask = self._sam_last_mask
                return img if mask is None else self._draw_mask_and_maybe_start_kcf(img, mask, self)

            # ejecuta cada N frames; si no toca, reutiliza máscara
            if self._frame_count - self._sam_last_frame_idx < self._sam_every_n and self._sam_last_mask is not None:
                mask = self._sam_last_mask
                return self._draw_mask_and_maybe_start_kcf(img, mask, self)

            # inferencia
            try:
                with torch.inference_mode():
                    results = None
                    if points is not None:
                        results = self._sam(
                            img,
                            points=points, labels=labels,
                            retina_masks=False,   # más rápido
                            imgsz=384,            # más rápido
                            conf=0.25, iou=0.7
                        )
                    elif b is not None:
                        results = self._sam(
                            img,
                            bboxes=b,
                            retina_masks=False,
                            imgsz=384,
                            conf=0.25, iou=0.7
                        )
                if results is not None:
                    res = results[0]
                    if hasattr(res, "masks") and res.masks is not None and len(res.masks.data) > 0:
                        m = res.masks.data[0].detach().cpu().numpy()
                        if m.ndim == 3:  # a veces (1,H,W)
                            m = m[0]
                        m = (m > 0.5).astype(np.uint8) * 255
                        # Redimensiona la máscara al tamaño del frame
                        H, W = img.shape[:2]
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                        mask = seg.postprocess_mask(m, min_area=150, close=5, fill_holes=True)
                        self._sam_last_mask = mask
                        self._sam_last_frame_idx = self._frame_count

            except Exception as e:
                print(f"[seg] Error SAM: {e}")
                mask = self._sam_last_mask  # si falla, intenta reutilizar la última

        # dibuja contorno si hay máscara
        if mask is not None:
            mask = seg.postprocess_mask(mask, min_area=150, close=5, fill_holes=True)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 255, 255), 2)

            # (opcional) arrancar KCF desde máscara si estás en SOT y no hay tracker aún
            bbox = seg.get_bbox_from_mask(mask, getattr(self, "_seg_click", None))
            if bbox and self._sot is None and self._last_frame is not None:
                # Re-chequeo: si el usuario ya apagó la segmentación, NO cambies el modo
                if not self.settings.seg_on:
                    return img
                if self.settings.mode != "sot":
                    self.settings.mode = "sot"
                self._sot = sot.MultiKCFTracker(self._last_frame, [bbox])
            
            
            
            # bbox = seg.get_bbox_from_mask(mask, getattr(self, "_seg_click", None))
            # if bbox and self._sot is None and self._last_frame is not None:
            #     # cambia a SOT si venías de DET para que el bucle entre por _process_tracking_local()
            #     if self.settings.mode != "sot":
            #         self.settings.mode = "sot"
            #     self._sot = sot.MultiKCFTracker(self._last_frame, [bbox])

        return img


    def request_init_sot(self, frame_bgr, rois_xywh):
        self._pending_init_sot = (frame_bgr.copy(), list(rois_xywh))

    def request_kcf_from_detection(self):
        if self._last_dets is None or len(self._last_dets) == 0 or self._last_frame is None:
            return
        xyxy = self._last_dets.xyxy.astype(int)
        pick = None
        if self._focus_box is None and self._seg_click:
            cx, cy = self._seg_click
            for (x1,y1,x2,y2) in xyxy:
                if x1<=cx<=x2 and y1<=cy<=y2:
                    pick = (x1, y1, x2-x1, y2-y1); break
        if pick is None:
            areas = [(x2-x1)*(y2-y1) for (x1,y1,x2,y2) in xyxy]
            i = int(np.argmax(areas))
            x1,y1,x2,y2 = xyxy[i]; pick = (x1, y1, x2-x1, y2-y1)
        self.request_init_sot(self._last_frame, [pick])

    def request_focus_at(self, pt_xy):
        if self._focus_box is not None:
            x, y, w, h = (self._focus_box.xywh if hasattr(self._focus_box, "xywh") else self._focus_box)
            if x <= pt_xy[0] <= x + w and y <= pt_xy[1] <= y + h:
                self._focus_box = None
                self._seg_click = pt_xy  
                return

        self._seg_click = pt_xy

        for b in self._current_boxes:
            if b.contains(*pt_xy):
                self._focus_box = b.xywh
                return

        if self._last_dets is not None and len(self._last_dets) > 0:
            for (x1, y1, x2, y2) in self._last_dets.xyxy.astype(int):
                if x1 <= pt_xy[0] <= x2 and y1 <= pt_xy[1] <= y2:
                    self._focus_box = (x1, y1, x2 - x1, y2 - y1)
                    return

        self._focus_box = None



    def clear_focus(self):
        self._focus_box = None

    def request_seg_click(self, pt_xy):
        self._seg_click = (int(pt_xy[0]), int(pt_xy[1]))

    def exit_sot(self):
        self._sot = None
        self._last_sot_boxes = []
        self._focus_box = None

    def reset_sam(self):
        self._sam_points = []
        self._sam_ready = False
        self._sam_last_mask = None
        self._sam_last_frame_idx = -999
        self._seg_click = None  

    def add_sam_click(self, pt: Tuple[int,int], positive: bool = True):
        self._sam_points.append((int(pt[0]), int(pt[1]), 1 if positive else 0))
        self._sam_ready = True

    def set_roi_rect(self, rect_xywh: Tuple[int,int,int,int]):
        self._last_roi_rect = rect_xywh
        self._grabcut_mask = None   

    @staticmethod
    def _draw_mask_and_maybe_start_kcf(img, mask, worker):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 255, 255), 2)

        bbox = seg.get_bbox_from_mask(mask, getattr(worker, "_seg_click", None))
        if bbox and worker._sot is None and worker._last_frame is not None:
            
            if not worker.settings.seg_on:
                return img
            if worker.settings.mode != "sot":
                worker.settings.mode = "sot"
            worker._sot = sot.MultiKCFTracker(worker._last_frame, [bbox])        
        return img

    def _class_name(self, cid) -> str:
        try:
            m = getattr(self._det_model, "model", self._det_model)
            names = getattr(m, "names", None)
            if isinstance(names, dict):
                return names.get(int(cid), str(cid))
            if isinstance(names, (list, tuple)) and 0 <= int(cid) < len(names):
                return names[int(cid)]
        except Exception:
            pass
        return str(cid)


# -------------------------------------------------------------
# Widget de vídeo
# -------------------------------------------------------------
class VideoWidget(QWidget):
    clicked = Signal(int, int)  # x, y en coords de imagen

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel("Sin señal")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 360)
        self.label.setScaledContents(True)  # facilita el mapeo de clics
        lay = QVBoxLayout(self)
        lay.addWidget(self.label)
        self._last_qimg_size: Optional[Tuple[int, int]] = None

    @Slot(QImage)
    def set_image(self, qimg: QImage):
        if qimg.isNull():
            return
        self._last_qimg_size = (qimg.width(), qimg.height())
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def mousePressEvent(self, ev):
        pm = self.label.pixmap()
        if self._last_qimg_size is None or pm is None:
            return

        # Tamaño real del QPixmap mostrado (tiene en cuenta DPI/escala)
        pw, ph = pm.width(), pm.height()

        # Tamaño del frame original (QImage)
        iw, ih = self._last_qimg_size

        # Coordenadas del clic en el label
        lx = float(ev.position().x())
        ly = float(ev.position().y())

        # Mapea a coords de imagen usando el tamaño del pixmap 
        x_img = int(lx * iw / max(pw, 1))
        y_img = int(ly * ih / max(ph, 1))

        self.clicked.emit(x_img, y_img)


# -------------------------------------------------------------
# Ventana principal
# -------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tracking App – PySide6 GUI (esqueleto)")
        self.resize(1200, 800)

        # Estado
        self.cfg = load_models_yaml()
        self.settings = PipelineSettings()

        # Centro (vídeo)
        self.video = VideoWidget(self)
        self.video.clicked.connect(self.on_video_clicked)
        self.setCentralWidget(self.video)

        # Docks
        self._init_controls_dock()
        self._init_status_dock()

        # Barra de estado
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Listo")

        # Workers
        self.pipeline = PipelineWorker(self.cfg)
        self.pipeline.image_ready.connect(self.on_image)        # ← solo on_image
        self.pipeline.stats_ready.connect(self.on_stats)
        self.pipeline.error.connect(self.on_error)
        self.pipeline.perf_ready.connect(self.on_perf)          # ← conecta ANTES de start
        self.pipeline.start()

        self.camera = CameraWorker(0)
        self.camera.frame_ready.connect(self.on_frame)
        self.camera.opened.connect(self.on_camera_opened)
        self.camera.error.connect(self.on_error)
        self.camera.start()

        # Acciones menú archivo (abrir vídeo, salir)
        self._init_menu()

        # Inicializar UI con valores
        self._apply_settings_to_ui()

        self._avg_win = 30  # ventana en nº de frames (ajústala si quieres)
        self._avg = {
            "FPS": deque(maxlen=self._avg_win),
            "Δe2e": deque(maxlen=self._avg_win),
            "ΔYOLO": deque(maxlen=self._avg_win),
            "ΔMOT": deque(maxlen=self._avg_win),
            "ΔSOT": deque(maxlen=self._avg_win),
            "ΔSEG": deque(maxlen=self._avg_win),
        }


    # ---- UI: docks ----
    def _init_controls_dock(self):
        dock = QDockWidget("Controles", self)
        dock.setObjectName("dock_controls")

        tabs = QTabWidget()

        # ---------- Pestaña GENERAL ----------
        general = QWidget(); v = QVBoxLayout(general)

        gb_mode = QGroupBox("Modo")
        b_det = QRadioButton("Detección (YOLOv8 + ByteTrack)")
        b_sot = QRadioButton("Seguimiento local (Multi-KCF)")
        self.grp_mode = QButtonGroup(self)
        self.grp_mode.addButton(b_det, 1); self.grp_mode.addButton(b_sot, 2)
        b_det.setChecked(True)
        mlay = QVBoxLayout(gb_mode); mlay.addWidget(b_det); mlay.addWidget(b_sot)

        # --- Re-detección (solo KCF) ---
        self.gb_redetect = QGroupBox("Re-detección (solo KCF)")
        red_v = QVBoxLayout(self.gb_redetect)

        self.rb_redetect_off = QRadioButton("No redetectar")
        self.rb_redetect_on  = QRadioButton("Sí, cada N frames:")

        self.spn_redetect = QSpinBox()
        self.spn_redetect.setRange(1, 600)      # 1..600 cuando está ON
        self.spn_redetect.setSuffix(" frames")

        # Estado inicial (por defecto OFF → 0)
        if int(self.settings.kcf_redetect_interval) == 0:
            self.rb_redetect_off.setChecked(True)
            self.spn_redetect.setValue(30)      # valor por defecto cuando elijas Sí
        else:
            self.rb_redetect_on.setChecked(True)
            self.spn_redetect.setValue(int(self.settings.kcf_redetect_interval))

        row = QHBoxLayout()
        row.addWidget(self.rb_redetect_on)
        row.addWidget(self.spn_redetect, 1)

        red_v.addWidget(self.rb_redetect_off)
        red_v.addLayout(row)

        # Solo disponible en modo KCF; el spin solo si está ON
        self.gb_redetect.setEnabled(False)
        self.spn_redetect.setEnabled(self.rb_redetect_on.isChecked())

        # Conexiones
        self.rb_redetect_off.toggled.connect(self._on_redetect_option_changed)
        self.rb_redetect_on.toggled.connect(self._on_redetect_option_changed)
        self.spn_redetect.valueChanged.connect(self._on_redetect_value_changed)



        self.chk_focus = QCheckBox("Focus ON/OFF")
        self.chk_seg = QCheckBox("Segmentación asistida")
        self.btn_seg_params = QPushButton("Parámetros…")
        self.btn_seg_params.clicked.connect(self.open_seg_params)

        # --- Backend de segmentación ---
        gb_seg_backend = QGroupBox("Backend de segmentación")
        seg_lay = QVBoxLayout(gb_seg_backend)

        self.cmb_seg_backend = QComboBox()
        self.cmb_seg_backend.addItems(["Básico (FloodFill/HSV)", "SAM-ligero (clics)"])
        # sincroniza valor inicial
        idx0 = {"basic":0, "sam":1}.get(self.settings.seg_backend, 0)
        self.cmb_seg_backend.setCurrentIndex(idx0)
        self.cmb_seg_backend.currentIndexChanged.connect(self.on_seg_backend_changed)
        
        # Controles para SAM (solo visibles con SAM)
        sam_row = QHBoxLayout()
        self.btn_sam_pos = QPushButton("Clic +")
        self.btn_sam_neg = QPushButton("Clic –")
        self.btn_sam_reset = QPushButton("Reset clics")
        for b in (self.btn_sam_pos, self.btn_sam_neg, self.btn_sam_reset):
            b.setEnabled(False)
        sam_row.addWidget(self.btn_sam_pos); sam_row.addWidget(self.btn_sam_neg); sam_row.addWidget(self.btn_sam_reset)
        seg_lay.addWidget(self.cmb_seg_backend); seg_lay.addLayout(sam_row)

        gb_model = QGroupBox("Modelo de detección")
        self.cmb_model = QComboBox(); self.cmb_model.addItems(["coco", "militar"])
        self.cmb_model.setCurrentText("coco")
        fl = QFormLayout(gb_model); fl.addRow("Modelo:", self.cmb_model)

        # Estos checkboxes gobernarán la visibilidad de las pestañas de la derecha
        self.chk_info = QCheckBox("Mostrar panel de información")
        self.chk_coords = QCheckBox("Mostrar panel de coordenadas")
        self.chk_keys = QCheckBox("Mostrar panel de ayuda")
        self.chk_logcsv = QCheckBox("Grabar métricas (CSV)")
        self.chk_logcsv.toggled.connect(self._on_logcsv_toggled)
        for c in (self.chk_info, self.chk_coords, self.chk_keys):
            c.setChecked(True)

        self.btn_exit = QPushButton("Exit"); self.btn_exit.clicked.connect(self.close)

        self.btn_sam_pos.clicked.connect(lambda: self.set_sam_click_mode(True))
        self.btn_sam_neg.clicked.connect(lambda: self.set_sam_click_mode(False))
        self.btn_sam_reset.clicked.connect(self.reset_sam_clicks) 

        self.btn_roi = QPushButton("Pantallazo / Seleccionar ROI")
        self.btn_roi.clicked.connect(self.capture_roi)

        for wdg in (gb_mode, self.chk_focus, self.chk_seg, self.btn_seg_params,
                    gb_model, self.btn_roi,
                    self.chk_info, self.chk_coords, self.chk_keys, self.btn_exit,
                    self.gb_redetect, gb_seg_backend, self.chk_logcsv):
            v.addWidget(wdg)
        v.addStretch(1)

        tabs.addTab(general, "General")

        dock.setWidget(tabs)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.controls_dock = dock

        # Conexiones
        self.grp_mode.idToggled.connect(self.on_mode_changed)
        self.chk_focus.toggled.connect(self.on_focus_toggled)
        self.chk_seg.toggled.connect(self.on_seg_toggled)
        self.cmb_model.currentTextChanged.connect(self.on_model_changed)

        # Estas 3 conexiones controlan las pestañas del dock derecho
        self.chk_info.toggled.connect(self._refresh_status_tabs_visibility)
        self.chk_coords.toggled.connect(self._refresh_status_tabs_visibility)
        self.chk_keys.toggled.connect(self._refresh_status_tabs_visibility)

    def _init_status_dock(self):
        dock = QDockWidget("Estado", self)
        dock.setObjectName("dock_status")

        self.status_tabs = QTabWidget()

        # ---- Tab INFO ----
        self.tab_info = QWidget(); vi = QVBoxLayout(self.tab_info)
        self.lbl_fps = QLabel("FPS: -")
        self.lbl_mode = QLabel("Modo: -")
        self.lbl_model = QLabel("Modelo: -")
        self.lbl_weights = QLabel("Pesos: -")

        self.lbl_timings = QLabel("ΔYOLO: --   ΔMOT: --   ΔSOT: --   ΔSEG: --")
        self.lbl_misc    = QLabel("N=0   img=?   ROIs=0")
        self.lbl_extra = QLabel("Resolución: -    Centro óptico: -    #Objetos: -")
        for lab in (self.lbl_fps, self.lbl_mode, self.lbl_model, self.lbl_weights, self.lbl_timings, self.lbl_misc, self.lbl_extra):
            vi.addWidget(lab)

        self.log = QTextEdit(); self.log.setReadOnly(True)

        for lab in (self.lbl_fps, self.lbl_mode, self.lbl_model, self.lbl_weights, self.lbl_timings, self.lbl_misc):
            vi.addWidget(lab)
        vi.addWidget(self.log)

        # ---- Tab COORDS ----
        self.tab_coords = QWidget(); vc = QVBoxLayout(self.tab_coords)
        self.txt_coords = QTextEdit(); self.txt_coords.setReadOnly(True)
        vc.addWidget(self.txt_coords)

        self.tbl_objects = QTableWidget(0, 4)
        self.tbl_objects.setHorizontalHeaderLabels(["ID", "Clase", "BBox (x,y,w,h)", "Centro (px)"])
        self.tbl_objects.verticalHeader().setVisible(False)
        self.tbl_objects.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_objects.setSelectionMode(QAbstractItemView.NoSelection)
        vc.addWidget(self.tbl_objects)


        # ---- Tab AYUDA ----
        self.tab_keys = QWidget(); vk = QVBoxLayout(self.tab_keys)
        self.txt_keys = QTextEdit()
        self.txt_keys.setReadOnly(True)
        self.txt_keys.setAcceptRichText(True)          
        self.txt_keys.setLineWrapMode(QTextEdit.WidgetWidth)                   
        self.txt_keys.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.txt_keys.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)      
        self.txt_keys.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)         
        self.txt_keys.document().setDocumentMargin(12)                         
        vk.addWidget(self.txt_keys)       

        # Añadimos inicialmente las tres
        self.status_tabs.addTab(self.tab_info, "Info")
        self.status_tabs.addTab(self.tab_coords, "Coordenadas")
        self.status_tabs.addTab(self.tab_keys, "Ayuda")

        dock.setWidget(self.status_tabs)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.status_dock = dock
        self._populate_help_tab()


    def _populate_help_tab(self):
        html = """
        <style>
        body { font-family: sans-serif; }
        h3 { margin: 0.4em 0 0.2em; }
        h4 { margin: 0.6em 0 0.3em; }
        ul { margin-top: 0.2em; }
        code, .kbd { font-family: ui-monospace, Consolas, monospace; }
        .kbd { border:1px solid #aaa; border-radius:4px; padding:0 4px; }
        .note { color:#444; }
        </style>

        <h3>Ayuda rápida</h3>
        <p>Resumen de opciones, interacciones y notas de la aplicación.</p>

        <h4>Modos de operación</h4>
        <ul>
        <li><b>Detección (YOLOv8 + ByteTrack)</b>: cajas con ID persistente. Métricas dominadas por <code>ΔYOLO</code>.</li>
        <li><b>Seguimiento local (Multi-KCF)</b>: seguimiento por una o varias ROI. Métrica clave <code>ΔSOT</code>.</li>
        </ul>

        <h4>Re-detección (solo KCF)</h4>
        <ul>
        <li>Opciones: <b>No redetectar</b> / <b>Sí, cada N frames</b>.</li>
        <li>Compromiso: N bajo corrige deriva con más invocaciones a YOLO; N alto reduce carga pero tolera mayor deriva.</li>
        </ul>

        <h4>Focus</h4>
        <ul>
        <li>Activa <b>Focus ON/OFF</b> y haz clic dentro de una caja para centrar/ampliar esa zona. Segundo clic quita el foco.</li>
        </ul>

        <h4>Segmentación asistida</h4>
        <ul>
        <li><b>Backend básico (FloodFill/HSV)</b>:
            <ul>
            <li><b>Preset 1</b>: FloodFill.</li>
            <li><b>Preset 2</b>: FloodFill + morfología.</li>
            <li><b>Preset 3</b>: Umbral HSV.</li>
            <li>Presets 1/2 requieren un clic semilla en el vídeo.</li>
            </ul>
        </li>
        <li><b>Backend SAM-ligero</b>:
            <ul>
            <li>Usa botones <span class="kbd">Clic +</span> / <span class="kbd">Clic –</span> para añadir puntos positivos/negativos.</li>
            <li><span class="kbd">Reset clics</span> borra los prompts.</li>
            <li>La máscara se convierte a <em>bbox</em> y puede inicializar automáticamente KCF (cambia a modo SOT).</li>
            </ul>
        </li>
        </ul>

        <h4>Modelo de detección</h4>
        <ul>
        <li>Selector: <b>coco</b> / <b>militar</b>. Al conmutar se recargan pesos; ByteTrack puede reiniciar IDs para evitar ambigüedades.</li>
        </ul>

        <h4>Pantallazo / Seleccionar ROI</h4>
        <ul>
        <li>Abre un diálogo de selección. <span class="kbd">Enter</span> confirma, <span class="kbd">Esc</span> cancela.</li>
        <li>Si la vista está reescalada, las coordenadas se remapean al tamaño original del frame.</li>
        <li>La ROI inicializa KCF y el sistema pasa a modo SOT.</li>
        </ul>

        <h4>Paneles y registro</h4>
        <ul>
        <li><b>Info</b>: FPS y latencias (<code>ΔYOLO</code>, <code>ΔMOT</code>, <code>ΔSOT</code>, <code>ΔSEG</code>, <code>Δe2e</code>), modelo y pesos.</li>
        <li><b>Coordenadas</b>: listado textual y tabla con <em>ID, clase, bbox, centro</em>.</li>
        <li><b>Ayuda</b>: esta guía rápida y observaciones.</li>
        <li><b>Grabar métricas (CSV)</b>: guarda un CSV con medidas por frame en <code>runs/metrics</code>.</li>
        </ul>

        <h4>Menú Archivo</h4>
        <ul>
        <li><b>Abrir vídeo…</b> para reproducir un archivo en lugar de la cámara.</li>
        <li><b>Salir</b>.</li>
        </ul>

        <h4>Observaciones</h4>
        <ul class="note">
        <li><b>Δe2e</b> mide desde la captura hasta la presentación; por concurrencia y descarte de frames, no es la suma exacta de etapas ni coincide necesariamente con <code>1/FPS</code>.</li>
        <li>En SOT con redetección, <b>ΔYOLO</b> aparece sólo en los ciclos que invocan el detector; el promedio integra frames con/sin detección.</li>
        <li>El centro óptico mostrado es el punto medio de la resolución actual.</li>
        </ul>
        """
        self.txt_keys.setHtml(html)

    def _init_menu(self):
        menu = self.menuBar()
        m_file = menu.addMenu("Archivo")
        act_open = QAction("Abrir vídeo…", self)
        act_open.triggered.connect(self.open_video)
        m_file.addAction(act_open)
        act_exit = QAction("Salir", self)
        act_exit.triggered.connect(self.close)
        m_file.addAction(act_exit)

    # ---- Slots / eventos ----
    @Slot(dict)
    def on_stats(self, d: dict):
        self.lbl_mode.setText(f"Modo: {d.get('mode','-')}")
        self.lbl_model.setText(f"Modelo: {d.get('model','-')}")
        self.lbl_weights.setText(f"Pesos: {d.get('weights','-')}")
        if hasattr(self, "txt_coords"):
            self.txt_coords.setPlainText(d.get("coords_text", ""))        
        if hasattr(self, "txt_keys"):
            txt = d.get("keys_text", None)
            if txt:  # solo si el worker manda algo explícito
                self.txt_keys.setPlainText(txt)        

        # Línea extra en Info
        res = d.get("resolution", "-")
        oc  = d.get("optical_center", ("-","-"))
        nob = d.get("n_objects", 0)
        self.lbl_extra.setText(f"Resolución: {res}    Centro óptico: {oc}    #Objetos: {nob}")

        # Tabla de objetos
        rows = d.get("objects_rows", [])
        if hasattr(self, "tbl_objects"):
            t = self.tbl_objects
            t.setRowCount(len(rows))
            for r, obj in enumerate(rows):
                t.setItem(r, 0, QTableWidgetItem(str(obj.get("id", "-"))))
                t.setItem(r, 1, QTableWidgetItem(str(obj.get("class", "-"))))
                bx = obj.get("bbox", None)
                cx = obj.get("center", None)
                t.setItem(r, 2, QTableWidgetItem("-" if not bx else f"{bx[0]},{bx[1]},{bx[2]},{bx[3]}"))
                t.setItem(r, 3, QTableWidgetItem("-" if not cx else f"{cx[0]},{cx[1]}"))
            t.resizeColumnsToContents()



    @Slot(str)
    def on_error(self, msg: str):
        self.status.showMessage(msg, 5000)
        self.log.append(msg)

    @Slot(int, int, float)
    def on_camera_opened(self, w: int, h: int, fps: float):
        self.log.append(f"Cámara abierta: {w}x{h} @ {fps:.1f} FPS")

    @Slot(np.ndarray)
    def on_frame(self, frame: np.ndarray):
        # Empujar al pipeline
        self.pipeline.enqueue(frame)

    @Slot(dict)
    def on_perf(self, perf: dict):
        # guarda último dict de perf; Δe2e se cierra al presentar imagen
        self._last_perf = perf

    @Slot(QImage)
    def on_image(self, qimg: QImage):
        # marca de presentación
        t_pres = _now_ns()
        self.video.set_image(qimg)   # pinta
        # si hay perf, calcula Δ… y actualiza panel / CSV
        if hasattr(self, "_last_perf") and self._last_perf:
            self._consume_and_render_metrics(self._last_perf, t_pres)
            self._last_perf = None

    def _apply_settings_to_ui(self):
        # refleja self.settings en los controles y avisa al pipeline
        id_mode = 1 if self.settings.mode == "det" else 2
        self.grp_mode.button(id_mode).setChecked(True)
        self.chk_focus.setChecked(self.settings.focus_on)
        self.chk_seg.setChecked(self.settings.seg_on)
        self.cmb_model.setCurrentText(self.settings.model_name)
        self.pipeline.update_settings(self.settings)

        if int(self.settings.kcf_redetect_interval) == 0:
            self.rb_redetect_off.setChecked(True)
        else:
            self.rb_redetect_on.setChecked(True)
            self.spn_redetect.setValue(int(self.settings.kcf_redetect_interval))
        self._apply_redetect_controls_enabled()

    # --- callbacks de controles ---
    def on_mode_changed(self, _id: int, checked: bool):
        if not checked:
            return
        if _id == 1:   
            self.settings.mode = "det"
            self.pipeline.exit_sot()
            self.pipeline.reset_sam()            
            self.log.append("Modo → det")
        else:         
            self.settings.mode = "sot"
            self.pipeline.request_kcf_from_detection()
            self.log.append("Modo → sot (init desde detección)")
        self.pipeline.update_settings(self.settings)
        self._apply_redetect_controls_enabled()
        self._avg_reset()


    def on_focus_toggled(self, on: bool):
        self.settings.focus_on = on
        self.pipeline.update_settings(self.settings)
        if not on:
            self.pipeline.clear_focus()
        else:
            self.status.showMessage("Focus ON: haz clic dentro de una bbox", 3000)


    def on_seg_toggled(self, on: bool):
        self.settings.seg_on = on
        self.pipeline.update_settings(self.settings)

        if not on:
            # Apagaste la segmentación: limpia SAM/SOT y fuerza modo detección
            self.pipeline.reset_sam()
            self.pipeline.exit_sot()
            self.settings.mode = "det"
            # Sincroniza radio button y re-aplica
            self.grp_mode.button(1).setChecked(True)
            self.pipeline.update_settings(self.settings)
            self.log.append("Segmentación OFF → reinicio SAM/KCF y modo DET")

        if on and self.settings.seg_backend == "basic":
            self.open_seg_params()
                        
        is_sam = (self.settings.seg_backend == "sam")
        for b in (self.btn_sam_pos, self.btn_sam_neg, self.btn_sam_reset):
            b.setEnabled(is_sam and on)

    def on_model_changed(self, name: str):
        self.settings.model_name = name
        self.log.append(f"Modelo → {name}")
        # Aquí podrías también resetear trackers, etc.
        self.pipeline.update_settings(self.settings)
        self._avg_reset()

    def open_video(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Abrir vídeo", "", "Vídeos (*.mp4 *.avi *.mov);;Todos (*.*)")
        if fn:
            # Parar anterior cámara y lanzar nueva fuente
            self.camera.stop()
            self.camera.wait(1000)
            self.camera = CameraWorker(fn)
            self.camera.frame_ready.connect(self.on_frame)
            self.camera.opened.connect(self.on_camera_opened)
            self.camera.error.connect(self.on_error)
            self.camera.start()
            self.log.append(f"Fuente de vídeo: {fn}")

    def open_seg_params(self):
        dlg = SegParamsDialog(current_preset=self.settings.seg_preset, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.settings.seg_preset = dlg.selected_preset()
            self.log.append(f"Segmentación preset → {self.settings.seg_preset}")
            self.pipeline.update_settings(self.settings)
        else:
            if not self.settings.seg_on:
                return
            # Si cancelas al activarla, apaga la segmentación
            self.chk_seg.setChecked(False)
            self.settings.seg_on = False
            self.pipeline.update_settings(self.settings)

    def capture_roi(self):

        # 1) Congelar imagen
        canvas = None
        if hasattr(self.pipeline, "snapshot_canvas"):
            canvas = self.pipeline.snapshot_canvas()
        frame_orig = None
        if hasattr(self.pipeline, "snapshot_frame"):
            frame_orig = self.pipeline.snapshot_frame()

        if canvas is None and frame_orig is None and hasattr(self, "camera"):
            frame_orig = self.camera.grab_last_frame()

        if frame_orig is None and canvas is None:
            self.on_error("No hay frame disponible para ROI")
            return

        # Si no hay canvas, usamos el original también como "display"
        disp = canvas if canvas is not None else frame_orig.copy()

        # 2) Selector ROI de OpenCV (bloqueante, como en main.py)
        try:
            r = cv2.selectROI("Selecciona ROI (Enter=OK, Esc=Cancelar)", disp, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Selecciona ROI (Enter=OK, Esc=Cancelar)")
        except Exception as e:
            self.on_error(f"No se pudo abrir selector ROI: {e}")
            return

        x_d, y_d, w_d, h_d = map(int, r)
        if w_d <= 2 or h_d <= 2:
            self.on_error("ROI no válida")
            return

        # 3) Remap a ORIGINAL si pintaste sobre canvas escalado
        if frame_orig is None:
            frame_orig = disp  # por seguridad

        H, W = frame_orig.shape[:2]
        dH, dW = disp.shape[:2]
        if (W, H) != (dW, dH):
            sx, sy = W / float(dW), H / float(dH)
            x = int(round(x_d * sx)); y = int(round(y_d * sy))
            w = int(round(w_d * sx)); h = int(round(h_d * sy))
        else:
            x, y, w, h = x_d, y_d, w_d, h_d

        # Clipping
        x = max(0, min(W - 1, x)); y = max(0, min(H - 1, y))
        w = max(3, min(W - x, w)); h = max(3, min(H - y, h))

        # 4) Inicializa KCF + cambia a SOT sin disparar on_mode_changed
        self.pipeline.request_init_sot(frame_orig, [(x, y, w, h)])
        self.pipeline.set_roi_rect((x, y, w, h))

        self.settings.mode = "sot"
        self.grp_mode.blockSignals(True)
        self.grp_mode.button(2).setChecked(True)  # 1=Detección, 2=SOT
        self.grp_mode.blockSignals(False)
        self.pipeline.update_settings(self.settings)
        self.log.append(f"ROI seleccionado (orig): x={x} y={y} w={w} h={h}")


    def closeEvent(self, e: QCloseEvent):
        try:
            self.camera.stop()
            self.pipeline.stop()
            self.camera.wait(1500)
            self.pipeline.wait(1500)
        finally:
            e.accept()

    def on_video_clicked(self, x: int, y: int):
        # Focus por clic (si está ON)
        if self.settings.focus_on:
            self.pipeline.request_focus_at((x, y))

        # Segmentación básica: presets 1/2 necesitan seed SIEMPRE
        if self.settings.seg_on and self.settings.seg_backend == "basic" and self.settings.seg_preset in (1, 2):
            self.pipeline.request_seg_click((x, y))

        # SAM: un solo camino, clics +/–
        if self.settings.seg_on and self.settings.seg_backend == "sam":
            positive = getattr(self, "_sam_mode_positive", True)
            self.pipeline.add_sam_click((x, y), positive)


    def use_detection_for_kcf(self):
        self.pipeline.request_kcf_from_detection()
        self.settings.mode = "sot"
        self._apply_settings_to_ui()

    def _refresh_status_tabs_visibility(self):
        tabs = self.status_tabs
        while tabs.count():
            tabs.removeTab(0)
        if self.chk_info.isChecked():
            tabs.addTab(self.tab_info, "Info")
        if self.chk_coords.isChecked():
            tabs.addTab(self.tab_coords, "Coordenadas")
        if self.chk_keys.isChecked():
            tabs.addTab(self.tab_keys, "Ayuda")

    def _apply_redetect_controls_enabled(self):
        enable = (self.settings.mode == "sot")
        self.gb_redetect.setEnabled(enable)
        self.spn_redetect.setEnabled(enable and self.rb_redetect_on.isChecked())

    def _on_redetect_option_changed(self, checked: bool):
        if self.rb_redetect_off.isChecked():
            self.settings.kcf_redetect_interval = 0
        else:
            self.settings.kcf_redetect_interval = max(1, int(self.spn_redetect.value()))
        self.pipeline.update_settings(self.settings)
        self._apply_redetect_controls_enabled()

    def _on_redetect_value_changed(self, val: int):
        if self.rb_redetect_on.isChecked():
            self.settings.kcf_redetect_interval = max(1, int(val))
            self.pipeline.update_settings(self.settings)

    def on_seg_backend_changed(self, idx: int):
        mapping = {0:"basic", 1:"sam"}
        self.settings.seg_backend = mapping.get(idx)
        self.pipeline.update_settings(self.settings)
        # Habilitar/ocultar controles SAM
        is_sam = (self.settings.seg_backend == "sam")
        for b in (self.btn_sam_pos, self.btn_sam_neg, self.btn_sam_reset):
            b.setEnabled(is_sam and self.chk_seg.isChecked())

    def set_sam_click_mode(self, positive: bool):
        # Marca modo (se usa al hacer click en el video)
        self._sam_mode_positive = positive
        # feedback rápido en status bar
        self.status.showMessage(f"SAM: modo clic {'+' if positive else '–'}", 2000)

    def reset_sam_clicks(self):
        self.pipeline.reset_sam()

    def _consume_and_render_metrics(self, perf: dict, t_pres_ns: int):
        out = {}
        if perf.get("tyi") and perf.get("tyo"):
            out["ΔYOLO"] = _ms(perf["tyo"] - perf["tyi"])
        if perf.get("tmi") and perf.get("tmo"):
            out["ΔMOT"]  = _ms(perf["tmo"] - perf["tmi"])
        if perf.get("tsi") and perf.get("tso"):
            out["ΔSOT"]  = _ms(perf["tso"] - perf["tsi"])
        if perf.get("tsegi") and perf.get("tsego"):
            out["ΔSEG"]  = _ms(perf["tsego"] - perf["tsegi"])
        if perf.get("t_cap"):
            out["Δe2e"]  = _ms(t_pres_ns - perf["t_cap"])

        if hasattr(self, "_prev_t_pres_ns") and self._prev_t_pres_ns:
            dt_ms = _ms(t_pres_ns - self._prev_t_pres_ns)
            if dt_ms > 0:
                out["FPS"] = 1000.0 / dt_ms
        self._prev_t_pres_ns = t_pres_ns

        # 1) Actualiza medias móviles con los valores instantáneos
        self._avg_push(out)

        # 2) Calcula las medias
        fps_avg  = self._avg_mean("FPS")
        e2e_avg  = self._avg_mean("Δe2e")
        yolo_avg = self._avg_mean("ΔYOLO")
        mot_avg  = self._avg_mean("ΔMOT")
        sot_avg  = self._avg_mean("ΔSOT")
        seg_avg  = self._avg_mean("ΔSEG")

        # 3) Formateo seguro (instante y promedio)
        def fmt_num(v, dec=1, fallback="-"):
            try:
                return f"{v:.{dec}f}"
            except Exception:
                return fallback

        def fmt_ms(v, fallback="--"):
            try:
                return f"{v:.0f} ms"
            except Exception:
                return fallback

        # Línea 1: FPS y Δe2e
        fps_i = out.get("FPS")
        e2e_i = out.get("Δe2e")
        line1 = (
            f"FPS: {fmt_num(fps_i, dec=1)}"
            + (f"  (avg {fmt_num(fps_avg, dec=1)})" if fps_avg is not None else "")
            + "   "
            + f"Δe2e: {fmt_ms(e2e_i)}"
            + (f"  (avg {fmt_ms(e2e_avg)})" if e2e_avg is not None else "")
        )

        # Línea 2: ΔYOLO/ΔMOT/ΔSOT/ΔSEG con avg si existe
        y_i, y_a = out.get("ΔYOLO"), yolo_avg
        m_i, m_a = out.get("ΔMOT"),  mot_avg
        s_i, s_a = out.get("ΔSOT"),  sot_avg
        g_i, g_a = out.get("ΔSEG"),  seg_avg

        y_txt = (f"{fmt_ms(y_i)}" + (f" (avg {fmt_ms(y_a)})" if y_a is not None else "")) if (y_i is not None or y_a is not None) else "--"
        m_txt = (f"{fmt_ms(m_i)}" + (f" (avg {fmt_ms(m_a)})" if m_a is not None else "")) if (m_i is not None or m_a is not None) else "--"
        s_txt = (f"{fmt_ms(s_i)}" + (f" (avg {fmt_ms(s_a)})" if s_a is not None else "")) if (s_i is not None or s_a is not None) else "--"
        g_txt = (f"{fmt_ms(g_i)}" + (f" (avg {fmt_ms(g_a)})" if g_a is not None else "")) if (g_i is not None or g_a is not None) else "--"

        line2 = f"ΔYOLO: {y_txt}   ΔMOT: {m_txt}   ΔSOT: {s_txt}   ΔSEG: {g_txt}"

        # Línea 3 (metadatos), como ya la formateabas:
        line3 = f"N={perf.get('N',0)}  img={perf.get('imgsz','?')}  ROIs={perf.get('num_rois',0)}"

        # Pintar
        self.lbl_fps.setText(line1)
        self.lbl_mode.setText(f"Modo: {perf.get('mode','-')}")
        self.lbl_timings.setText(line2)
        self.lbl_misc.setText(line3)

        # CSV (mantienes el instantáneo, tal y como ya lo tenías)
        if getattr(self, "_logging_enabled", False):
            self._write_csv_row(perf, out)
            try: self._csv_fp.flush()
            except Exception: pass


    def _on_logcsv_toggled(self, on: bool):
        if on:
            self._open_metrics_csv("escenario")  # cámbialo al nombre que quieras
            self.status.showMessage("Grabación de métricas ON", 3000)
        else:
            self._close_metrics_csv()
            self.status.showMessage("Grabación de métricas OFF", 3000)

    def _open_metrics_csv(self, scenario="escenario"):
        Path("runs/metrics").mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_fp = open(f"runs/metrics/{ts}_{scenario}.csv", "w", newline="")
        self._csv_writer = csv.writer(self._csv_fp)
        self._csv_writer.writerow([
            "ts","mode","model","N","imgsz","num_rois",
            "ΔYOLO_ms","ΔMOT_ms","ΔSOT_ms","ΔSEG_ms","Δe2e_ms","FPS",
            "ids_new","ids_lost"
        ])
        self._logging_enabled = True

    def _close_metrics_csv(self):
        if getattr(self, "_csv_fp", None):
            self._csv_fp.flush(); self._csv_fp.close()
        self._logging_enabled = False

    def _write_csv_row(self, perf: dict, out: dict):
        row = [
            dt.datetime.now().isoformat(timespec="milliseconds"),
            perf.get("mode",""), perf.get("model",""),
            perf.get("N",""), perf.get("imgsz",""), perf.get("num_rois",""),
            f"{out.get('ΔYOLO','')}", f"{out.get('ΔMOT','')}", f"{out.get('ΔSOT','')}",
            f"{out.get('ΔSEG','')}",  f"{out.get('Δe2e','')}", f"{out.get('FPS','')}",
            perf.get("ids_new",""), perf.get("ids_lost","")
        ]
        self._csv_writer.writerow(row)

    def _avg_reset(self):
        for q in self._avg.values():
            q.clear()

    def _avg_push(self, out: dict):
        # Empuja sólo si hay valor numérico
        for k, q in self._avg.items():
            v = out.get(k, None)
            if isinstance(v, (int, float)) and not math.isnan(v):
                q.append(float(v))

    def _avg_mean(self, key: str):
        q = self._avg.get(key)
        if not q or len(q) == 0:
            return None
        return sum(q) / len(q)


# -------------------------------------------------------------
# main
# -------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
