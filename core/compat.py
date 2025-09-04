"""
Compatibilidad entre tu API antigua y los módulos refactorizados.
Importa esto ANTES de importar tus propios módulos o Ultralytics.
"""
from pathlib import Path
import sys

# Asegúrate de que la carpeta con bbox.py, deteccion.py, etc. está en el path
sys.path.insert(0, str(Path(__file__).parent))

# ---- 1. Detección ----------------------------------------------------------
from detector_yolov8 import DetectionConfig, detectar_objetos as _detectar_nuevo
def detectar_objetos(modelo, frame,
                     conf_threshold=0.35, iou_threshold=0.50,
                     classes_to_detect=None):
    cfg = DetectionConfig(conf=conf_threshold,
                          iou=iou_threshold,
                          classes=classes_to_detect)
    return _detectar_nuevo(modelo, frame, cfg)

# ---- 2. Trackers utils (tuplas ↔︎ BBox) ------------------------------------
from bounding_box import BBox
from tracker_utils import initialize_tracker as _init_tracker
def initialize_tracker(frame, bbox_xywh, kind="KCF"):
    if not isinstance(bbox_xywh, BBox):
        bbox_xywh = BBox(*bbox_xywh)
    return _init_tracker(frame, bbox_xywh, kind)

# ---- 3. MouseSelector alias ------------------------------------------------
from ui_mouse import MouseSelector as _MouseSelectorNew
class MouseSelector(_MouseSelectorNew):
    # Tu código leía .selected_object   ➜  lo redirigimos a .selected
    @property
    def selected_object(self):
        return self.selected
    @selected_object.setter
    def selected_object(self, value):
        self.selected = value

    # El alias track_object que ya teníamos
    @property
    def track_object(self):
        return self.selected
    

# ---- 4. clip_bbox redirect (temporal) --------------------------------------
def clip_bbox(bbox, frame_shape):
    from bounding_box import BBox
    return BBox(*bbox).clip(*frame_shape).xywh
