from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO


MODEL_ZOO = {
    "coco":  "yolov8n.pt",
    "militar": "config/models/best.pt",       
}


###############################################################################
# Configuración declarativa
###############################################################################
@dataclass(frozen=True)
class DetectionConfig:
    model_path: str = "yolov8n.pt"
    conf: float = 0.50
    iou: float = 0.65
    imgsz: int = 640
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    half: bool = True                # FP16 si hay CUDA
    classes: Optional[Sequence[int]] = None  # None → todas

###############################################################################
# Carga perezosa del modelo (se reutiliza con @lru_cache)
###############################################################################
@lru_cache(maxsize=1)
def _load_model(path: str, device: str, half: bool) -> YOLO:

    model = YOLO(path).to(device)
    if half and device.startswith("cuda"):
        model = model.half()
    return model

###############################################################################
# API pública
###############################################################################

def inicializar_modelo(tipo: str = "coco"):
    if tipo not in MODEL_ZOO:
        raise ValueError(f"Modelo desconocido: {tipo}. Usa {list(MODEL_ZOO)}")
    # usa la carga con device/half y cacheada
    cfg = DetectionConfig(model_path=MODEL_ZOO[tipo])
    return _load_model(cfg.model_path, cfg.device, cfg.half)



# def inicializar_modelo(tipo: str = "coco"):
#     """
#     Devuelve un objeto YOLO cargado con el modelo deseado.
#     """
#     if tipo not in MODEL_ZOO:
#         raise ValueError(f"Modelo desconocido: {tipo}. Usa {list(MODEL_ZOO)}")
#     return YOLO(MODEL_ZOO[tipo])


# def inicializar_modelo(cfg: DetectionConfig | None = None) -> YOLO:
#     """Devuelve una instancia de YOLO lista para inferir."""
#     cfg = cfg or DetectionConfig()
#     return _load_model(cfg.model_path, cfg.device, cfg.half)


def detectar_objetos(
    model: YOLO,
    frame: np.ndarray,
    cfg: DetectionConfig | None = None,
) -> sv.Detections:
    """
    Ejecuta YOLO y devuelve un objeto supervision.Detections siempre válido
    (aunque no haya detecciones).
    """
    cfg = cfg or DetectionConfig()

    # Ultralytics admite generar un iterador con stream=True
    results = model.predict(
        frame,
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=cfg.iou,
        classes=cfg.classes,
        stream=False,
        verbose=False,
        agnostic_nms=True,   # evita conflictos entre clases si un día amplías
        # max_det=200,       # opcional
        # vid_stride=2,      # opcional: reduce jitter en vídeo muy ruidoso
    )
    

    if not results or len(results[0].boxes) == 0:
        return sv.Detections.empty()

    # Convertir a objeto Detections y filtrar clases si procede
    dets = sv.Detections.from_ultralytics(results[0])
    if cfg.classes is not None:
        mask = np.isin(dets.class_id, cfg.classes)
        dets = dets[mask]

    return dets
