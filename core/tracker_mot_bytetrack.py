# seguimiento.py – v3  (auto-compatible con todas las versiones de supervision)
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import inspect
import supervision as sv

# --------------------------------------------------------------------------- #
@dataclass
class TrackerConfig:
    track_high_thresh: float = 0.4
    track_low_thresh: float = 0.3               #0.1       #En deportes como atletismo mejor  subir track_low y bajar new_track
    new_track_thresh: float = 0.6               #0.7
    track_thresh: float = 0.25          # fallback para APIs antiguas
    match_thresh: float = 0.8           #    «         «

# --------------------------------------------------------------------------- #
def _build_kwargs(cfg: TrackerConfig) -> dict:
    """
    Devuelve solo los argumentos que el ByteTrack de tu instalación soporta.
    """
    sig = inspect.signature(sv.ByteTrack.__init__)
    valid_params = sig.parameters.keys()

    # Mapeo universal → params concretos
    candidate_kwargs = {
        # API nuevas (>=0.5.x)
        "track_high_thresh": cfg.track_high_thresh,
        "track_low_thresh":  cfg.track_low_thresh,
        "new_track_thresh":  cfg.new_track_thresh,
        # API antiguas (<=0.4.x)
        "track_thresh":      cfg.track_thresh,
        "match_thresh":      cfg.match_thresh,
    }
    # Filtra sólo los que existen en la firma
    return {k: v for k, v in candidate_kwargs.items() if k in valid_params}

# --------------------------------------------------------------------------- #
def inicializar_seguimiento(cfg: Optional[TrackerConfig] = None) -> sv.ByteTrack:
    """
    Crea un objeto ByteTrack con los parámetros adecuados para la versión
    instalada del paquete supervision.
    """
    cfg = cfg or TrackerConfig()
    kwargs = _build_kwargs(cfg)
    return sv.ByteTrack(**kwargs)

# --------------------------------------------------------------------------- #
def actualizar_seguimiento(
    tracker: sv.ByteTrack,
    detections: sv.Detections,
) -> sv.Detections:
    """Actualiza ByteTrack con las detecciones y devuelve las pistas activas."""
    return tracker.update_with_detections(detections)

# --------------------------------------------------------------------------- #
def reset_seguimiento(tracker: sv.ByteTrack) -> None:
    """Reinicia el estado interno sin crear otro objeto."""
    tracker.reset()
