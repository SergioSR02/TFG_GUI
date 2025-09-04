# bbox.py
from __future__ import annotations
import sys
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
#  Decorador adaptable: usa slots si la versión de Python lo soporta
# --------------------------------------------------------------------------- #
if sys.version_info >= (3, 10):
    _dataclass = dataclass(slots=True, frozen=True)
else:
    _dataclass = dataclass(frozen=True)          # sin 'slots' en Py < 3.10

# --------------------------------------------------------------------------- #
@_dataclass
class BBox:
    """
    Bounding-box en formato (x, y, w, h) con utilidades básicas.
    • x,y: esquina superior-izquierda
    • w,h: ancho y alto
    """
    # Para Py < 3.10 añadimos __slots__ manualmente
    if sys.version_info < (3, 10):
        __slots__ = ("x", "y", "w", "h")

    x: int
    y: int
    w: int
    h: int

    # ---------------- Conversión de formatos ---------------- #
    @property
    def xywh(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> "BBox":
        return cls(x1, y1, x2 - x1, y2 - y1)

    # ---------------- Operaciones geométricas ---------------- #
    def contains(self, px: int, py: int) -> bool:
        x1, y1, x2, y2 = self.xyxy
        return x1 <= px <= x2 and y1 <= py <= y2

    def iou(self, other: "BBox") -> float:
        ax1, ay1, ax2, ay2 = self.xyxy
        bx1, by1, bx2, by2 = other.xyxy
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        union = self.w * self.h + other.w * other.h - inter + 1e-6
        return inter / union

    def clip(self, frame_h: int, frame_w: int) -> "BBox":
        x = max(0, min(self.x, frame_w - 1))
        y = max(0, min(self.y, frame_h - 1))
        w = min(self.w, frame_w - x)
        h = min(self.h, frame_h - y)
        return BBox(x, y, w, h)
