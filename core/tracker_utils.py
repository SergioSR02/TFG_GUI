from __future__ import annotations
import cv2
from typing import Callable, Protocol
from bounding_box import BBox         

class OpenCVTracker(Protocol):        
    def init(self, frame, bbox_xywh): ...
    def update(self, frame) -> tuple[bool, tuple[int,int,int,int]]: ...

def _make(name: str) -> Callable[[], OpenCVTracker]:
    creator = getattr(cv2, f"Tracker{name}_create", None)
    if creator is None:
        raise RuntimeError(f"Tu OpenCV no incluye Tracker{name}.")
    return creator

TRACKER_FACTORIES = {
    "KCF": _make("KCF"),        
}

def initialize_tracker(frame, bbox, kind: str = "KCF") -> OpenCVTracker:   
    if not isinstance(bbox, BBox):
        bbox = BBox(*bbox)     
    tracker = TRACKER_FACTORIES[kind]()
    tracker.init(frame, bbox.xywh)    
    return tracker
