# manual_roi_manager.py  ·  v2
from __future__ import annotations
import cv2, logging
from typing import List, Tuple
from bounding_box import BBox
from tracker_utils import initialize_tracker, OpenCVTracker

class ManualROIManager:
    """
    Permite añadir → seguir → eliminar ROI manuales (clic-para-borrar).
    API:
      • __init__(window_name, tracker_kind='KCF')
      • add_tracker(frame)
      • update_and_draw(frame) → frame
    """
    def __init__(self, window_name: str = "frame", tracker_kind: str = "KCF"):
        self.window_name = window_name
        self.tracker_kind = tracker_kind
        self.trackers: List[Tuple[OpenCVTracker, BBox]] = []   # [(tracker, bbox)]
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse_click)

    # ---------- callbacks ---------- #
    def _on_mouse_click(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for i, (trk, box) in enumerate(self.trackers):
            if box.contains(x, y):
                logging.info(f"Eliminando ROI manual {box}")
                self.trackers.pop(i)
                break

    # ---------- API de usuario ---------- #
    def add_tracker(self, frame) -> None:
        roi = cv2.selectROI("Selecciona ROI", frame, False, False)  # xywh
        cv2.destroyWindow("Selecciona ROI")
        if roi and roi != (0, 0, 0, 0):
            bbox = BBox(*map(int, roi))
            tracker = initialize_tracker(frame, bbox, self.tracker_kind)
            self.trackers.append((tracker, bbox))
            logging.info(f"Añadido tracker manual {bbox}")

    def update_and_draw(self, frame):
        nuevos: list[Tuple[OpenCVTracker, BBox]] = []
        for trk, box in self.trackers:
            ok, new_xywh = trk.update(frame)
            if ok:
                box = BBox.from_xyxy(*map(int, new_xywh))
                cv2.rectangle(frame, (box.x, box.y),
                              (box.x + box.w, box.y + box.h), (255, 0, 0), 2)
                nuevos.append((trk, box))
            else:
                cv2.putText(frame, "ROI manual perdido", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        self.trackers = nuevos
        return frame
