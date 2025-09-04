# main.py  
# --------------------------------------------------
#
# Funcionalidad:
#   Tecla  d  →  detección (YOLOv8 + ByteTrack)
#   Tecla  t  →  tracking   (Multi-KCF)
#   Tecla  f  →  focus / zoom sobre bbox clicada
#   Tecla  c  →  segmentación por FloodFill / HSV (1-3)
#   Tecla  p  →  ROI manual con tracker KCF
#   Tecla  q  →  salir
# --------------------------------------------------

# ----- Compatibilidad y wrappers (debe ir antes de otros imports) ----------
import compat as compat  # NO eliminar aunque no se use explícitamente

# ----- Librerías estándar / 3rd-party --------------------------------------
import cv2
import numpy as np
import logging
import time
from pathlib import Path
from typing import List, Tuple

# ----- Módulos internos -----------------------------------------------------
from bounding_box import BBox
from detector_yolov8 import inicializar_modelo            # carga perezosa + FP16
from compat import detectar_objetos, initialize_tracker, MouseSelector
from tracker_mot_bytetrack import inicializar_seguimiento, actualizar_seguimiento
from tracker_sot_multikcf import MultiKCFTracker
from roi_segmentation import (
    segment_floodfill,
    segment_color_threshold,
    get_bbox_from_mask,
)
from ui_overlay import UIOverlay
from collections import deque
from time import perf_counter
import math
from comms import UdpPublisher
from appearance import hsv_fingerprint
from coord_utils import Optics, load_homography, pixel_to_plane
# ----- Configuración de logging --------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def to_jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return [to_jsonable(x) for x in o]
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    return o



# ========================================================================== #
#                                APLICACIÓN                                  #
# ========================================================================== #
class DetectorHybridTracker:
    def __init__(self, video_source, classes_to_detect=None):
        # --------- Entrada de vídeo ----------------------------------------
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise IOError(f"No se pudo abrir la fuente de vídeo: {video_source}")

        # --------- YOLO + ByteTrack ----------------------------------------
        self.modelo_actual = "coco"
        self.model = inicializar_modelo(self.modelo_actual)
        self.class_names = self.model.names

        self.classes_to_detect = classes_to_detect or []
        self.bt_tracker = inicializar_seguimiento()

        # --------- Estados --------------------------------------------------
        self.tracking_mode = False         # False = detección ; True = KCF
        self.focus_mode = False            # Zoom sobre objeto seleccionado
        self.segmentation_active = False   # Segmentación por FloodFill / HSV
        self.segmentation_method = 1       # 1:FloodFill, 2:+Morph, 3:HSV
        self.frame_count = 0
        self.redetect_interval = 30        # Re-detección cada N frames (modo KCF)

        # --------- Estructuras internas ------------------------------------
        self.current_boxes: List[BBox] = []          # Boxes visibles (detección o track)
        self.manual_trackers: List[Tuple[compat.OpenCVTracker, BBox]] = []
        self.mouse_selector = MouseSelector()
        self.kcf: MultiKCFTracker | None = None
        self.frame_current = None                    # Último frame (raw)
        

        # ---- Datos auxiliares para el zoom ----
        self._focus_roi = None          # (x1, y1, x2, y2) en coords originales
        self._focus_scale = None        # (sx, sy)  factor de escala -> display

        # --- HUD interactivo ---
        self.overlay = UIOverlay(scale=1.0, anchor="right")  # requiere ui_overlay.py
        self.overlay.show_menu = True  # pon False si prefieres que arranque oculto
        self._queued_virtual_key = None  # cola para clics en el HUD (teclas virtuales)

        self._t0 = perf_counter()
        self._lat_hist = deque(maxlen=30)
        self._yolo_ms = 0.0
        self._bt_ms   = 0.0
        self._kcf_ms  = 0.0

        ok_probe, frame0 = self.cap.read()
        if not ok_probe:
            raise IOError("No se pudo leer frame inicial para configurar geometría.")
        H0, W0 = frame0.shape[:2]
        self.frame_current = frame0
        self.opt = Optics(W=W0, H=H0, fovx_deg=78.0, fovy_deg=50.0)
        self.coord_mode = 0   # 0 PIXEL, 1 NORM, 2 ANGLE, 3 PLANE
        self.H_plane = load_homography("homography.json")  # opcional

        self._last_theta = None
        self._last_t = time.time()

        self.pub = UdpPublisher("239.0.0.1", 5005)
        # --------- Ventana principal ---------------------------------------
        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", self.on_mouse_click)

    # ====================================================================== #
    #                             EVENTOS RATÓN                              #
    # ====================================================================== #
    def on_mouse_click(self, event, x, y, flags, param):
        """Gestión global del clic (trackers manuales · segmentación · focus)."""

        # --- SIEMPRE delega primero al overlay (keycaps, arrastre, [X]) ---
        self.overlay.handle_mouse(event, x, y, flags)
        v = self.overlay.last_clicked_key()
        if v is not None:
            self._queued_virtual_key = v
            return

        # Si estamos arrastrando el panel o el clic cae dentro del panel, no propagar
        if getattr(self.overlay, "dragging", False):
            return
        if event == cv2.EVENT_LBUTTONDOWN and self.overlay.pointer_inside(x, y):
            return     
        
        if self.focus_mode and self._focus_roi and self._focus_scale:
            x1, y1, _, _ = self._focus_roi
            sx, sy = self._focus_scale
            x_o = int(x / sx + x1)
            y_o = int(y / sy + y1)

            if (event == cv2.EVENT_LBUTTONDOWN and
                    self.mouse_selector.selected_object and
                    self.mouse_selector.selected_object.contains(x_o, y_o)):
                # ⇲ Salir de focus
                self.focus_mode = False
                self.mouse_selector.selected_object = None
                self._focus_roi = None
                self._focus_scale = None
                logging.info("Focus desactivado.")
                return

            # Continuamos usando las coords originales para el resto de lógicas
            x, y = x_o, y_o

        # ------------------------------------------------------------------
        # 1) Eliminar tracker manual si clic dentro de su bbox
        # ------------------------------------------------------------------
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (trk, box) in enumerate(self.manual_trackers):
                if box.contains(x, y):
                    logging.info(f"Eliminando tracker manual: {box}")
                    self.manual_trackers.pop(i)
                    return

        # 2) Segmentación activa → crear tracker KCF sobre ROI segmentada -----
        if self.segmentation_active and event == cv2.EVENT_LBUTTONDOWN and self.frame_current is not None:
            lo = cv2.getTrackbarPos("loDiff", "Parametros")
            up = cv2.getTrackbarPos("upDiff", "Parametros")
            ksz = cv2.getTrackbarPos("Kernel", "Parametros")

            if self.segmentation_method in (1, 2):
                use_morph = self.segmentation_method == 2
                mask = segment_floodfill(self.frame_current, x, y, lo, up, ksz, use_morph)
            else:
                mask = segment_color_threshold(self.frame_current, ksz)

            bbox_t = get_bbox_from_mask(mask, (x, y))  # -> tuple or None
            if bbox_t:
                bbox = BBox(*bbox_t)
                tracker = initialize_tracker(self.frame_current, bbox)
                self.manual_trackers.append((tracker, bbox))
                logging.info(f"Tracker segmentación añadido: {bbox}")
            return  # no delegamos a focus si hay segmentación

        # 3) Resto de clics → MouseSelector (focus) ---------------------------
        self.mouse_selector.mouse_callback(event, x, y, flags, self.current_boxes)

    # ====================================================================== #
    #                               BUCLE MAIN                               #
    # ====================================================================== #
    def run(self):
        while True:
            frame_t0 = perf_counter()
            ok, frame = self.cap.read()
            if not ok:
                logging.info("Fin del vídeo / cámara.")
                break

            self.frame_count += 1
            self.frame_current = frame            # se usa en segmentación
            self.current_boxes.clear()
            display = frame.copy()
            tracked = None
            # --------------------------- DETECCIÓN --------------------------

            if not self.tracking_mode:
                # --- medir YOLO ---
                y0 = perf_counter()
                dets = detectar_objetos(self.model, frame,
                                        classes_to_detect=self.classes_to_detect)
                y1 = perf_counter()
                self._yolo_ms = (y1 - y0) * 1000.0

                tracked = None
                self._bt_ms = 0.0

                if dets and hasattr(dets, "xyxy"):
                    # --- medir ByteTrack ---
                    bt0 = perf_counter()
                    tracked = actualizar_seguimiento(self.bt_tracker, dets)
                    bt1 = perf_counter()
                    self._bt_ms = (bt1 - bt0) * 1000.0

                    xyxy = tracked.xyxy.astype(int)
                    cls_ids = getattr(tracked, "class_id", None)
                    ids     = getattr(tracked, "tracker_id", None)

                    for i, (x1, y1, x2, y2) in enumerate(xyxy):
                        box = BBox.from_xyxy(x1, y1, x2, y2)
                        self.current_boxes.append(box)
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        if cls_ids is not None:
                            cls_id = int(cls_ids[i])
                            label = self.class_names.get(cls_id, str(cls_id))
                            cv2.putText(display, label, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        if ids is not None:
                            tid = int(ids[i]) if ids[i] is not None else -1
                            cv2.putText(display, f"ID {tid}", (x1, y1 - 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                else:
                    tracked = None  # explícito

                self._kcf_ms = 0.0
                cv2.putText(display, "Modo Deteccion (ByteTrack)", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --------------------------- TRACKING (KCF) ---------------------
            else:
                # Re-detección cada N frames
                if (self.redetect_interval > 0
                        and self.frame_count % self.redetect_interval == 0
                        and self.kcf):
                    dets = detectar_objetos(self.model, frame,
                                             classes_to_detect=self.classes_to_detect)
                    boxes_new = []
                    if dets and hasattr(dets, "xyxy"):
                        for x1, y1, x2, y2 in dets.xyxy.astype(int):
                            boxes_new.append(BBox.from_xyxy(x1, y1, x2, y2))
                    if boxes_new:
                        self.kcf = MultiKCFTracker(frame, [ (b.x, b.y, b.w, b.h) for b in boxes_new ])
                        logging.info("Re-detección lanzada (KCF refrescado).")

                # Actualizar KCF
                if self.kcf:
                    k0 = perf_counter()
                    boxes_kcf = self.kcf.update(frame)                                                                               
                    for x, y, w, h in boxes_kcf:
                        box = BBox(x, y, w, h)
                        self.current_boxes.append(box)
                        cv2.rectangle(display, (x, y), (x + w, y + h),
                                    (0, 0, 255), 2, cv2.LINE_AA)

                    k1 = perf_counter()
                    self._kcf_ms = (k1 - k0) * 1000.0
                    self._yolo_ms = 0.0
                    self._bt_ms   = 0.0 
                cv2.putText(display, "Modo Tracking (KCF)", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)




            # ----------------------------- FOCUS ----------------------------
            target = None
            if self.focus_mode and self.mouse_selector.selected_object:
                target = self.mouse_selector.selected_object
                focus_box = self.mouse_selector.selected_object
                best = max(self.current_boxes, default=None,
                        key=lambda b: focus_box.iou(b))

                if best:
                    self.mouse_selector.selected_object = best
                    margin = 50
                    x1 = max(0, best.x - margin)
                    y1 = max(0, best.y - margin)
                    x2 = min(frame.shape[1], best.x + best.w + margin)
                    y2 = min(frame.shape[0], best.y + best.h + margin)

                    roi = frame[y1:y2, x1:x2]

                    if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                        # No se puede ampliar; sal de focus “suave”
                        self.focus_mode = False
                        self._focus_roi = None
                        self._focus_scale = None
                        logging.info("Focus cancelado (ROI vacio).")
                    else:
                        # factores de escala...
                        sx = frame.shape[1] / roi.shape[1]
                        sy = frame.shape[0] / roi.shape[0]

                    display = cv2.resize(roi, (frame.shape[1], frame.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)

                    # guardamos para mapear clics inversamente
                    self._focus_roi = (x1, y1, x2, y2)
                    self._focus_scale = (sx, sy)

                    # dibujamos la bbox (-margen) sobre la vista ampliada
                    bx1 = int((best.x - x1) * sx)
                    by1 = int((best.y - y1) * sy)
                    bx2 = int((best.x + best.w - x1) * sx)
                    by2 = int((best.y + best.h - y1) * sy)
                    cv2.rectangle(display, (bx1, by1), (bx2, by2),
                                (0, 255, 255), 2)

            elif self.current_boxes:
                # sin focus → target por tamaño
                target = max(self.current_boxes, key=lambda b: b.w * b.h)
            else:
                # si no estamos en focus, anulamos transformaciones guardadas
                self._focus_roi = None
                self._focus_scale = None

            # ---------- ángulos (θ), velocidad (ω) y mando gimbal ----------
            theta = None
            omega = (0.0, 0.0)
            if target:
                cx, cy = self.opt.center_px(target.x, target.y, target.w, target.h)
                thx, thy = self.opt.angles(cx, cy)   # radianes
                theta = (thx, thy)

                # velocidad angular estimada
                t_now = time.time()
                dt_ang = max(1e-3, t_now - getattr(self, "_last_t", t_now))
                if self._last_theta is not None:
                    omega = ((thx - self._last_theta[0]) / dt_ang,
                            (thy - self._last_theta[1]) / dt_ang)
                self._last_theta = theta
                self._last_t = t_now
            
            
            # --------------- Trackers manuales (ROI / segmentación) ---------
            updated_manual: List[Tuple[compat.OpenCVTracker, BBox]] = []
            for trk, box in self.manual_trackers:
                ok, new_xywh = trk.update(frame)
                if ok:
                    nb = BBox(*map(int, new_xywh))
                    cv2.rectangle(display, (nb.x, nb.y),
                                  (nb.x + nb.w, nb.y + nb.h), (255, 0, 0), 2)
                    updated_manual.append((trk, nb))
                else:
                    cv2.putText(display, "Tracker manual perdido", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            self.manual_trackers = updated_manual

            # --------------------------- Overlay UI -------------------------            
            if self.segmentation_active:
                cv2.putText(display,
                            f"Segmentacion ACTIVA (metodo {self.segmentation_method})",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)

            # --- Sincronizar estado del HUD con la app ---
            self.overlay.state.mode = "tracking" if self.tracking_mode else "detection"
            self.overlay.state.model = "militar" if self.modelo_actual == "militar" else "COCO"
            self.overlay.state.focus = self.focus_mode
            self.overlay.state.seg_assisted = self.segmentation_active
            self.overlay.state.seg_method = int(self.segmentation_method)

            # --- Dibujar HUD interactivo sobre el frame ---
            lat_ms = (perf_counter() - frame_t0) * 1000.0
            self._lat_hist.append(lat_ms)
            avg_ms = sum(self._lat_hist) / max(1, len(self._lat_hist))
            fps = 1000.0 / max(1e-6, avg_ms)

            H, W = frame.shape[0], frame.shape[1]
            num_obj = len(self.current_boxes)

            info_kv = {
                "Uptime (s)": f"{perf_counter() - self._t0:.1f}",
                "FPS (media)": f"{fps:.1f}",
                "Lat total (ms)": f"{avg_ms:.1f}",
                "YOLO (ms)": f"{self._yolo_ms:.1f}",
                "ByteTrack (ms)": f"{self._bt_ms:.1f}",
                "KCF (ms)": f"{self._kcf_ms:.1f}",
                "Resolucion": f"{W}x{H}",
                "Modelo": self.modelo_actual.upper(),
                "#Objetos": str(num_obj),
            }


            # -------- TABLA DE DETECCIONES --------
            rows = []

            def add_row(label_tid, label_cls, x1, y1, x2, y2):
                # bbox y centro en píxeles
                w_, h_ = (x2 - x1), (y2 - y1)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # columna "Local" según modo
                if self.coord_mode == 0:        # PIXEL
                    local = f"C=({cx},{cy})"
                elif self.coord_mode == 1:      # NORM
                    u, v = self.opt.norm_uv(cx, cy)
                    local = f"u,v=({u:+.3f},{v:+.3f})"
                elif self.coord_mode == 2:      # ANGLE
                    thx, thy = self.opt.angles(cx, cy)
                    local = f"θ=({math.degrees(thx):+.2f},{math.degrees(thy):+.2f})"
                else:                           # PLANE
                    XY = pixel_to_plane(self.H_plane, cx, cy) if self.H_plane is not None else None
                    local = "XY=NA" if XY is None else f"XY=({XY[0]:+.2f},{XY[1]:+.2f})"

                # fila con 5 columnas: (ID, Clase, BBox, Centro, Local)
                rows.append((
                    label_tid,
                    label_cls,
                    (int(x1), int(y1), int(w_), int(h_)),
                    (int(cx), int(cy)),
                    local
                ))

            # --- Detección (ByteTrack) con IDs ---
            if tracked is not None and hasattr(tracked, "xyxy"):
                xyxy = tracked.xyxy.astype(int)
                ids = getattr(tracked, "tracker_id", None)
                cls_ids = getattr(tracked, "class_id", None)

                for i, (x1, y1, x2, y2) in enumerate(xyxy):
                    # ID y clase como tipos nativos (evita np.int32)
                    tid = int(ids[i]) if ids is not None and ids[i] is not None else "-"
                    if cls_ids is not None:
                        cls_id = int(cls_ids[i])
                        # self.class_names puede ser dict o lista
                        if isinstance(self.class_names, dict):
                            cls_label = self.class_names.get(cls_id, str(cls_id))
                        else:
                            cls_label = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id)
                    else:
                        cls_label = "?"

                    add_row(tid, cls_label, x1, y1, x2, y2)

            # --- KCF (sin IDs/clase) ---
            elif self.current_boxes:
                for b in self.current_boxes:
                    add_row("-", "KCF", b.x, b.y, b.x + b.w, b.y + b.h)

            # --- Fallback: si rows está vacío, muestra una fila informativa ---
            if not rows:
                rows.append(("-", "-", "-", "-", "sin objetos"))


            if theta:
                info_kv["Theta (deg)"] = f"({math.degrees(theta[0]):+.2f}, {math.degrees(theta[1]):+.2f})"
                info_kv["Omega (deg/s)"] = f"({math.degrees(omega[0]):+.1f}, {math.degrees(omega[1]):+.1f})"
                info_kv["Centro optico"] = f"({int(self.opt.cx_eff)}, {int(self.opt.cy_eff)})"

            self.overlay.set_info(info_kv, rows)

            # ---- Datos KV para la pestaña Coordenadas ----
            mode_label = ["PIXEL","NORM","ANGLE","PLANE"][self.coord_mode]
            coord_kv = {
                "Centro optico": f"({int(self.opt.cx_eff)}, {int(self.opt.cy_eff)})",
                "Modo": mode_label
            }
            # Muestra FOV o K según lo disponible
            if self.opt.fx and self.opt.fy and self.opt.cx is not None and self.opt.cy is not None:
                coord_kv["K (fx,fy)"] = f"({float(self.opt.fx):.1f}, {float(self.opt.fy):.1f})"
            else:
                coord_kv["FOV (deg)"] = f"({float(self.opt.fovx_deg):.1f}, {float(self.opt.fovy_deg):.1f})"

            # sincroniza estado y pasa filas al panel Coordenadas
            self.overlay.state.coord_mode_label = mode_label  # (si lo muestras en la barra)
            self.overlay.set_coord(rows, mode_label, coord_kv)

            # (opcional para depurar)
            # logging.debug(f"[COORD] filas enviadas: {len(rows)}")




            display = self.overlay.render(display)


            cv2.imshow("frame", display)
            # ----------------------------------------------------------------
            #                       GESTIÓN DE TECLAS (físicas + HUD)
            # ----------------------------------------------------------------
            key_raw = cv2.waitKeyEx(1)  # -1 si no hay tecla
            vkey = self._queued_virtual_key
            self._queued_virtual_key = None

            key_chr = ""
            if key_raw != -1 and (key_raw & 0xFF) < 128:
                key_chr = chr(key_raw & 0xFF).lower()

            # Comando final: prioriza la tecla virtual clicada en el HUD
            cmd = vkey or key_chr
            if not cmd and key_raw not in (27,):   # nada que hacer
                continue

            # Marca visual en el HUD
            if cmd:
                self.overlay.mark_pressed(cmd)

            # Salida con 'q' o ESC
            if cmd == "q" or key_raw == 27:
                break

            # ----- modos detección / tracking --------------------------------
            elif cmd == "t" and not self.tracking_mode:
                init_boxes = ([self.mouse_selector.selected_object]
                            if self.mouse_selector.selected_object
                            else self.current_boxes)
                if init_boxes:
                    self.kcf = MultiKCFTracker(
                        frame, [(b.x, b.y, b.w, b.h) for b in init_boxes]
                    )
                    self.tracking_mode = True
                    self.frame_count = 0
                    logging.info(f"Tracking iniciado con {len(init_boxes)} bbox(es).")
                else:
                    logging.info("No hay objetos para trackear.")

            elif cmd == "d" and self.tracking_mode:
                self.tracking_mode = False
                self.kcf = None
                self.mouse_selector.selected_object = None
                logging.info("Modo detección re-activado.")

            # ----- focus ------------------------------------------------------
            elif cmd == "f":
                self.focus_mode = not self.focus_mode
                if not self.focus_mode:
                    self.mouse_selector.selected_object = None

            # ----- segmentación ----------------------------------------------
            elif cmd == "c":
                self.segmentation_active = not self.segmentation_active
                if self.segmentation_active:
                    cv2.namedWindow("Parametros")
                    cv2.createTrackbar("loDiff", "Parametros", 20, 100, lambda *_: None)
                    cv2.createTrackbar("upDiff", "Parametros", 20, 100, lambda *_: None)
                    cv2.createTrackbar("Kernel", "Parametros", 5, 50, lambda *_: None)
                else:
                    cv2.destroyWindow("Parametros")

            elif self.segmentation_active and cmd in ("1", "2", "3"):
                self.segmentation_method = int(cmd)
                logging.info(f"Método de segmentación → {self.segmentation_method}")

            # ----- alternar modelo -------------------------------------------
            elif cmd == "m":
                self.modelo_actual = "coco" if self.modelo_actual == "militar" else "militar"
                self.model = inicializar_modelo(self.modelo_actual)
                self.class_names = self.model.names
                self.bt_tracker = inicializar_seguimiento()  # limpiar IDs
                logging.info(f"Modelo cambiado a: {self.modelo_actual.upper()}")

            # ----- ROI manual -------------------------------------------------
            elif cmd == "p":
                snap = self.frame_current.copy()
                bbox_t = cv2.selectROI("Selecciona ROI", snap, False, False)
                cv2.destroyWindow("Selecciona ROI")
                if bbox_t and bbox_t != (0, 0, 0, 0):
                    bbox = BBox(*map(int, bbox_t))
                    trk = initialize_tracker(self.frame_current, bbox)
                    self.manual_trackers.append((trk, bbox))
                    logging.info(f"Tracker ROI manual añadido: {bbox}")

            # ----- HUD on/off -------------------------------------------------
            elif cmd == "h":
                self.overlay.toggle_menu()

            elif cmd == "i":
                self.overlay.toggle_info()


            elif cmd == "o":
                # togglear panel Coordenadas (independiente de Info)
                self.overlay.toggle_coord()


            elif cmd in ("[", "]"):
                delta = 1 if cmd == "]" else -1
                nm = (self.coord_mode + delta) % 4
                if nm == 3 and self.H_plane is None:   # si no hay homografía, salta PLANE
                    nm = (nm + delta) % 4
                self.coord_mode = nm
                logging.info(f"Coord mode → {['PIXEL','NORM','ANGLE','PLANE'][self.coord_mode]}")

        # ---------------------- Fin del bucle -----------------------------
        self.cap.release()
        cv2.destroyAllWindows()


# ========================================================================== #
#                                MAIN                                        #
# ========================================================================== #
if __name__ == "__main__":
    # --- Ejemplos de configuración ----------------------------------------
    # Detectar todas las clases (0-79 COCO)
    
    #clases = list(range(80))

    clases = [0]

    # Para filtrar: classes_to_detect = [0, 2]  # persona y coche, por ejemplo
    #video_source = 0  # "videos/race.mp4" o ruta a cámara
    video_source = "videos/race.mp4"
    app = DetectorHybridTracker(video_source, classes_to_detect=clases)
    t0 = time.time()
    app.run()
    logging.info("Tiempo total: %.1f s", time.time() - t0)
