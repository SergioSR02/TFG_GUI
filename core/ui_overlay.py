# ui_overlay.py
# ------------------------------------------------------------
# Menú/HUD interactivo de atajos de teclado para una app OpenCV
# - Mostrar/ocultar con toggle (propiedad show_menu)
# - Tabla con columnas: Tecla | Acción | Observaciones
# - "Keycaps" clicables: devuelven una tecla virtual en handle_mouse()
# - Estados dinámicos: modo, modelo, segmentación, focus, método 1/2/3
# - Sin dependencias extra (solo cv2 y numpy)
#
# Uso rápido (en tu bucle principal):
#   overlay = UIOverlay()
#   cv2.setMouseCallback(window_name, overlay.handle_mouse)
#   ...
#   frame = overlay.render(frame)     # cuando quieras pintar el HUD
#   vkey = overlay.last_clicked_key() # si el usuario hizo click
#   if vkey is not None:
#       # Trátalo como si fuese una tecla real (d,t,f,c,1,2,3,p,m,h,q)
#       ...
#   # Mantén actualizados los estados:
#   overlay.state.mode = "detection" / "tracking"
#   overlay.state.model = "COCO" / "militar"
#   overlay.state.focus = True/False
#   overlay.state.seg_assisted = True/False
#   overlay.state.seg_method = 1/2/3
#
# ------------------------------------------------------------

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
import time


# ---------------------------- utilidades de dibujo ----------------------------

def _rounded_rect(img, pt1, pt2, radius, color, thickness=-1):
    """Dibuja un rectángulo redondeado simple."""
    (x1, y1), (x2, y2) = pt1, pt2
    w, h = x2 - x1, y2 - y1
    r = min(radius, w // 2, h // 2)

    if thickness < 0:
        # relleno
        overlay = img.copy()
        cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
            cv2.ellipse(overlay, (cx, cy), (r, r), 0, 0, 360, color, -1)
        cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
    else:
        # borde (menos usado aquí)
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
        for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
            cv2.ellipse(img, (cx, cy), (r, r), 0, 0, 360, color, thickness)


def _put_text(img, text, org, scale=0.6, color=(240, 240, 240), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_keycap(img, x, y, w, h, label, active=False, hot=False):
    """Dibuja una tecla (keycap). Devuelve su bounding box."""
    bg = (50, 50, 55) if not active else (70, 100, 70)          # activo = verdoso
    if hot:
        bg = (90, 90, 140)                                      # clic reciente = azulado
    _rounded_rect(img, (x, y), (x + w, y + h), radius=6, color=bg, thickness=-1)
    # borde sutil
    cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 20), 1, cv2.LINE_AA)
    # texto centrado
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    tx, ty = x + (w - tw) // 2, y + (h + th) // 2 - 2
    _put_text(img, label, (tx, ty), 0.6, (240, 240, 240), 1)
    return (x, y, x + w, y + h)


# ---------------------------- estado del HUD ----------------------------------

@dataclass
class UIState:
    mode: str = "detection"      # "detection" | "tracking"
    model: str = "COCO"          # "COCO" | "militar"
    focus: bool = False
    seg_assisted: bool = False
    seg_method: int = 1          # 1 | 2 | 3


# ---------------------------- overlay principal -------------------------------

class UIOverlay:
    def __init__(self, scale: float = 1.0, anchor: str = "right"):
        """
        scale: factor de tamaño del panel
        anchor: "right" o "left" (dónde se pega el panel)
        """
        self.scale = scale
        self.anchor = anchor
        self.state = UIState()
        self.show_menu = True
        self._last_click_key: Optional[str] = None
        self._hotkeys_boxes: Dict[str, Tuple[int, int, int, int]] = {}
        self._recent_pressed: Dict[str, float] = {}  # para parpadear
        self._panel_cache_size = None  # para recomputar layout si cambia la ventana
        # Posición/arrastre y geometría memorizada
        self.pos = None                   # (x0, y0) si el usuario lo ha movido
        self.dragging = False
        self._drag_offset = (0, 0)        # delta desde la esquina del panel
        self._panel_size = (0, 0)         # (w, h) del último render
        self._panel_rect = None           # (x1,y1,x2,y2) del panel
        self._header_rect = None          # barra de título para arrastrar
        self._close_rect = None           # caja del botón [X]
        self._last_frame_shape = (0, 0)   # (h, w) del último frame

        self.show_info = False
        self.info_kv = {}       # dict: "clave" -> "valor"
        self.info_rows = []     # lista de tuplas: (ID, clase, bbox, centro, (u,v))
        self._prev_show_menu = True  

        self.show_coord = False
        self.coord_rows = []              
        self.coord_mode_label = "PIXEL"   
        self.coord_kv = {}                

        self._coord_rect = [40, 40, 560, 280]  
        self._coord_drag = False
        self._coord_drag_off = (0, 0)

        self.rows = [
            # MODOS
            dict(key="d",
                action="Modo deteccion (YOLOv8 + ByteTrack)",
                obs="Cajas verdes + etiqueta de clase; IDs por ByteTrack"),
            dict(key="t",
                action="Modo seguimiento local (Multi-KCF)",
                obs="Requiere cajas iniciales; re-deteccion periodica"),
            dict(key="f",
                action="Focus / zoom",
                obs="Zoom sobre bbox seleccionada; clic dentro para salir"),

            # SEGMENTACION
            dict(key="c",
                action="Segmentacion asistida (on/off)",
                obs="Abre ventana Parametros: loDiff, upDiff, Kernel"),
            dict(key="1/2/3",
                action="Metodo de segmentacion",
                obs="1: FloodFill; 2: FloodFill+Morph; 3: Umbral HSV"),

            # ROI MANUAL
            dict(key="p",
                action="ROI manual",
                obs="Selecciona bbox; crea tracker KCF; clic en caja para borrar"),

            # MODELOS
            dict(key="m",
                action="Alternar modelo (COCO / militar)",
                obs="Recarga modelo y reinicia ByteTrack (IDs cambian)"),

            # PANELES
            dict(key="h",
                action="Mostrar/ocultar menu de ayuda",
                obs="Panel arrastrable; teclas clicables"),
            dict(key="i",
                action="Informacion del sistema",
                obs="Uptime/FPS/latencias + tabla de objetos; cerrar con i o [X]"),
            dict(key="o",
                action="Panel Coordenadas",
                obs="Local (PIXEL/NORM/ANGLE/PLANE); mover con raton; cerrar con o"),
            dict(key="[",
                action="Coord: modo anterior",
                obs="Cicla PIXEL/NORM/ANGLE/PLANE (si hay homografia)"),
            dict(key="]",
                action="Coord: modo siguiente",
                obs="Cicla PIXEL/NORM/ANGLE/PLANE (si hay homografia)"),

            # SALIR
            dict(key="q/ESC",
                action="Salir",
                obs="Cierra ventana principal y libera recursos"),
        ]

    # --------------------- API pública para integrar ---------------------

    def toggle(self):
        self.show_menu = not self.show_menu

    def toggle_menu(self):
        """Si Info está activa, sal de Info y muestra el HUD; si no, toggle del HUD."""
        if self.show_info:
            # salir de Info y mostrar menú
            self.show_info = False
            self.show_menu = True
        else:
            self.show_menu = not self.show_menu

    def toggle_info(self):
        """Entra/sale de la vista de Información recordando el estado previo del HUD."""
        if not self.show_info:
            # vamos a Info: recuerda cómo estaba el menú
            self._prev_show_menu = self.show_menu
            self.show_info = True
        else:
            # salimos de Info: restaura cómo estaba el menú
            self.show_info = False
            self.show_menu = self._prev_show_menu



    def mark_pressed(self, key: str):
        """Llamar cuando se detecta una pulsación real para resaltarla en el HUD."""
        self._recent_pressed[key] = time.time()

    def last_clicked_key(self) -> Optional[str]:
        """Devuelve y limpia la última 'tecla virtual' clicada en el panel."""
        k, self._last_click_key = self._last_click_key, None
        return k

    def set_info(self, kv: dict, rows: list[tuple]):
        self.info_kv = kv or {}
        self.info_rows = rows or []


    def handle_mouse(self, event, x, y, flags, userdata=None):
        """Gestiona clic en keycaps, arrastre del panel y botón de cierre."""
        # if not self.show_menu:
        #     return
        x0, y0, w0, h0 = self._coord_rect
        if not (self.show_menu or self.show_info or self.show_coord):
            return

        # 1) Cerrar con el botón [X]

        if event == cv2.EVENT_LBUTTONDOWN and self._close_rect:
            x1, y1, x2, y2 = self._close_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                if self.show_info:
                    # cerrar Info y restaurar HUD según estaba
                    self.toggle_info()
                else:
                    # toggle del HUD
                    self.toggle_menu()
                return        

        # 2) Arrastre: pulsar sobre la barra de título
        if event == cv2.EVENT_LBUTTONDOWN and self._header_rect and self._panel_rect:
            hx1, hy1, hx2, hy2 = self._header_rect
            if hx1 <= x <= hx2 and hy1 <= y <= hy2:
                px1, py1, _, _ = self._panel_rect
                self.dragging = True
                self._drag_offset = (x - px1, y - py1)
                return

        if event == cv2.EVENT_MOUSEMOVE and self.dragging and self._panel_rect:
            # Nueva posición del panel (limitada a la ventana)
            h, w = self._last_frame_shape
            pw, ph = self._panel_size
            nx = max(0, min(w - pw, x - self._drag_offset[0]))
            ny = max(0, min(h - ph, y - self._drag_offset[1]))
            self.pos = (nx, ny)        # pasa a modo “posicion manual”
            self.anchor = None
            return

        if event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            return

        # 3) Keycaps clicables
        if event == cv2.EVENT_LBUTTONDOWN:
            for k, (x1, y1, x2, y2) in self._hotkeys_boxes.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._last_click_key = k
                    self._recent_pressed[k] = time.time()
                    break

        if event == cv2.EVENT_LBUTTONDOWN:
            # solo drag en la barra superior (sin botón X)
            if y0 <= y <= y0 + 32 and x0 <= x <= x0 + w0:
                self._coord_drag = True
                self._coord_drag_off = (x - x0, y - y0)
                return



        # if event == cv2.EVENT_LBUTTONDOWN:
        #     # caja [X] (20x20) en esquina superior derecha
        #     if x0 + w0 - 28 <= x <= x0 + w0 - 8 and y0 + 8 <= y <= y0 + 28:
        #         self.show_coord = False
        #         return
        #     # drag si clicas en la barra superior (alto 32 px)
        #     if y0 <= y <= y0 + 32 and x0 <= x <= x0 + w0:
        #         self._coord_drag = True
        #         self._coord_drag_off = (x - x0, y - y0)
        #         return
        elif event == cv2.EVENT_MOUSEMOVE and self._coord_drag:
            nx = x - self._coord_drag_off[0]
            ny = y - self._coord_drag_off[1]
            self._coord_rect[0] = max(0, nx)
            self._coord_rect[1] = max(0, ny)
            return
        elif event == cv2.EVENT_LBUTTONUP and self._coord_drag:
            self._coord_drag = False
            return

        
    def set_coord(self, rows, mode_label: str, kv: dict | None = None):
        """Actualiza contenido del panel Coordenadas."""
        self.coord_rows = rows or []
        self.coord_mode_label = mode_label
        self.coord_kv = kv or {}

    def toggle_coord(self):
        self.show_coord = not self.show_coord


    def pointer_inside(self, x, y) -> bool:
        """True si el puntero está sobre cualquiera de los paneles (HUD/Info o Coordenadas)."""
        # Panel HUD/Info
        if self._panel_rect:
            x1, y1, x2, y2 = self._panel_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        # Panel Coordenadas
        if self.show_coord and self._coord_rect:
            rx, ry, rw, rh = self._coord_rect
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return True
        return False

    # --------------------------- render ----------------------------------

    def render(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self._last_frame_shape = (h, w)

        # === HUD (solo si show_menu) ===
        
        
        if self.show_menu:
            frame = self._render_menu(frame)        

        # === INFO (si está activa) ===
        if self.show_info:
            frame = self._render_info(frame)

        # === COORDENADAS (si está activa) ===
        if self.show_coord:
            frame = self._render_coord(frame)

        return frame


    # ------------------------- helpers internos -------------------------

    def _wrap_text(self, text: str, max_width: int, scale: float) -> List[str]:
        """Partir una cadena según ancho máximo aproximado."""
        words = text.split()
        lines, current = [], ""
        for w in words:
            trial = (current + " " + w).strip()
            (tw, _), _ = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            if tw <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = w
        if current:
            lines.append(current)
        return lines


    def _render_info(self, frame):
        import cv2
        H, W = frame.shape[:2]

        # ===== layout y estilo =====
        pad = 12
        header_h = 38           # barra superior un pelín más alta
        kv_line_h = 22          # altura de línea para el bloque KV
        row_h = 24              # altura de cada fila de la tabla
        gap = 18
        txt_scale = 0.56
        txt_th = 1

        def tsize(s, scale_=txt_scale, th=txt_th):
            (tw, th_), _ = cv2.getTextSize(str(s), cv2.FONT_HERSHEY_SIMPLEX, scale_, th)
            return tw, th_

        # ===== datos =====
        # Bloque KV (diccionario con métricas), ya preparado por main.set_info()
        kv = self.info_kv or {}
        kv_keys_sorted = list(kv.keys())

        # Tabla de objetos: ID, Clase, BBox, Centro, Local
        rows = self.info_rows or []
        mode_label = getattr(self.state, "coord_mode_label", getattr(self, "coord_mode_label", "")) or ""
        headers = ["ID", "Clase", "BBox (x,y,w,h)", "Centro (px)", f"Local ({mode_label})"]

        # ===== cálculo de tamaño =====
        # alto del bloque KV
        kv_h = len(kv_keys_sorted) * kv_line_h if kv_keys_sorted else 0
        kv_h += (8 if kv_keys_sorted else 0)  # respiración inferior

        # anchuras de columna midiendo texto real
        col_w = [0]*5
        for i in range(5):
            col_w[i] = tsize(headers[i])[0]
        # incluye filas
        meas_rows = rows if rows else [[ "— sin objetos —", "", "", "", "" ]]
        for r in meas_rows:
            # normaliza a 5 columnas string
            vals = [str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4])]
            for i in range(5):
                col_w[i] = max(col_w[i], tsize(vals[i])[0])
        # padding lateral por columna
        col_w = [w + 10 for w in col_w]

        table_w = sum(col_w) + pad*2
        total_w = min(max(table_w, 560), W - 20)

        # alto de la tabla = cabecera + filas + padding
        rows_count = max(1, len(rows))
        table_h = row_h * (1 + rows_count) + 8

        total_h = header_h + pad + kv_h + table_h + pad

        # ===== posicionamiento =====
        x, y = self.pos if self.pos else (40, 40)
        if x + total_w > W: x = max(0, W - int(total_w) - 10)
        if y + total_h > H: y = max(0, H - int(total_h) - 10)

        self._panel_size = (int(total_w), int(total_h))

        # guarda rects para hit-test y drag
        self._panel_rect = (x, y, x + int(total_w), y + int(total_h))
        self._header_rect = (x, y, x + int(total_w), y + header_h)

        # ===== fondo y barra superior =====
        cv2.rectangle(frame, (x, y), (x + int(total_w), y + int(total_h)), (18, 18, 20), -1)
        cv2.rectangle(frame, (x, y), (x + int(total_w), y + header_h), (28, 28, 32), -1)

        # Título (mantenemos tu estilo: blanco)
        title = "Informacion    Sistema y objetos"
        _put_text(frame, title, (x + 12, y + 26), 0.8, (240, 240, 240), 2)

        # Botón [X] (se mantiene)
        cv2.rectangle(frame, (x + int(total_w) - 28, y + 8), (x + int(total_w) - 8, y + 28), (60, 60, 65), -1)
        _put_text(frame, "X", (x + int(total_w) - 23, y + 24), 0.6, (230, 230, 230), 1)
        self._close_rect = (x + int(total_w) - 28, y + 8, x + int(total_w) - 8, y + 28)

        cy = y + header_h + pad

        # ===== bloque KV =====
        for k in kv_keys_sorted:
            _put_text(frame, f"{k}: {kv[k]}", (x + pad, cy), 0.58, (230, 230, 230), 1)
            cy += kv_line_h

        if kv_keys_sorted:
            cy += 6  # separación extra antes de la tabla

        # ===== tabla =====
        # columna X iniciales
        col_x = [x + pad]
        for i in range(1, 5):
            col_x.append(col_x[i-1] + col_w[i-1])

        # margen extra debajo de la barra gris para que no se peguen las cabeceras
        cy += 6
        # línea separadora suave
        cv2.line(frame, (x + pad, cy - 4), (x + int(total_w) - pad, cy - 4), (70, 70, 80), 1, cv2.LINE_AA)

        # cabeceras (en BLANCO, no rojo)
        for i, htxt in enumerate(headers):
            _put_text(frame, htxt, (col_x[i], cy), 0.60, (245, 245, 245), 2)
        cy += row_h

        # filas
        if not rows:
            _put_text(frame, "— sin objetos —", (col_x[0], cy), 0.56, (180, 180, 180), 1)
            cy += row_h
        else:
            for r in rows:
                vals = [str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4])]
                for i, v in enumerate(vals):
                    _put_text(frame, v, (col_x[i], cy), 0.56, (210, 210, 210), 1)
                cy += row_h

        return frame





    # def _render_info(self, frame: np.ndarray) -> np.ndarray:
    #     # Panel del mismo estilo que el HUD, pero con dos bloques:
    #     # 1) key-values   2) tabla de detecciones
    #     h, w = frame.shape[:2]
    #     self._last_frame_shape = (h, w)
    #     scale = self.scale
    #     margin = int(12 * scale)

    #     panel_w = int(0.48 * w * scale)
    #     x0 = w - panel_w - margin if (self.pos is None and self.anchor == "right") else (self.pos[0] if self.pos else margin)
    #     y0 = (self.pos[1] if self.pos else margin)

    #     titlebar_h = int(56 * scale)
    #     col_l = x0 + 16
    #     line_h = int(18 * scale)

    #     # -------- medir alto ----------
    #     n_info = max(1, len(self.info_kv))
    #     n_rows = max(1, len(self.info_rows))
    #     table_header_h = int(26 * scale)
    #     table_row_h = int(22 * scale)
    #     body_top = y0 + titlebar_h + 8
    #     kv_h = n_info * line_h + 6
    #     table_h = table_header_h + n_rows * table_row_h + 10
    #     panel_h = min(body_top + kv_h + 10 + table_h - y0 + 12, h - y0 - margin)

    #     # guardar rects p/drag y [X]
    #     self._panel_size = (panel_w, panel_h)
    #     self._panel_rect = (x0, y0, x0 + panel_w, y0 + panel_h)
    #     self._header_rect = (x0, y0, x0 + panel_w, y0 + titlebar_h)

    #     # -------- fondo + titulo ----------
    #     panel = frame.copy()
    #     _rounded_rect(panel, (x0, y0), (x0 + panel_w, y0 + panel_h),
    #                 radius=12, color=(25, 25, 30), thickness=-1)
    #     cv2.addWeighted(panel, 0.85, frame, 0.15, 0, frame)
    #     _put_text(frame, "Informacion - Sistema y objetos",
    #             (x0 + 16, y0 + 28), 0.7, (255,255,255), 2)

    #     # botón [X]
    #     btn = int(22 * scale)
    #     bx2, by1 = x0 + panel_w - 12, y0 + 10
    #     bx1, by2 = bx2 - btn, by1 + btn
    #     _rounded_rect(frame, (bx1, by1), (bx2, by2), radius=6, color=(60,60,65), thickness=-1)
    #     cv2.rectangle(frame, (bx1, by1), (bx2, by2), (30,30,35), 1, cv2.LINE_AA)
    #     _put_text(frame, "X", (bx1 + 6, by2 - 6), 0.6, (230,230,230), 2)
    #     self._close_rect = (bx1, by1, bx2, by2)

    #     # -------- bloque KV ----------
    #     y = body_top
    #     for k, v in self.info_kv.items():
    #         _put_text(frame, f"{k}: {v}", (col_l, y), 0.6, (230,230,230), 1)
    #         y += line_h
    #     y += int(4 * scale)

    #     # -------- tabla ----------        
    #     headers = [
    #         "ID",
    #         "Clase",
    #         "BBox (x,y,w,h)",
    #         "Centro (px)",
    #         f"Local ({self.coord_mode_label})"
    #     ]        
        
    #     x_cols = [col_l,
    #             x0 + int(panel_w*0.18),
    #             x0 + int(panel_w*0.35),
    #             x0 + int(panel_w*0.65),
    #             x0 + int(panel_w*0.82)]
    #     for i,hdr in enumerate(headers):
    #         _put_text(frame, hdr, (x_cols[i], y), 0.56, (210,210,210), 2)
    #     cv2.line(frame, (x0 + 12, y + 6), (x0 + panel_w - 12, y + 6),
    #             (70,70,80), 1, cv2.LINE_AA)
    #     y += table_header_h - 4

    #     for rid, (tid, cls, bbox, center, local) in enumerate(self.info_rows):
    #         vals = [str(tid), str(cls), str(bbox), str(center), str(local)]
    #         for i, val in enumerate(vals):
    #             _put_text(frame, val, (x_cols[i], y), 0.54, (200,200,200), 1)
    #         y += table_row_h

    #     return frame


    def _render_coord(self, frame):
        import cv2, numpy as np
        H, W = frame.shape[:2]

        # --- layout base ---
        pad = 12
        header_h = 32
        row_h = 22
        col_gap = 18
        text_scale = 0.56
        text_th = 1

        # posición tentativa (se corrige si se sale por el borde)
        x, y, _, _ = self._coord_rect

        # ======= prepara datos =======
        headers = [
            "ID",
            "Clase",
            "BBox (x,y,w,h)",
            "Centro (px)",
            f"Local ({self.coord_mode_label})"
        ]

        # Normaliza filas a cadenas
        data_rows = []
        for (tid, cls, bbox, center, local) in (self.coord_rows or []):
            if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
                s_bbox = f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"
            else:
                s_bbox = str(bbox)
            if isinstance(center, (tuple, list)) and len(center) == 2:
                s_ctr = f"({center[0]}, {center[1]})"
            else:
                s_ctr = str(center)
            data_rows.append([str(tid), str(cls), s_bbox, s_ctr, str(local)])

        # Si no hay filas, ponemos una informativa
        empty_msg = "— sin objetos —"
        rows_to_measure = data_rows if data_rows else [[empty_msg, "", "", "", ""]]

        # ======= calcula anchos de columna midiendo texto =======
        def tsize(s, scale=text_scale, th=text_th):
            (tw, th_), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
            return tw, th_

        col_w = [0] * 5
        for i in range(5):
            col_w[i] = tsize(headers[i])[0]
            for r in rows_to_measure:
                col_w[i] = max(col_w[i], tsize(r[i])[0])
            col_w[i] += col_gap  # padding lateral

        table_w = sum(col_w) + pad * 2
        rows_count = max(1, len(data_rows))
        table_h = (1 + rows_count) * row_h + 8  # 1 = cabecera
        total_h = header_h + pad + table_h + pad
        total_w = min(table_w, W - 20)

        # corrige si se sale fuera de pantalla
        if x + total_w > W:
            x = max(0, W - int(total_w) - 10)
        if y + total_h > H:
            y = max(0, H - int(total_h) - 10)

        # guarda rect actualizado (para hit-test/drag)
        self._coord_rect = [x, y, int(total_w), int(total_h)]

        # ======= fondo y barra superior =======
        cv2.rectangle(frame, (x, y), (x + int(total_w), y + int(total_h)), (18, 18, 20), -1)
        cv2.rectangle(frame, (x, y), (x + int(total_w), y + header_h), (28, 28, 32), -1)

        title = f"Coordenadas  |  Modo: {self.coord_mode_label}   (o: cerrar, [: prev, ]: next)"
        _put_text(frame, title, (x + 10, y + 22), 0.6, (240, 240, 240), 1)

        # botón [X]
        # cv2.rectangle(frame, (x + int(total_w) - 28, y + 8), (x + int(total_w) - 8, y + 28), (60, 60, 65), -1)
        # _put_text(frame, "X", (x + int(total_w) - 23, y + 24), 0.6, (230, 230, 230), 1)

        cy = y + header_h + pad + 6   

        # ======= cabeceras (en rojo) =======
        col_x = [x + pad]
        for i in range(1, 5):
            col_x.append(col_x[i - 1] + col_w[i - 1])

        for i, htxt in enumerate(headers):
            _put_text(frame, htxt, (col_x[i], cy), 0.56, (0, 0, 255), 1)  # rojo puro (BGR)

        cy += row_h

        # ======= filas =======
        if not data_rows:
            _put_text(frame, empty_msg, (col_x[0], cy), 0.54, (170, 170, 170), 1)
            cy += row_h
        else:
            for r in data_rows:
                for i, v in enumerate(r):
                    _put_text(frame, v, (col_x[i], cy), 0.54, (210, 210, 210), 1)
                cy += row_h

        return frame


    def _render_menu(self, frame):
        import cv2
        H, W = frame.shape[:2]

        # ----- estilo y medidas -----
        scale = self.scale
        pad = 12
        header_h = 32
        row_base_h = int(24 * scale)
        txt_scale = 0.56
        txt_th = 1
        gap = 18
        keycap_w = int(48 * scale)
        keycap_h = int(28 * scale)
        key_gap  = int(6 * scale)

        def tsize(s, scale_=txt_scale, th=txt_th):
            (tw, th_), _ = cv2.getTextSize(str(s), cv2.FONT_HERSHEY_SIMPLEX, scale_, th)
            return tw, th_

        # ----- cabeceras -----
        headers = ["Tecla", "Accion", "Observaciones"]

        # ----- anchuras de columnas -----
        # Col 0: ancho máximo de keycaps por fila (cuenta "1/2/3" como 3)
        def keycap_total_w(key_label: str) -> int:
            if "/" in key_label:
                n = len(key_label.split("/"))
                return n * keycap_w + (n - 1) * key_gap
            return keycap_w

        col0_w = max(tsize(headers[0])[0], max(keycap_total_w(r["key"]) for r in self.rows)) + 8
        col1_w = max(tsize(headers[1])[0], max(tsize(r["action"])[0] for r in self.rows)) + 8

        # Anchura total tentativo (Observaciones ocupa el resto)
        min_obs_w = 260
        table_w = pad * 2 + col0_w + gap + col1_w + gap + min_obs_w
        total_w = min(max(table_w, 560), W - 20)

        col2_w = int(total_w - pad * 2 - col0_w - col1_w - 2 * gap)
        if col2_w < 160:  # nunca demasiado estrecha
            col2_w = 160
            total_w = pad * 2 + col0_w + col1_w + 2 * gap + col2_w

        # altura dinámica: calculamos alto por fila según el wrap de "Observaciones"
        def wrap_lines(text, max_w):
            line = ""
            lines = []
            for tok in str(text).split():
                test = (line + " " + tok).strip()
                if tsize(test)[0] <= max_w or not line:
                    line = test
                else:
                    lines.append(line)
                    line = tok
            if line:
                lines.append(line)
            return lines

        row_heights = []
        for r in self.rows:
            obs_lines = wrap_lines(r["obs"], col2_w)
            h_lines = len(obs_lines)
            row_h = max(row_base_h, h_lines * int(16 * scale) + 6)
            row_heights.append(row_h)

        table_h = header_h + pad + int(22 * scale) + sum(row_heights) + pad  # cabecera + filas + paddings
        total_h = table_h

        # --- posición final del panel (con límites) ---
        if self.pos is not None:
            x, y = self.pos
        elif self._panel_rect:
            x, y = self._panel_rect[0], self._panel_rect[1]
        else:
            x, y = 40, 40

        x = max(0, min(W - int(total_w), x))
        y = max(0, min(H - int(total_h), y))

        # guardar geometría para drag y barra
        self._panel_size = (int(total_w), int(total_h))
        self._panel_rect = (x, y, x + int(total_w), y + int(total_h))
        self._header_rect = (x, y, x + int(total_w), y + header_h)


        # if x + total_w > W: x = max(0, W - int(total_w) - 10)
        # if y + total_h > H: y = max(0, H - int(total_h) - 10)

        # # guarda rect para hit-test/drag/cerrar
        # self._panel_rect = (x, y, x + int(total_w), y + int(total_h))
        # self._header_rect = (x, y, x + int(total_w), y + header_h)

        # ----- fondo y barra superior -----
        cv2.rectangle(frame, (x, y), (x + int(total_w), y + int(total_h)), (18, 18, 20), -1)
        cv2.rectangle(frame, (x, y), (x + int(total_w), y + header_h), (28, 28, 32), -1)

        # Título limpio (como Coordenadas)
        title = "Atajos  |  (h: cerrar)"
        _put_text(frame, title, (x + 10, y + 22), 0.6, (240, 240, 240), 1)

        # Botón [X]
        cv2.rectangle(frame, (x + int(total_w) - 28, y + 8), (x + int(total_w) - 8, y + 28), (60, 60, 65), -1)
        _put_text(frame, "X", (x + int(total_w) - 23, y + 24), 0.6, (230, 230, 230), 1)
        self._close_rect = (x + int(total_w) - 28, y + 8, x + int(total_w) - 8, y + 28)

        # ----- posición de columnas -----
        col_x0 = x + pad
        col_x1 = col_x0 + col0_w + gap
        col_x2 = col_x1 + col1_w + gap
        cy = y + header_h + pad


        _put_text(frame, headers[0], (col_x0, cy), 0.56, (0, 0, 255), 1)
        _put_text(frame, headers[1], (col_x1, cy), 0.56, (0, 0, 255), 1)
        _put_text(frame, headers[2], (col_x2, cy), 0.56, (0, 0, 255), 1)

        sep_y = cy + int(6 * scale)
        cv2.line(frame, (x + pad, sep_y), (x + int(total_w) - pad, sep_y), (70, 70, 80), 1, cv2.LINE_AA)

        cy += int(28 * scale) 




        # ----- filas -----
        self._hotkeys_boxes.clear()
        now = time.time()

        for idx, r in enumerate(self.rows):
            row_h = row_heights[idx]

            # Tecla → keycaps (clicables)
            key = r["key"]
            xk = col_x0
            if "/" in key:
                labels = key.split("/")
            else:
                labels = [key]

            for lab in labels:
                is_active = (
                    (lab == "d" and self.state.mode == "detection") or
                    (lab == "t" and self.state.mode == "tracking") or
                    (lab == "f" and self.state.focus) or
                    (lab == "c" and self.state.seg_assisted) or
                    (lab == str(self.state.seg_method))
                )
                was_pressed = (lab in self._recent_pressed and (now - self._recent_pressed[lab] < 0.25))
                box = _draw_keycap(frame, xk, cy - keycap_h + 4, keycap_w, keycap_h,
                                lab.upper(), active=is_active, hot=was_pressed)
                self._hotkeys_boxes[lab] = box
                # caso especial: q/ESC
                if lab == "q/ESC" or lab == "q":
                    self._hotkeys_boxes["q"] = box
                xk += keycap_w + key_gap

            # Acción
            _put_text(frame, r["action"], (col_x1, cy), 0.56, (235, 235, 235), 1)

            # Observaciones (envuelve)
            obs_lines = wrap_lines(r["obs"], col2_w)
            oy = 0
            for line in obs_lines:
                _put_text(frame, line, (col_x2, cy + oy), 0.52, (200, 200, 200), 1)
                oy += int(16 * scale)

            cy += max(row_base_h, oy + 6)

        # Limpia marcas "hot"
        for k in list(self._recent_pressed.keys()):
            if now - self._recent_pressed[k] > 0.3:
                self._recent_pressed.pop(k, None)

        return frame

