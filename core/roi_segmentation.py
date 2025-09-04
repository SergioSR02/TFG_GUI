import cv2
import numpy as np
from typing import Tuple, Optional

def segment_floodfill(frame: np.ndarray, x: int, y: int, loDiff: int, upDiff: int, 
                      kernel_size: int = 5, use_morph: bool = False) -> np.ndarray:
    """
    Realiza la segmentación de la imagen mediante FloodFill a partir de un clic.

    Args:
        frame: Imagen de entrada.
        x, y: Coordenadas del clic.
        loDiff: Diferencia inferior para floodFill.
        upDiff: Diferencia superior para floodFill.
        kernel_size: Tamaño del kernel para operaciones morfológicas.
        use_morph: Si True, se aplica dilatación para suavizar la máscara.

    Returns:
        La máscara binaria resultante.
    """
    mask = np.zeros((frame.shape[0] + 2, frame.shape[1] + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    cv2.floodFill(frame.copy(), mask, (x, y), (0, 0, 255), (loDiff,)*3, (upDiff,)*3, flags)
    mask = mask[1:-1, 1:-1]
    if use_morph:
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def segment_color_threshold(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Realiza la segmentación basada en umbral en el espacio HSV.

    Args:
        frame: Imagen de entrada.
        kernel_size: Tamaño del kernel para operaciones morfológicas.

    Returns:
        La máscara binaria resultante.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_bg = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.bitwise_not(mask_bg)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def get_bbox_from_mask(mask: np.ndarray, click_point: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
    """
    Calcula el bounding box a partir de la máscara segmentada.

    Args:
        mask: La máscara binaria.
        click_point: Opcionalmente, las coordenadas del clic para escoger el contorno adecuado.

    Returns:
        El bounding box (x, y, w, h) o None si no se detecta ningún contorno.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    if click_point:
        for cnt in contours:
            if cv2.pointPolygonTest(cnt, click_point, False) >= 0:
                return cv2.boundingRect(cnt)
    return cv2.boundingRect(max(contours, key=cv2.contourArea))


def postprocess_mask(mask: np.ndarray, min_area: int = 150, close: int = 5, fill_holes: bool = True) -> np.ndarray:
    """Filtra ruido, cierra huecos y rellena agujeros opcionalmente."""
    m = (mask > 0).astype(np.uint8) * 255
    if fill_holes:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    if close and close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close | 1, close | 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    # quita componentes pequeñas
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep

def grabcut_from_rect(frame: np.ndarray, rect_xywh: Tuple[int,int,int,int], iters: int = 3) -> np.ndarray:
    """Segmenta con GrabCut inicializando con un rectángulo (ROI). Devuelve máscara 0/255."""
    (x, y, w, h) = rect_xywh
    H, W = frame.shape[:2]
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
    rect = (x, y, w, h)

    mask = np.zeros((H, W), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
    # conviértela a 0/255
    result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return result