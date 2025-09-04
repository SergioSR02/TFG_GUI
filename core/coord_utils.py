import json, math, numpy as np, cv2
from dataclasses import dataclass

@dataclass
class Optics:
    W: int; H: int
    # A: por FOV (si no calibras)
    fovx_deg: float | None = None
    fovy_deg: float | None = None
    # B: por intrínsecas reales (si calibras)
    fx: float | None = None; fy: float | None = None
    cx: float | None = None; cy: float | None = None

    @property
    def cx_eff(self): return self.cx if self.cx is not None else self.W * 0.5
    @property
    def cy_eff(self): return self.cy if self.cy is not None else self.H * 0.5

    def center_px(self, x,y,w,h):
        return int(x + w/2), int(y + h/2)

    def norm_uv(self, cx, cy):
        # centrado y normalizado [-1,1] con origen en el centro óptico
        u =  2.0 * (cx - self.cx_eff) / max(1.0, self.W)
        v =  2.0 * (cy - self.cy_eff) / max(1.0, self.H)
        return (u, v)

    def angles(self, cx, cy):
        # θx, θy en rad: usa K si la tienes; si no, deriva fx/fy del FOV y aplica atan
        if self.fx and self.fy and self.cx is not None and self.cy is not None:
            xn = (cx - self.cx)/self.fx
            yn = (cy - self.cy)/self.fy
        else:
            if not (self.fovx_deg and self.fovy_deg):
                raise ValueError("Falta FOV o K para calcular ángulos.")
            fx = (self.W*0.5)/math.tan(math.radians(self.fovx_deg*0.5))
            fy = (self.H*0.5)/math.tan(math.radians(self.fovy_deg*0.5))
            xn = (cx - self.cx_eff)/fx
            yn = (cy - self.cy_eff)/fy
        return math.atan(xn), math.atan(yn)

def load_homography(path:str) -> np.ndarray | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        H = np.array(data["H"], dtype=float).reshape(3,3)
        return H
    except Exception:
        return None

def pixel_to_plane(H: np.ndarray, cx:int, cy:int) -> tuple[float,float] | None:
    if H is None: return None
    p = np.array([cx, cy, 1.0], dtype=float)
    q = H @ p
    if abs(q[2]) < 1e-9: return None
    X, Y = q[0]/q[2], q[1]/q[2]
    return (float(X), float(Y))
