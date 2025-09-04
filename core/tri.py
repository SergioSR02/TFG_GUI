import numpy as np, cv2

def triangulate_two(K1, R1, t1, K2, R2, t2, pt1_px, pt2_px, D1=None, D2=None):
    # pt*_px = (u,v) en píxeles
    # undistorsiona
    pts1 = np.array([[pt1_px]], dtype=np.float32)
    pts2 = np.array([[pt2_px]], dtype=np.float32)
    if D1 is None: D1 = np.zeros((1,5))
    if D2 is None: D2 = np.zeros((1,5))
    pts1_ud = cv2.undistortPoints(pts1, K1, D1)
    pts2_ud = cv2.undistortPoints(pts2, K2, D2)

    P1 = K1 @ np.hstack([R1, t1.reshape(3,1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3,1)])
    X_h = cv2.triangulatePoints(P1, P2, pts1_ud.reshape(2,1), pts2_ud.reshape(2,1))
    X = (X_h[:3] / X_h[3]).reshape(3)
    return X  