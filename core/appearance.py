import cv2, numpy as np

def hsv_fingerprint(img_bgr, x1,y1,x2,y2, bins=(16,8,8)):
    # recorta ROI y calcula histograma HSV normalizado
    roi = img_bgr[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    h = cv2.calcHist([H],[0],None,[bins[0]],[0,180])
    s = cv2.calcHist([S],[0],None,[bins[1]],[0,256])
    v = cv2.calcHist([V],[0],None,[bins[2]],[0,256])
    feat = np.concatenate([h.flatten(), s.flatten(), v.flatten()]).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-6)
    return feat  # vector (32,) por ejemplo
