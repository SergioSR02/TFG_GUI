import cv2
import numpy as np

def clip_bbox(bbox, frame_shape):
    """
    Asegura que la bounding box esté dentro de los límites de la imagen.
    bbox: (x, y, w, h)
    frame_shape: (height, width)
    """
    x, y, w, h = bbox
    frame_h, frame_w = frame_shape
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)
    return (x, y, w, h)

class HybridImageTracker:
    """
    Tracker híbrido que utiliza:
      - Optical Flow (PyrLK) para estimar el desplazamiento.
      - Re-inicialización adaptativa de puntos en la ROI mediante ORB.
      - Sustracción de fondo (MOG2) para evaluar la validez del movimiento.
      - Filtro de Kalman para suavizar la estimación y reducir drift.
    """
    def __init__(self, initial_frame, initial_bboxes):
        self.prev_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        self.tracks = []  # Cada track es un diccionario para un objeto
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=True)
        self.orb = cv2.ORB_create(nfeatures=200)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Parámetros para re-inicialización adaptativa con ORB
        self.min_points_threshold = 5   # Si quedan menos puntos de seguimiento, se reinicializa ORB
        self.max_orb_interval = 5       # Reinicializa ORB cada 5 frames, aun si hay suficientes puntos

        for bbox in initial_bboxes:
            track = {}
            track['bbox'] = bbox
            x, y, w, h = bbox
            # Aseguramos que la ROI esté dentro de los límites del frame
            roi = self.prev_gray[y:y+h, x:x+w]
            pts = cv2.goodFeaturesToTrack(roi, maxCorners=50, qualityLevel=0.01, minDistance=5)
            if pts is not None:
                pts = pts + np.array([[x, y]], dtype=np.float32)
            else:
                pts = None
            track['points'] = pts
            track['orb_age'] = 0  # Contador de frames desde la última actualización ORB
            
            # Inicializar filtro de Kalman para suavizar la posición
            kf = cv2.KalmanFilter(4, 2)  # Estado: [x, y, dx, dy]; Medición: [x, y]
            kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], np.float32)
            kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
            cx, cy = x + w / 2, y + h / 2
            kf.statePre = np.array([cx, cy, 0, 0], np.float32)
            kf.statePost = np.array([cx, cy, 0, 0], np.float32)
            track['kalman'] = kf

            self.tracks.append(track)
    
    def update(self, current_frame):
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = current_gray.shape
        
        # Obtener la máscara de movimiento
        fg_mask = self.bg_subtractor.apply(current_frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        
        updated_bboxes = []
        for track in self.tracks:
            x, y, w, h = track['bbox']
            # Calcular optical flow si hay puntos disponibles
            if track['points'] is not None and len(track['points']) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, current_gray, track['points'], None, **self.lk_params)
                if p1 is not None and st is not None:
                    good_new = p1[st == 1]
                    good_old = track['points'][st == 1]
                    if len(good_new) > 0:
                        dx = np.median(good_new[:, 0] - good_old[:, 0])
                        dy = np.median(good_new[:, 1] - good_old[:, 1])
                    else:
                        dx, dy = 0, 0
                else:
                    dx, dy = 0, 0
            else:
                dx, dy = 0, 0
            
            # Calcular la nueva bbox a partir del desplazamiento
            new_bbox = (int(x + dx), int(y + dy), w, h)
            new_bbox = clip_bbox(new_bbox, (frame_h, frame_w))
            
            # Validar actualización con la máscara de movimiento
            roi_mask = fg_mask[new_bbox[1]:new_bbox[1] + new_bbox[3],
                               new_bbox[0]:new_bbox[0] + new_bbox[2]]
            if roi_mask.size > 0:
                motion_ratio = np.count_nonzero(roi_mask) / float(roi_mask.size)
                if motion_ratio < 0.1:
                    # Si no se detecta suficiente movimiento, se conserva la bbox anterior
                    new_bbox = (x, y, w, h)
                    new_bbox = clip_bbox(new_bbox, (frame_h, frame_w))
            
            # Actualizar el filtro de Kalman con la medición (centro de la bbox)
            kf = track['kalman']
            measured_center = np.array([[np.float32(new_bbox[0] + new_bbox[2] / 2)],
                                        [np.float32(new_bbox[1] + new_bbox[3] / 2)]])
            kf.correct(measured_center)
            predicted = kf.predict()
            pred_center = (predicted[0], predicted[1])
            new_x = int(pred_center[0] - new_bbox[2] / 2)
            new_y = int(pred_center[1] - new_bbox[3] / 2)
            new_bbox = (new_x, new_y, new_bbox[2], new_bbox[3])
            new_bbox = clip_bbox(new_bbox, (frame_h, frame_w))
            track['bbox'] = new_bbox
            updated_bboxes.append(new_bbox)
            
            # Actualizar el contador de ORB
            track['orb_age'] += 1
            
            # Condición para re-inicializar puntos con ORB:
            #   - Si quedan pocos puntos (menos de min_points_threshold)
            #   - O si ha pasado un cierto número de frames (max_orb_interval)
            reinit_orb = False
            if track['points'] is None or len(track['points']) < self.min_points_threshold:
                reinit_orb = True
            elif track['orb_age'] >= self.max_orb_interval:
                reinit_orb = True
            
            if reinit_orb:
                roi = current_gray[new_bbox[1]:new_bbox[1] + new_bbox[3],
                                   new_bbox[0]:new_bbox[0] + new_bbox[2]]
                if roi.size > 0:
                    keypoints = self.orb.detect(roi, None)
                    keypoints, descriptors = self.orb.compute(roi, keypoints)
                    if keypoints is not None and len(keypoints) > 0:
                        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
                        pts += np.array([[new_bbox[0], new_bbox[1]]], dtype=np.float32)
                        track['points'] = pts
                    else:
                        track['points'] = None
                else:
                    track['points'] = None
                track['orb_age'] = 0  # Reiniciamos el contador

        self.prev_gray = current_gray.copy()
        return updated_bboxes


class MultiKCFTracker:
    def __init__(self, initial_frame, initial_bboxes):
        self.trackers = []
        for bbox in initial_bboxes:

            try:
                tracker = cv2.TrackerKCF_create()
            except AttributeError:
                tracker = cv2.legacy.TrackerKCF_create()

            tracker.init(initial_frame, tuple(bbox))
            self.trackers.append(tracker)

    def update(self, frame):
        updated_boxes = []
        for tracker in self.trackers:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                updated_boxes.append((x, y, w, h))
        return updated_boxes
