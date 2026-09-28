import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import scale_boxes

class CameraMovementEstimator():
    """Estima el movimiento de camara (paneo + zoom) con optical flow.

    Una camara de TV gira sobre su tripie y hace zoom: entre dos frames la imagen cambia segun
    una homografia. Con ella cada frame se puede llevar a las coordenadas de un frame de
    referencia (donde se calibro la cancha).

    Una transmision tambien corta a otras camaras (primeros planos, repeticiones). El optical flow
    no puede seguir un corte, asi que el video se parte en tomas: dentro de cada una se sigue con
    optical flow, y las tomas se enlazan entre si comparando frames con SIFT. Las tomas que no se
    pueden enlazar con la del frame de referencia (otra camara) quedan sin homografia (None).
    """
    def __init__(self,frame, overlay_boxes=None, text_box=(0,0,500,100), reference_frame=0):
        self.reference_frame = reference_frame
        self.text_box = text_box

        self.min_tracked_features = 150

        # Corte de camara: el histograma de color cambia de golpe entre dos frames seguidos
        # (en fifa_clip.mp4: 0.47-0.89 en los cortes, nunca menos de 0.989 dentro de una toma)
        self.cut_hist_corr = 0.95
        # ... o casi ningun punto encaja en una homografia (en los cortes quedan ~6 de 500)
        self.min_inliers = 12

        # Enlace entre tomas con SIFT: minimo de coincidencias que encajan y cambio de zoom maximo
        # (la camara principal vuelve mas o menos a donde estaba; un primer plano es 3-5x)
        self.link_min_inliers = 40
        self.link_max_zoom = 2.5
        self.anchor_every = 50   # frames entre anclas SIFT dentro de una toma

        self.lk_params = dict(
            winSize = (21,21),
            maxLevel = 3,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,30,0.01)
        )

        # Buscar puntos en todo el frame excepto los overlays de la TV (marcador, logo),
        # que no se mueven con la camara
        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.ones_like(first_frame_grayscale)
        for x1,y1,x2,y2 in scale_boxes(overlay_boxes or [], frame.shape):
            mask_features[y1:y2,x1:x2] = 0

        self.features = dict(
            maxCorners = 500,
            qualityLevel = 0.01,
            minDistance =10,
            blockSize = 7,
            mask = mask_features
        )

        # SIFT a media resolucion (suficiente para enlazar tomas y 4x mas rapido)
        self.sift = cv2.SIFT_create(2000)
        self.sift_scale = 0.5
        self.sift_mask = cv2.resize(mask_features, None, fx=self.sift_scale, fy=self.sift_scale,
                                    interpolation=cv2.INTER_NEAREST)

    def to_reference(self, camera_movement_per_frame, frame_num):
        # Homografia: pixeles del frame frame_num -> pixeles del frame de referencia.
        # None si el frame es de otra toma que no se pudo enlazar (primer plano, otra camara)
        if camera_movement_per_frame[frame_num] is None:
            return None
        H_to_first = np.array(camera_movement_per_frame[frame_num])
        H_ref_to_first = np.array(camera_movement_per_frame[self.reference_frame])
        return np.linalg.inv(H_ref_to_first) @ H_to_first

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                H = self.to_reference(camera_movement_per_frame, frame_num)
                for track_id, track_info in track.items():
                    if H is None:
                        tracks[object][frame_num][track_id]['position_adjusted'] = None
                        continue
                    position = np.float32(track_info['position']).reshape(1,1,2)
                    position_adjusted = cv2.perspectiveTransform(position, H).ravel()
                    tracks[object][frame_num][track_id]['position_adjusted'] = tuple(position_adjusted.tolist())

    def get_camera_movement(self,frames,read_from_stub=False, stub_path=None):
        # Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        shots = self.track_shots(frames)
        camera_movement = self.link_shots(frames, shots)

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path) or '.', exist_ok=True)
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement

    def color_hist(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1], self.features['mask'], [30,32], [0,180,0,256])
        return cv2.normalize(hist, hist).flatten()

    def track_shots(self, frames):
        # Parte el video en tomas continuas. Cada toma: {'start': frame inicial, 'H': [homografia de
        # cada frame de la toma -> primer frame de la toma]}
        shots = []
        prev_gray, prev_hist = None, None
        key_features = features = None

        for frame_num, frame in enumerate(frames):
            frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame_hist = self.color_hist(frame)

            # Encadenar una homografia por frame acumula error (drift). En vez de eso seguimos los
            # mismos puntos desde un "keyframe" y calculamos la homografia directo contra el; solo
            # cuando quedan pocos puntos (la camara ya mostro otra zona) se cambia de keyframe.
            H = None
            is_cut = prev_hist is None or cv2.compareHist(prev_hist, frame_hist, cv2.HISTCMP_CORREL) < self.cut_hist_corr
            if not is_cut and features is not None and len(features) >= self.min_inliers:
                new_features, status,_ = cv2.calcOpticalFlowPyrLK(prev_gray,frame_gray,features,None,**self.lk_params)
                good = status.ravel() == 1
                key_features, features = key_features[good], new_features[good]

                # RANSAC descarta los puntos que se mueven por su cuenta (jugadores, pantallas LED);
                # la mayoria son cancha y tribunas, que solo se mueven con la camara
                if len(features) >= self.min_inliers:
                    H, inliers = cv2.findHomography(features, key_features, cv2.RANSAC, 2.0)
                    if H is not None and inliers.sum() < self.min_inliers:
                        H = None

            if H is None:
                # Corte de camara (o se perdio el seguimiento): empieza una toma nueva con este frame
                # como raiz. link_shots decide despues si se puede enlazar con las demas.
                shots.append({'start': frame_num, 'H': [np.eye(3)]})
                H_key_to_root = np.eye(3)
                key_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)
                features = None if key_features is None else key_features.copy()
            else:
                H_to_root = H_key_to_root @ H
                H_to_root /= H_to_root[2,2]
                shots[-1]['H'].append(H_to_root)

                # Los puntos que no encajan (jugadores) se descartan para siempre
                inliers = inliers.ravel() == 1
                key_features, features = key_features[inliers], features[inliers]
                if len(features) < self.min_tracked_features:
                    H_key_to_root = H_to_root
                    key_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)
                    features = None if key_features is None else key_features.copy()

            prev_gray, prev_hist = frame_gray, frame_hist

        return shots

    def sift_features(self, frame, cache, frame_num):
        if frame_num not in cache:
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None,
                               fx=self.sift_scale, fy=self.sift_scale, interpolation=cv2.INTER_AREA)
            cache[frame_num] = self.sift.detectAndCompute(small, self.sift_mask)
        return cache[frame_num]

    def match_frames(self, frames, cache, frame_a, frame_b):
        # Homografia pixeles de frame_a -> pixeles de frame_b con SIFT, o None si no son la misma escena
        kp_a, desc_a = self.sift_features(frames[frame_a], cache, frame_a)
        kp_b, desc_b = self.sift_features(frames[frame_b], cache, frame_b)
        if desc_a is None or desc_b is None or len(kp_a) < 2 or len(kp_b) < 2:
            return None
        matches = cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.7*m[1].distance]
        if len(good) < self.link_min_inliers:
            return None
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]) / self.sift_scale
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]) / self.sift_scale
        H, inliers = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 3.0)
        if H is None or inliers.sum() < self.link_min_inliers:
            return None

        # Zoom relativo (raiz del cambio de area del frame): un primer plano de otra camara que
        # "encaja" en un pedazo de tribuna da un zoom enorme, la camara principal no
        h, w = frames[frame_a].shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        x, y = cv2.perspectiveTransform(corners, H).reshape(-1,2).T
        zoom = np.sqrt(0.5*abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1))) / (w*h))
        if not (1/self.link_max_zoom <= zoom <= self.link_max_zoom):
            return None
        return H

    def link_shots(self, frames, shots):
        # Lleva cada toma al sistema de coordenadas de la toma del frame de referencia.
        # camera_movement[k] = homografia frame k -> primer frame de la toma de referencia
        # (None si la toma de k no se pudo enlazar)
        def anchors(shot):
            end = shot['start'] + len(shot['H']) - 1
            return sorted(set(range(shot['start'], end, self.anchor_every)) | {end})

        ref_shot = next(i for i, s in enumerate(shots)
                        if s['start'] <= self.reference_frame < s['start'] + len(s['H']))
        to_ref_root = {ref_shot: np.eye(3)}   # toma -> homografia (su raiz -> raiz de la toma de referencia)
        cache, tried = {}, set()

        progress = True
        while progress:
            progress = False
            for i, shot in enumerate(shots):
                if i in to_ref_root:
                    continue
                # Probar primero contra las anclas mas cercanas en el tiempo de tomas ya enlazadas
                # (la camara principal vuelve a donde estaba antes del corte)
                pairs = sorted(((a, b, j) for j in to_ref_root for a in anchors(shot) for b in anchors(shots[j])
                                if (a, b) not in tried), key=lambda p: abs(p[0]-p[1]))
                for a, b, j in pairs[:6]:
                    tried.add((a, b))
                    M = self.match_frames(frames, cache, a, b)
                    if M is None:
                        continue
                    H_a = shot['H'][a - shot['start']]
                    H_b = shots[j]['H'][b - shots[j]['start']]
                    link = to_ref_root[j] @ H_b @ M @ np.linalg.inv(H_a)
                    to_ref_root[i] = link / link[2,2]
                    progress = True
                    break

        camera_movement = []
        for i, shot in enumerate(shots):
            for H in shot['H']:
                if i in to_ref_root:
                    H = to_ref_root[i] @ H
                    camera_movement.append((H / H[2,2]).tolist())
                else:
                    camera_movement.append(None)

        linked_frames = sum(len(shots[i]['H']) for i in to_ref_root)
        print(f"Camara: {len(shots)} tomas, {len(to_ref_root)} enlazadas con la del frame de referencia; "
              f"{len(frames)-linked_frames} frames sin referencia (otra camara)")
        return camera_movement

    def get_camera_summary(self, camera_movement_per_frame, frame_shape):
        # Resumen legible por frame: desplazamiento (px) y zoom respecto al frame de referencia
        h, w = frame_shape[:2]
        segment = np.float32([[w/2-50, h/2], [w/2+50, h/2]]).reshape(-1,1,2)
        summary = []
        for frame_num in range(len(camera_movement_per_frame)):
            H = self.to_reference(camera_movement_per_frame, frame_num)
            if H is None:
                summary.append((np.nan, np.nan, np.nan))
                continue
            (x1,y1),(x2,y2) = cv2.perspectiveTransform(segment, H).reshape(-1,2)
            zoom = 100/max(np.hypot(x2-x1,y2-y1),1e-6)
            summary.append(((x1+x2)/2-w/2, (y1+y2)/2-h/2, zoom))
        return np.array(summary)

    def draw_camera_movement(self,frames, camera_movement_per_frame):
        # Dibuja directamente sobre los frames (sin copiarlos) para no duplicar la RAM
        output_frames=[]
        x1,y1,x2,y2 = scale_boxes([self.text_box], frames[0].shape)[0]
        scale = frames[0].shape[0]/1080
        summary = self.get_camera_summary(camera_movement_per_frame, frames[0].shape)

        for frame_num, frame in enumerate(frames):
            overlay = frame.copy()
            cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement, zoom = summary[frame_num]
            thickness = max(1, round(3*scale))
            if np.isnan(zoom):
                lines = ["Camara: otra toma", "(sin referencia)"]
            else:
                lines = [f"Camara X: {x_movement:.0f} px  Y: {y_movement:.0f} px", f"Zoom: x{zoom:.2f}"]
            for i, text in enumerate(lines):
                frame = cv2.putText(frame,text,(x1+int(10*scale),y1+int((40+40*i)*scale)), cv2.FONT_HERSHEY_SIMPLEX,0.9*scale,(0,0,0),thickness)

            output_frames.append(frame)

        return output_frames
