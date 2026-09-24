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
    """
    def __init__(self,frame, overlay_boxes=None, text_box=(0,0,500,100), reference_frame=0):
        self.reference_frame = reference_frame
        self.text_box = text_box

        self.min_tracked_features = 150

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

    def to_reference(self, camera_movement_per_frame, frame_num):
        # Homografia: pixeles del frame frame_num -> pixeles del frame de referencia
        H_to_first = np.array(camera_movement_per_frame[frame_num])
        H_ref_to_first = np.array(camera_movement_per_frame[self.reference_frame])
        return np.linalg.inv(H_ref_to_first) @ H_to_first

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                H = self.to_reference(camera_movement_per_frame, frame_num)
                for track_id, track_info in track.items():
                    position = np.float32(track_info['position']).reshape(1,1,2)
                    position_adjusted = cv2.perspectiveTransform(position, H).ravel()
                    tracks[object][frame_num][track_id]['position_adjusted'] = tuple(position_adjusted.tolist())

    def get_camera_movement(self,frames,read_from_stub=False, stub_path=None):
        # Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        # camera_movement[k] = homografia que lleva el frame k al frame 0
        camera_movement = [np.eye(3).tolist()]

        # Encadenar una homografia por frame acumula error (drift). En vez de eso seguimos los mismos
        # puntos desde un "keyframe" y calculamos la homografia directo contra el; solo cuando quedan
        # pocos puntos (la camara ya mostro otra zona) se cambia de keyframe.
        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        key_features = cv2.goodFeaturesToTrack(old_gray,**self.features)
        features = key_features.copy()
        H_key_to_first = np.eye(3)

        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            new_features, status,_ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,features,None,**self.lk_params)
            good = status.ravel() == 1
            key_features, features = key_features[good], new_features[good]

            # RANSAC descarta los puntos que se mueven por su cuenta (jugadores, pantallas LED);
            # la mayoria son cancha y tribunas, que solo se mueven con la camara
            H = None
            if len(features) >= 15:
                H, inliers = cv2.findHomography(features, key_features, cv2.RANSAC, 2.0)
            if H is None:
                H_to_first = np.array(camera_movement[-1])
                inliers = np.zeros((len(features),1))
            else:
                H_to_first = H_key_to_first @ H
                H_to_first /= H_to_first[2,2]
            camera_movement.append(H_to_first.tolist())

            # Los puntos que no encajan (jugadores) se descartan para siempre
            inliers = inliers.ravel() == 1
            key_features, features = key_features[inliers], features[inliers]
            if len(features) < self.min_tracked_features:
                H_key_to_first = H_to_first
                key_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)
                features = key_features.copy()

            old_gray = frame_gray

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path) or '.', exist_ok=True)
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement

    def get_camera_summary(self, camera_movement_per_frame, frame_shape):
        # Resumen legible por frame: desplazamiento (px) y zoom respecto al frame de referencia
        h, w = frame_shape[:2]
        segment = np.float32([[w/2-50, h/2], [w/2+50, h/2]]).reshape(-1,1,2)
        summary = []
        for frame_num in range(len(camera_movement_per_frame)):
            H = self.to_reference(camera_movement_per_frame, frame_num)
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
            frame = cv2.putText(frame,f"Camara X: {x_movement:.0f} px  Y: {y_movement:.0f} px",(x1+int(10*scale),y1+int(40*scale)), cv2.FONT_HERSHEY_SIMPLEX,0.9*scale,(0,0,0),thickness)
            frame = cv2.putText(frame,f"Zoom: x{zoom:.2f}",(x1+int(10*scale),y1+int(80*scale)), cv2.FONT_HERSHEY_SIMPLEX,0.9*scale,(0,0,0),thickness)

            output_frames.append(frame)

        return output_frames
