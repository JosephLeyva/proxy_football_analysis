import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import scale_boxes

class CameraMovementEstimator():
    def __init__(self,frame, overlay_boxes=None, text_box=(0,0,500,100)):
        self.minimum_distance = 1
        self.text_box = text_box

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        # Buscar puntos en todo el frame excepto los overlays de la TV (marcador, logo),
        # que no se mueven con la camara
        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.ones_like(first_frame_grayscale)
        for x1,y1,x2,y2 in scale_boxes(overlay_boxes or [], frame.shape):
            mask_features[y1:y2,x1:x2] = 0

        self.features = dict(
            maxCorners = 300,
            qualityLevel = 0.01,
            minDistance =10,
            blockSize = 7,
            mask = mask_features
        )

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    # Si la camara se movio +X, la imagen se corrio -X: se suma para volver
                    # a las coordenadas del primer frame (donde se calibro la perspectiva)
                    position_adjusted = (position[0]+camera_movement[0],position[1]+camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted



    def get_camera_movement(self,frames,read_from_stub=False, stub_path=None):
        # Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        # Movimiento acumulado de la camara respecto al primer frame
        camera_movement = [[0,0]]*len(frames)
        total_x, total_y = 0.0, 0.0

        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)

        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)
            new_features, status,_ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)

            # La mediana del desplazamiento ignora a los jugadores (que se mueven solos);
            # la mayoria de los puntos son cancha y tribunas, que solo se mueven con la camara
            good = status.ravel() == 1
            if good.sum() > 10:
                shift = np.median(old_features[good].reshape(-1,2) - new_features[good].reshape(-1,2), axis=0)
                if np.hypot(*shift) > self.minimum_distance:
                    total_x += float(shift[0])
                    total_y += float(shift[1])

            camera_movement[frame_num] = [total_x,total_y]
            old_gray = frame_gray

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path) or '.', exist_ok=True)
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement

    def draw_camera_movement(self,frames, camera_movement_per_frame):
        # Dibuja directamente sobre los frames (sin copiarlos) para no duplicar la RAM
        output_frames=[]
        x1,y1,x2,y2 = self.text_box

        for frame_num, frame in enumerate(frames):
            overlay = frame.copy()
            cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camara X: {x_movement:.1f} px",(x1+10,y1+35), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camara Y: {y_movement:.1f} px",(x1+10,y1+75), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame)

        return output_frames
