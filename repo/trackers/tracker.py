from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position, get_pitch_mask, scale_boxes, point_in_boxes
from collections import Counter

class Tracker:
    def __init__(self, model_path, conf=0.1, overlay_boxes=None, ball_control_box=(1350,850,1900,970)):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()
        self.conf = conf
        self.overlay_boxes = overlay_boxes or []
        self.ball_control_box = ball_control_box

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=self.conf,verbose=False)
            detections += detections_batch
            print(f"Detectando: {min(i+batch_size,len(frames))}/{len(frames)} frames", end='\r')
        print()
        return detections

    def filter_detections(self, frame, detections):
        # Quitar detecciones fuera de la cancha (tribunas, anuncios) o sobre el marcador de la TV
        pitch_mask = get_pitch_mask(frame)
        overlay_boxes = scale_boxes(self.overlay_boxes, frame.shape)
        keep = []
        for bbox in detections.xyxy:
            x, y = get_foot_position(bbox)
            x = min(max(x,0),frame.shape[1]-1)
            y = min(max(y,0),frame.shape[0]-1)
            keep.append(bool(pitch_mask[y,x]) and not point_in_boxes(get_center_of_bbox(bbox), overlay_boxes))
        return detections[np.array(keep, dtype=bool)]

    def fix_track_classes(self, tracks):
        # YOLO a veces confunde jugador/arbitro en frames sueltos:
        # cada track_id se queda con la clase que tuvo en la mayoria de los frames
        votes = {}
        for object in ["players","referees"]:
            for track in tracks[object]:
                for track_id in track:
                    votes.setdefault(track_id, Counter())[object] += 1
        majority = {track_id: counter.most_common(1)[0][0] for track_id, counter in votes.items()}

        for frame_num in range(len(tracks["players"])):
            for object, other in [("players","referees"),("referees","players")]:
                for track_id in list(tracks[object][frame_num].keys()):
                    if majority[track_id] == other:
                        tracks[other][frame_num][track_id] = tracks[object][frame_num].pop(track_id)

    def remove_short_tracks(self, tracks, min_frames=5):
        # Tracks que duran unos pocos frames son falsos positivos (parpadeos en el borde de la cancha)
        lifetimes = Counter(track_id for object in ["players","referees"]
                            for track in tracks[object] for track_id in track)
        for object in ["players","referees"]:
            for track in tracks[object]:
                for track_id in [t for t in track if lifetimes[t] < min_frames]:
                    del track[track_id]

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_supervision = self.filter_detections(frames[frame_num], detection_supervision)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            # Balon: si hay varias detecciones, quedarse con la de mayor confianza
            ball_detections = detection_supervision[detection_supervision.class_id == cls_names_inv['ball']]
            if len(ball_detections) > 0:
                best = int(np.argmax(ball_detections.confidence))
                tracks["ball"][frame_num][1] = {"bbox":ball_detections.xyxy[best].tolist()}

        self.fix_track_classes(tracks)
        self.remove_short_tracks(tracks)

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path) or '.', exist_ok=True)
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control,team_colors=None):
        # Draw a semi-transparent rectaggle 
        x1, y1, x2, y2 = self.ball_control_box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        total = max(team_1_num_frames+team_2_num_frames, 1)
        team_1 = team_1_num_frames/total
        team_2 = team_2_num_frames/total

        cv2.putText(frame, f"Posesion Equipo 1: {team_1*100:.1f}%",(x1+60,y1+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Posesion Equipo 2: {team_2*100:.1f}%",(x1+60,y1+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        # Cuadrito con el color de cada equipo
        if team_colors:
            for i, team in enumerate([1,2]):
                cv2.rectangle(frame, (x1+20,y1+25+50*i), (x1+45,y1+50+50*i), team_colors[team], cv2.FILLED)
                cv2.rectangle(frame, (x1+20,y1+25+50*i), (x1+45,y1+50+50*i), (0,0,0), 2)

        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control,team_colors=None):
        # Dibuja directamente sobre los frames (sin copiarlos) para no duplicar la RAM
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(255,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, team_colors)

            output_video_frames.append(frame)

        return output_video_frames