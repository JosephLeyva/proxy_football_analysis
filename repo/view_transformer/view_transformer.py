import numpy as np
import cv2

class ViewTransformer():
    def __init__(self, pixel_points, pitch_points, pitch_length=105, pitch_width=68, margin=0):
        # Homografia: pixeles del frame de calibracion -> metros sobre la cancha.
        # Con 4 puntos basta; con mas, findHomography reparte el error entre todos.
        self.pixel_vertices = np.array(pixel_points, dtype=np.float32)
        self.target_vertices = np.array(pitch_points, dtype=np.float32)

        self.persepctive_trasnformer, _ = cv2.findHomography(self.pixel_vertices, self.target_vertices)

        # Limites de la cancha en metros (x desde el medio campo, y desde la banda superior).
        # El margen deja dentro a jueces de linea y saques de banda, pero no a quien esta tras la valla
        self.x_limit = pitch_length/2 + margin
        self.y_min = -margin
        self.y_max = pitch_width + margin

    def pixel_to_meters(self, point, frame_to_reference=np.eye(3)):
        # Pixel de cualquier frame -> metros, sin revisar si cae dentro de la cancha
        H = self.persepctive_trasnformer @ frame_to_reference
        return cv2.perspectiveTransform(np.float32(point).reshape(1,1,2), H).ravel()

    def transform_point(self,point):
        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_point,self.persepctive_trasnformer).reshape(-1,2)

        x, y = tranform_point[0]
        is_inside = abs(x) <= self.x_limit and self.y_min <= y <= self.y_max
        if not is_inside:
            return None
        return tranform_point

    def add_transformed_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    # Sin position_adjusted (toma de otra camara) no se sabe donde esta: no se mide,
                    # pero tampoco se marca como fuera de la cancha
                    if position is None:
                        tracks[object][frame_num][track_id]['position_transformed'] = None
                        tracks[object][frame_num][track_id]['off_pitch'] = False
                        continue
                    position = np.array(position)
                    position_trasnformed = self.transform_point(position)
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed
                    tracks[object][frame_num][track_id]['off_pitch'] = position_trasnformed is None
