import numpy as np
import cv2

class ViewTransformer():
    def __init__(self, pixel_points, pitch_points, pitch_length=105, pitch_width=68):
        # Homografia: pixeles del primer frame -> metros sobre la cancha.
        # Con 4 puntos basta; con mas, findHomography reparte el error entre todos.
        self.pixel_vertices = np.array(pixel_points, dtype=np.float32)
        self.target_vertices = np.array(pitch_points, dtype=np.float32)

        self.persepctive_trasnformer, _ = cv2.findHomography(self.pixel_vertices, self.target_vertices)

        # Limites de la cancha en metros (x desde el medio campo, y desde la banda superior)
        self.x_limit = pitch_length/2
        self.y_limit = pitch_width

    def transform_point(self,point):
        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_point,self.persepctive_trasnformer).reshape(-1,2)

        x, y = tranform_point[0]
        is_inside = abs(x) <= self.x_limit and 0 <= y <= self.y_limit
        if not is_inside:
            return None
        return tranform_point

    def add_transformed_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_trasnformed = self.transform_point(position)
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed
