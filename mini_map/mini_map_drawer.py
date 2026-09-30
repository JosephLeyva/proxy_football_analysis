import cv2
import sys
sys.path.append('../')
from utils import scale_boxes


class MiniMapDrawer():
    """Radar esquematico de la cancha completa con la posicion de cada jugador.

    Convierte 'position_transformed' (metros, misma convencion que config.PITCH_POINTS_M:
    x = distancia a la linea de medio campo, y = distancia a la banda superior) a pixeles
    dentro de un recuadro fijo del video, para ver a todo el equipo aunque la camara solo
    encuadre una parte de la cancha.
    """

    # Medidas reglamentarias (metros), igual que el radio de 9.15 m del circulo central
    PENALTY_AREA_DEPTH_M = 16.5
    PENALTY_AREA_WIDTH_M = 40.32
    GOAL_AREA_DEPTH_M = 5.5
    GOAL_AREA_WIDTH_M = 18.32

    def __init__(self, pitch_length, pitch_width, box,
                 bg_color=(40, 90, 40), bg_alpha=0.6, line_color=(255, 255, 255),
                 player_radius=5, ball_carrier_color=(0, 255, 255)):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.box = box
        self.bg_color = bg_color
        self.bg_alpha = bg_alpha
        self.line_color = line_color
        self.player_radius = player_radius
        self.ball_carrier_color = ball_carrier_color

    def _pitch_to_local_xy(self, x_m, y_m, rect_w, rect_h):
        local_x = (x_m + self.pitch_length / 2) / self.pitch_length * rect_w
        local_y = y_m / self.pitch_width * rect_h
        return int(round(local_x)), int(round(local_y))

    def _area_corners(self, depth_m, width_m, rect_w, rect_h):
        """Esquinas (izq, der) de un area rectangular simetrica pegada a cada porteria."""
        y_top = self.pitch_width / 2 - width_m / 2
        y_bottom = self.pitch_width / 2 + width_m / 2

        left_x1, left_y1 = self._pitch_to_local_xy(-self.pitch_length / 2, y_top, rect_w, rect_h)
        left_x2, left_y2 = self._pitch_to_local_xy(-self.pitch_length / 2 + depth_m, y_bottom, rect_w, rect_h)

        right_x1, right_y1 = self._pitch_to_local_xy(self.pitch_length / 2 - depth_m, y_top, rect_w, rect_h)
        right_x2, right_y2 = self._pitch_to_local_xy(self.pitch_length / 2, y_bottom, rect_w, rect_h)

        return (left_x1, left_y1, left_x2, left_y2), (right_x1, right_y1, right_x2, right_y2)

    def draw_mini_map(self, frames, tracks):
        x1, y1, x2, y2 = scale_boxes([self.box], frames[0].shape)[0]
        scale = frames[0].shape[0] / 1080
        line_thickness = max(1, round(2 * scale))
        player_radius = max(2, round(self.player_radius * scale))

        # Recuadro interno con la misma proporcion que la cancha (evita distorsionarla)
        box_w, box_h = x2 - x1, y2 - y1
        pitch_ratio = self.pitch_length / self.pitch_width
        if box_w / box_h > pitch_ratio:
            rect_h = box_h
            rect_w = int(round(rect_h * pitch_ratio))
        else:
            rect_w = box_w
            rect_h = int(round(rect_w / pitch_ratio))
        rect_x1 = x1 + (box_w - rect_w) // 2
        rect_y1 = y1 + (box_h - rect_h) // 2

        # Geometria estatica de la cancha (borde, medio campo, circulo central)
        half_x, _ = self._pitch_to_local_xy(0, 0, rect_w, rect_h)
        half_x += rect_x1
        center_x, center_y = self._pitch_to_local_xy(0, self.pitch_width / 2, rect_w, rect_h)
        center_x += rect_x1
        center_y += rect_y1
        circle_radius = int(round(9.15 / self.pitch_length * rect_w))

        penalty_left, penalty_right = self._area_corners(
            self.PENALTY_AREA_DEPTH_M, self.PENALTY_AREA_WIDTH_M, rect_w, rect_h)
        goal_left, goal_right = self._area_corners(
            self.GOAL_AREA_DEPTH_M, self.GOAL_AREA_WIDTH_M, rect_w, rect_h)
        area_rects = [penalty_left, penalty_right, goal_left, goal_right]

        output_video_frames = []
        for frame_num, frame in enumerate(frames):
            overlay = frame.copy()
            cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x1 + rect_w, rect_y1 + rect_h),
                          self.bg_color, -1)
            cv2.addWeighted(overlay, self.bg_alpha, frame, 1 - self.bg_alpha, 0, frame)

            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x1 + rect_w, rect_y1 + rect_h),
                          self.line_color, line_thickness)
            cv2.line(frame, (half_x, rect_y1), (half_x, rect_y1 + rect_h),
                     self.line_color, line_thickness)
            cv2.circle(frame, (center_x, center_y), circle_radius,
                       self.line_color, line_thickness)
            for ax1, ay1, ax2, ay2 in area_rects:
                cv2.rectangle(frame, (rect_x1 + ax1, rect_y1 + ay1), (rect_x1 + ax2, rect_y1 + ay2),
                              self.line_color, line_thickness)

            for _, track_info in tracks['players'][frame_num].items():
                position_transformed = track_info.get('position_transformed')
                if position_transformed is None:
                    continue

                x_m, y_m = position_transformed
                lx, ly = self._pitch_to_local_xy(x_m, y_m, rect_w, rect_h)
                px, py = rect_x1 + lx, rect_y1 + ly

                color = (self.ball_carrier_color if track_info.get('has_ball', False)
                         else track_info.get('team_color', (0, 0, 255)))
                cv2.circle(frame, (px, py), player_radius, color, -1)

            output_video_frames.append(frame)

        return output_video_frames
