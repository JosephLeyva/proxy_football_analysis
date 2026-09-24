import argparse
import config
from utils import read_video, save_video, get_video_fps
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main(recompute=False, draw_ids=config.DRAW_TRACK_IDS, draw_speed=config.DRAW_SPEED_DISTANCE,
         output_path=config.OUTPUT_PATH):
    use_stubs = not recompute

    # Read Video
    video_frames = read_video(config.VIDEO_PATH, max_seconds=config.MAX_VIDEO_SECONDS)
    fps = get_video_fps(config.VIDEO_PATH)
    print(f"{len(video_frames)} frames a {fps:.0f} fps")

    # Initialize Tracker
    tracker = Tracker(config.MODEL_PATH,
                      conf=config.DETECTION_CONF,
                      overlay_boxes=config.OVERLAY_BOXES,
                      ball_control_box=config.BALL_CONTROL_BOX)

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=use_stubs,
                                       stub_path=config.TRACK_STUB_PATH)
    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames,
                                    tracks['players'])
    moved = team_assigner.fix_referees(video_frames, tracks, max_ratio=config.REFEREE_TEAM_COLOR_RATIO)
    print(f"{len(moved)} tracks de 'arbitro' con camiseta de equipo pasaron a jugadores")

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0],
                                                        overlay_boxes=config.OVERLAY_BOXES,
                                                        text_box=config.CAMERA_MOVEMENT_BOX,
                                                        reference_frame=config.CALIBRATION_FRAME)
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=use_stubs,
                                                                                stub_path=config.CAMERA_STUB_PATH)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer(config.PITCH_POINTS_PX, config.PITCH_POINTS_M,
                                       config.PITCH_LENGTH_M, config.PITCH_WIDTH_M,
                                       margin=config.PITCH_MARGIN_M)
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Solo quienes estan en la cancha (quita stewards y fotografos detras de la valla)
    removed = tracker.remove_off_pitch(tracks)
    print(f"{removed} detecciones fuera de la cancha eliminadas")

    # Porteros: equipo segun la porteria que defienden (su camiseta no es del color del equipo)
    goalkeeper_teams = team_assigner.assign_goalkeepers(video_frames, tracks)
    if goalkeeper_teams:
        confidence = max(team_assigner.side_vote, 1-team_assigner.side_vote)
        print(f"Equipo {team_assigner.team_right} defiende la porteria derecha "
              f"({confidence*100:.0f}% de los frames); porteros: {goalkeeper_teams}")

    # Balon: de las candidatas de YOLO, la que sigue la trayectoria (descarta el punto penal, etc.)
    def ball_to_meters(frame_num, point):
        H = camera_movement_estimator.to_reference(camera_movement_per_frame, frame_num)
        return view_transformer.pixel_to_meters(point, H)
    rejected = tracker.select_ball(tracks, ball_to_meters, fps)
    print(f"{rejected} detecciones de balon descartadas por no seguir la trayectoria")

    # Interpolate Ball Positions (solo huecos cortos)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"], max_gap=config.BALL_MAX_GAP_FRAMES)

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator(frame_rate=fps,
                                                              max_speed_kmh=config.MAX_PLAYER_SPEED_KMH)
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner(max_player_ball_distance=config.MAX_PLAYER_BALL_DISTANCE)
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball = tracks['ball'][frame_num].get(1)
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball['bbox']) if ball else -1

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # Nadie cerca del balon (o no se ve): se mantiene el ultimo equipo con posesion (0 = aun nadie)
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    team_ball_control= np.array(team_ball_control)


    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control,team_assigner.team_colors,draw_ids=draw_ids)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    if draw_speed:
        speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # Save video
    save_video(output_video_frames, output_path, fps=fps)
    print(f"Video guardado en {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analisis de futbol con YOLO')
    parser.add_argument('--recompute', action='store_true',
                        help='Ignorar los stubs (.pkl) y volver a correr YOLO y optical flow')
    parser.add_argument('--ids', action=argparse.BooleanOptionalAction, default=config.DRAW_TRACK_IDS,
                        help='Dibujar el ID de cada jugador (default: config.DRAW_TRACK_IDS)')
    parser.add_argument('--speed', action=argparse.BooleanOptionalAction, default=config.DRAW_SPEED_DISTANCE,
                        help='Dibujar velocidad y distancia recorrida (default: config.DRAW_SPEED_DISTANCE)')
    parser.add_argument('--output', default=config.OUTPUT_PATH,
                        help='Ruta del video de salida (default: config.OUTPUT_PATH)')
    args = parser.parse_args()
    main(recompute=args.recompute, draw_ids=args.ids, draw_speed=args.speed, output_path=args.output)
