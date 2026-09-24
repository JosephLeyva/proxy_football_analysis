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


def main(recompute=False):
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

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0],
                                                        overlay_boxes=config.OVERLAY_BOXES,
                                                        text_box=config.CAMERA_MOVEMENT_BOX)
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=use_stubs,
                                                                                stub_path=config.CAMERA_STUB_PATH)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer(config.PITCH_POINTS_PX, config.PITCH_POINTS_M,
                                       config.PITCH_LENGTH_M, config.PITCH_WIDTH_M)
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator(frame_rate=fps,
                                                              max_speed_kmh=config.MAX_PLAYER_SPEED_KMH)
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames,
                                    tracks['players'])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner(max_player_ball_distance=config.MAX_PLAYER_BALL_DISTANCE)
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # Nadie cerca del balon: se mantiene el ultimo equipo con posesion (0 = aun nadie)
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    team_ball_control= np.array(team_ball_control)


    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control,team_assigner.team_colors)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # Save video
    save_video(output_video_frames, config.OUTPUT_PATH, fps=fps)
    print(f"Video guardado en {config.OUTPUT_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analisis de futbol con YOLO')
    parser.add_argument('--recompute', action='store_true',
                        help='Ignorar los stubs (.pkl) y volver a correr YOLO y optical flow')
    args = parser.parse_args()
    main(recompute=args.recompute)
