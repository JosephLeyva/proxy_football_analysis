# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Computer-vision pipeline that detects/tracks players, referees, and the ball in a football (soccer) clip with YOLO, assigns players to teams via K-Means on shirt color, tracks ball possession, estimates camera movement (pan **and zoom**) via optical flow, applies a perspective homography to convert pixel positions to meters, and computes player speed/distance. This is the adapted version used for a workshop (`taller/`) on a 2026 World Cup clip; the original base project is abdullahtarek/football_analysis.

## Commands

```bash
pip install -r requirements.txt

python main.py                       # run pipeline, reusing cached stubs in stubs/ if present
python main.py --recompute           # ignore stubs; rerun YOLO detection/tracking and optical flow
python main.py --no-ids              # skip drawing player track IDs
python main.py --no-speed            # skip drawing speed/distance overlay
python main.py --output path.mp4     # custom output path

python tools/calibrate_pitch.py                     # click pitch points on config.CALIBRATION_FRAME to derive PITCH_POINTS_PX/M
python tools/calibrate_pitch.py --check              # overlay projected pitch lines on CALIBRATION_FRAME to validate the homography
python tools/calibrate_pitch.py --check --frame 2000 # validate on another frame using the camera-movement stub (checks pan/zoom compensation, not just the base homography)
python tools/find_wide_shots.py video.mp4            # list continuous single-camera-take scenes (scene cuts break tracking/optical flow)
python tools/find_wide_shots.py video.mp4 --cut 3 --duration 25 --output input_videos/clip.mp4  # extract one scene as a clip
```

There is no test suite, linter, or build step in this repo.

## Architecture

`main.py` wires together a fixed pipeline of independent stage objects, each living in its own top-level package (`trackers`, `camera_movement_estimator`, `team_assigner`, `player_ball_assigner`, `view_transformer`, `speed_and_distance_estimator`, `utils`). All tunable parameters (paths, pitch calibration, thresholds, overlay boxes, draw flags) live in `config.py`, not in the stage classes — always change behavior there rather than hardcoding values in modules.

Pipeline order (see `main.py`):
1. `utils.read_video` loads frames into memory (capped by `config.MAX_VIDEO_SECONDS` — the whole clip is held in RAM; comments in `config.py` estimate RAM per frame at the source resolution).
2. `trackers.Tracker` runs YOLO + ByteTrack, producing a `tracks` dict keyed by `"players"`, `"referees"`, `"ball"` (per-frame `{track_id: {...}}`). Each player track carries `is_goalkeeper` (whether YOLO's `goalkeeper` class fired for that detection, even though goalkeepers are stored under `"players"`). Each ball detection keeps all raw candidate boxes (`candidates`), not just the highest-confidence one, for later disambiguation. `remove_short_tracks` drops tracks lasting only a few frames (edge-of-pitch flicker).
3. `TeamAssigner.assign_team_color` clusters shirt colors with K-Means; `get_player_team` re-samples each track's color every `sample_every` frames and votes over a sliding window (`vote_window`) rather than fixing the team permanently — this self-corrects after a tracker ID switch. `fix_referees` reclassifies "referee" tracks whose shirt color is much closer to one team's color than the other (`config.REFEREE_TEAM_COLOR_RATIO`) as players — YOLO sometimes mislabels a player as referee for long stretches.
4. `CameraMovementEstimator` estimates camera pan **and zoom** via optical flow, expressing every frame's homography relative to a fixed `reference_frame` (`config.CALIBRATION_FRAME`) rather than frame 0, to avoid drift from chaining per-frame homographies (it re-anchors to a new "keyframe" of tracked features once too few survive). `add_adjust_positions_to_tracks` and `to_reference` project any frame's pixel positions into the reference frame's coordinate system. Frames from another camera that can't be linked to the reference shot have no homography (`None`) — see "Camera model" below.
5. `ViewTransformer` builds a static homography from `config.PITCH_POINTS_PX`/`PITCH_POINTS_M`, calibrated on `config.CALIBRATION_FRAME` (not necessarily frame 0) via `tools/calibrate_pitch.py`, and adds transformed (meters) positions to tracks, combining it with each frame's camera-movement homography. `config.PITCH_MARGIN_M` keeps points slightly outside the true pitch boundary (linesmen, throw-ins) but excludes people further out (stewards, photographers).
6. `tracker.remove_off_pitch` drops any player/referee detection marked `off_pitch` by step 5 (measured outside the pitch + margin) — detections in frames without a camera homography are kept, since their position is unknown.
7. `TeamAssigner.assign_goalkeepers` assigns goalkeepers to a team by field side rather than shirt color (goalkeepers don't wear team colors): it votes, per frame, which team occupies the right side of the pitch on average, then assigns each goalkeeper track to whichever side its median x-position sits on.
8. `tracker.select_ball` disambiguates ball candidates per frame using a `to_meters` callback (composes the camera-movement and view-transformer homographies): candidates that reappear motionless far from any player (field markings, e.g. the penalty spot) are discarded, then the candidate closest to the last known ball position is chosen if reachable within `max_speed` (m/s), falling back to the highest-confidence candidate when there's no recent ball position to anchor to.
9. `tracker.interpolate_ball_positions` interpolates only gaps shorter than `config.BALL_MAX_GAP_FRAMES`; longer gaps are left without a ball rather than drawing a straight-line guess across a long occlusion.
10. `SpeedAndDistance_Estimator` uses the real video FPS (`utils.get_video_fps`) and discards speeds above `config.MAX_PLAYER_SPEED_KMH` as measurement noise.
11. `PlayerBallAssigner` assigns possession per frame by foot-to-ball distance (`config.MAX_PLAYER_BALL_DISTANCE`); when no one is close enough (or the ball wasn't resolved that frame), possession carries over from the previous frame.
12. Drawing/annotation happens last and mutates frames in place (not copies, to limit RAM) before `utils.save_video` writes H.264 output. `--ids`/`--no-ids` and `--speed`/`--no-speed` CLI flags (defaulting to `config.DRAW_TRACK_IDS`/`DRAW_SPEED_DISTANCE`) toggle what gets drawn. All on-screen boxes/text in `config.py` and drawing code are defined in 1920x1080 pixel space and scaled to the source resolution via `utils.scale_boxes`.

### Camera model: pan + zoom, not just translation

Unlike a simple panning-only assumption, `CameraMovementEstimator` fits a full homography (via `cv2.findHomography` + RANSAC) between tracked background features and a keyframe, so it also compensates zoom.

Broadcast clips also cut to other cameras (close-ups, replays), which optical flow cannot follow. `track_shots` splits the video into shots — a cut is a sudden color-histogram change between consecutive frames (`cut_hist_corr`) or too few RANSAC inliers (`min_inliers`) — and tracks with optical flow within each shot. `link_shots` then chains shots to the one containing `reference_frame` by SIFT-matching anchor frames, accepting only matches with enough inliers and a moderate zoom change (`link_max_zoom`), so the main camera returning after a close-up is re-registered but a close-up is not. Frames of unlinked shots get `None` instead of a homography: `to_reference` returns `None`, their tracks get `position_adjusted`/`position_transformed = None` and `off_pitch = False` (they are not measured in meters, but not removed either), and `select_ball` falls back to the highest-confidence candidate there. `tools/find_wide_shots.py` is still useful to extract a clip, but it is no longer required to feed a single take.

### Pitch calibration is frame-specific, not always frame 0

`config.CALIBRATION_FRAME` (not necessarily 0) is the frame `PITCH_POINTS_PX`/`PITCH_POINTS_M` were measured on — pick a frame where enough known pitch lines/landmarks are visible and re-run `tools/calibrate_pitch.py` whenever the source video or camera framing changes. `tools/calibrate_pitch.py --check --frame N` validates the calibration on a different frame by composing it with the camera-movement stub, which also serves as a sanity check that pan/zoom compensation itself is working.

Stubs (`stubs/*.pkl`) cache the expensive YOLO tracking and optical flow results per source video; regenerate them with `--recompute` whenever the input video, detection model, or `config.CALIBRATION_FRAME` changes. `config.py` is the single source of truth for which video/model/stub files are active — when switching to a different clip, update `VIDEO_PATH`, the stub paths, and re-run pitch calibration.

`taller/` and `training/` contain workshop/notebook material (Spanish-language), not part of the runtime pipeline.
