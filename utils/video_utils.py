import cv2
import shutil
import subprocess
import os

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 24

def read_video(video_path, max_seconds=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(max_seconds*fps) if max_seconds else None
    if max_frames and total_frames > max_frames:
        # Todos los frames se guardan en RAM (~6 MB por frame a 1080p)
        print(f"Aviso: el video dura {total_frames/fps:.0f}s, solo se leen los primeros {max_seconds}s. "
              f"Recorta un clip mas corto con ffmpeg.")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and len(frames) >= max_frames):
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(ouput_video_frames,output_video_path, fps=24):
    os.makedirs(os.path.dirname(output_video_path) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()

    # mp4v no se reproduce en navegadores ni en Colab; si hay ffmpeg, recodificar a H.264
    if shutil.which('ffmpeg'):
        tmp_path = output_video_path + '.tmp.mp4'
        os.replace(output_video_path, tmp_path)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp_path, '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', '-crf', '23', output_video_path], check=True)
        os.remove(tmp_path)
