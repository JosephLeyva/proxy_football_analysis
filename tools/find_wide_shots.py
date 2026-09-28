"""Encuentra tomas continuas (sin cortes) en un video de transmision y opcionalmente recorta una.

Uso:
  python tools/find_wide_shots.py video.mp4                  # lista tomas de 15 s o mas
  python tools/find_wide_shots.py video.mp4 --cut 3 --duration 25 --output input_videos/clip.mp4

El pipeline asume una sola toma: un corte de camara rompe el tracking y el optical flow.
Revisa las tomas listadas: las repeticiones y primeros planos tambien aparecen como tomas.
"""
import argparse
import subprocess

from scenedetect import detect, ContentDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('--min-seconds', type=float, default=15)
    parser.add_argument('--cut', type=int, help='Numero de toma a recortar')
    parser.add_argument('--start-offset', type=float, default=0, help='Segundos a saltar dentro de la toma')
    parser.add_argument('--duration', type=float, default=25)
    parser.add_argument('--output', default='input_videos/clip.mp4')
    args = parser.parse_args()

    scenes = detect(args.video, ContentDetector())
    long_scenes = [(s, e) for s, e in scenes if (e - s).get_seconds() >= args.min_seconds]

    for i, (start, end) in enumerate(long_scenes, 1):
        print(f"{i:3d}  {start.get_timecode()} -> {end.get_timecode()}  ({(end - start).get_seconds():.1f} s)")

    if args.cut:
        start, end = long_scenes[args.cut - 1]
        t0 = start.get_seconds() + args.start_offset
        duration = min(args.duration, end.get_seconds() - t0)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(t0), '-t', str(duration), '-i', args.video,
                        '-an', '-c:v', 'libx264', '-crf', '18', args.output], check=True)
        print(f"Clip guardado en {args.output} ({duration:.1f} s)")


if __name__ == '__main__':
    main()
