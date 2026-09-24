"""Calibracion de perspectiva (pixeles -> metros) sobre el primer frame del video.

Uso:
  python tools/calibrate_pitch.py            # clic en puntos de la cancha y escribir sus metros
  python tools/calibrate_pitch.py --check    # dibuja las lineas de la cancha usando config.py

Coordenadas en metros: x = distancia a la linea de medio campo (positivo a la derecha),
y = distancia a la banda superior. Referencias utiles (cancha 105 x 68 m):
  medio campo con bandas: (0, 0) y (0, 68)
  circulo central (radio 9.15): (0, 24.85), (0, 43.15), (-9.15, 34), (9.15, 34)
  area grande: 16.5 m de profundidad, 40.32 m de ancho -> esquinas en y = 13.84 y 54.16
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config


def read_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f"No se pudo leer {video_path}")
    return frame


def pitch_lines_m(length=105, width=68):
    # Polilineas de la cancha en metros (mismo sistema de coordenadas que config.py)
    half = length/2
    lines = [
        [(-half, 0), (half, 0)], [(-half, width), (half, width)],
        [(-half, 0), (-half, width)], [(half, 0), (half, width)],
        [(0, 0), (0, width)],
    ]
    angles = np.linspace(0, 2*np.pi, 60)
    lines.append([(9.15*np.cos(a), 34 + 9.15*np.sin(a)) for a in angles])
    for side in (-1, 1):
        x_goal, x_box = side*half, side*(half-16.5)
        lines.append([(x_goal, 13.84), (x_box, 13.84), (x_box, 54.16), (x_goal, 54.16)])
    return lines


def draw_check(frame):
    H, _ = cv2.findHomography(np.float32(config.PITCH_POINTS_PX), np.float32(config.PITCH_POINTS_M))
    H_inv = np.linalg.inv(H)
    out = frame.copy()
    for line in pitch_lines_m(config.PITCH_LENGTH_M, config.PITCH_WIDTH_M):
        pts = cv2.perspectiveTransform(np.float32(line).reshape(-1, 1, 2), H_inv)
        cv2.polylines(out, [pts.astype(np.int32)], False, (255, 0, 255), 2)
    # Cuadricula cada 10 m
    for x in range(-50, 51, 10):
        pts = cv2.perspectiveTransform(np.float32([(x, 0), (x, 68)]).reshape(-1, 1, 2), H_inv)
        cv2.polylines(out, [pts.astype(np.int32)], False, (0, 255, 255), 1)
    for px in config.PITCH_POINTS_PX:
        cv2.circle(out, tuple(map(int, px)), 6, (0, 0, 255), -1)

    # Error de reproyeccion de los puntos de calibracion
    proj = cv2.perspectiveTransform(np.float32(config.PITCH_POINTS_PX).reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(proj - np.float32(config.PITCH_POINTS_M), axis=1)
    print("Error por punto (m):", np.round(errors, 2))
    return out


def click_points(frame):
    points = []
    display = frame.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(display, str(len(points)), (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.namedWindow('calibracion', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('calibracion', on_click)
    print("Haz clic en al menos 4 puntos conocidos de la cancha. Enter para terminar, Esc para cancelar.")
    while True:
        cv2.imshow('calibracion', display)
        key = cv2.waitKey(20)
        if key == 13 and len(points) >= 4:
            break
        if key == 27:
            sys.exit(0)
    cv2.destroyAllWindows()

    meters = []
    for i, p in enumerate(points, 1):
        value = input(f"Punto {i} {p} en metros 'x,y': ")
        meters.append(tuple(float(v) for v in value.split(',')))

    print("\nCopia esto en config.py:\n")
    print("PITCH_POINTS_PX = " + repr(points))
    print("PITCH_POINTS_M = " + repr(meters))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=config.VIDEO_PATH)
    parser.add_argument('--check', action='store_true', help='Dibujar la cancha proyectada con config.py')
    parser.add_argument('--output', default='output_videos/calibration_check.jpg')
    args = parser.parse_args()

    frame = read_first_frame(args.video)
    if args.check:
        cv2.imwrite(args.output, draw_check(frame))
        print(f"Imagen guardada en {args.output}")
    else:
        click_points(frame)
