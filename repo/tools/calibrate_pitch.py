"""Calibracion de perspectiva (pixeles -> metros) sobre un frame del video.

Uso:
  python tools/calibrate_pitch.py                 # clic en puntos de la cancha del frame CALIBRATION_FRAME
  python tools/calibrate_pitch.py --check         # dibuja la cancha proyectada en ese frame usando config.py
  python tools/calibrate_pitch.py --check --frame 2000
      # dibuja la cancha en otro frame siguiendo el movimiento de camara (requiere el stub de camara)

Elige un frame donde se vean varias lineas conocidas y pon su numero en config.CALIBRATION_FRAME.

Al clickear un punto NO se escriben metros a mano: se muestra una ventana con un diagrama de la
cancha con marcas numeradas (circulo central, esquinas de area, banderines de corner, etc.) y solo
hay que indicar que numero de marca es el punto clickeado. Los metros de cada marca ya vienen
correctos de fabrica (ver LANDMARKS) — la aritmetica de perspectiva/metros que antes se hacia a
mano (y donde se colaban los errores) queda eliminada.
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config


def read_frame(video_path, frame_num):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f"No se pudo leer el frame {frame_num} de {video_path}")
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
        x_small = side*(half-5.5)
        lines.append([(x_goal, 24.84), (x_small, 24.84), (x_small, 43.16), (x_goal, 43.16)])
    return lines


def build_landmarks(length=105, width=68):
    # Marcas conocidas de la cancha, con sus metros ya correctos (mismas constantes que pitch_lines_m:
    # circulo radio 9.15, area grande 16.5x40.32, area chica 5.5x18.32, punto penal a 11m del arco).
    # "Neg"/"Pos" y "Banda A"/"Banda B" son solo nombres para distinguir cada mitad/banda en el
    # diagrama: no importa cual banda o mitad fisica del video sea cada una, solo hay que ser
    # consistente al elegir la marca de cada punto clickeado.
    half = length/2
    circle_r = 9.15
    box_x, small_x = half-16.5, half-5.5
    box_near, box_far = width/2-20.16, width/2+20.16
    small_near, small_far = width/2-9.16, width/2+9.16
    return {
        1:  ("Centro de la cancha",                (0, width/2)),
        2:  ("Medio campo - Banda A",               (0, 0)),
        3:  ("Medio campo - Banda B",               (0, width)),
        4:  ("Circulo central - lado Banda A",      (0, width/2-circle_r)),
        5:  ("Circulo central - lado Banda B",      (0, width/2+circle_r)),
        6:  ("Circulo central - lado Neg",          (-circle_r, width/2)),
        7:  ("Circulo central - lado Pos",          (circle_r, width/2)),
        8:  ("Area grande Neg - esquina Banda A",   (-box_x, box_near)),
        9:  ("Area grande Neg - esquina Banda B",   (-box_x, box_far)),
        10: ("Area grande Pos - esquina Banda A",   (box_x, box_near)),
        11: ("Area grande Pos - esquina Banda B",   (box_x, box_far)),
        12: ("Area chica Neg - esquina Banda A",    (-small_x, small_near)),
        13: ("Area chica Neg - esquina Banda B",    (-small_x, small_far)),
        14: ("Area chica Pos - esquina Banda A",    (small_x, small_near)),
        15: ("Area chica Pos - esquina Banda B",    (small_x, small_far)),
        16: ("Esquina cancha Neg-A",                (-half, 0)),
        17: ("Esquina cancha Neg-B",                (-half, width)),
        18: ("Esquina cancha Pos-A",                (half, 0)),
        19: ("Esquina cancha Pos-B",                (half, width)),
        20: ("Punto penal Neg",                     (-(half-11), width/2)),
        21: ("Punto penal Pos",                     (half-11, width/2)),
    }


LANDMARKS = build_landmarks(config.PITCH_LENGTH_M, config.PITCH_WIDTH_M)


def render_landmark_diagram(landmarks, length=105, width=68, scale=8, margin_m=6):
    # Vista de pajaro de la cancha con cada marca numerada, para elegir "que punto es" a simple
    # vista en vez de calcular metros.
    w_px = int((length + 2*margin_m)*scale)
    h_px = int((width + 2*margin_m)*scale)
    img = np.full((h_px, w_px, 3), (60, 140, 60), dtype=np.uint8)

    def to_px(point):
        x, y = point
        return (int((x + length/2 + margin_m)*scale), int((y + margin_m)*scale))

    for line in pitch_lines_m(length, width):
        pts = np.array([to_px(p) for p in line], dtype=np.int32)
        cv2.polylines(img, [pts], False, (255, 255, 255), 2)

    for landmark_id, (_, point) in landmarks.items():
        px = to_px(point)
        cv2.circle(img, px, 11, (0, 0, 255), -1)
        cv2.putText(img, str(landmark_id), (px[0]-8, px[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img


def reprojection_errors(points_px, points_m):
    # Homografia con los puntos dados y error de reproyeccion (m) de cada uno: que tan lejos cae
    # el propio punto de calibracion al pasarlo pixeles -> metros -> pixeles otra vez.
    H, _ = cv2.findHomography(np.float32(points_px), np.float32(points_m))
    if H is None:
        return None, None
    proj = cv2.perspectiveTransform(np.float32(points_px).reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(proj - np.float32(points_m), axis=1)
    return H, errors


def camera_to_reference(frame_num):
    # Homografia frame_num -> CALIBRATION_FRAME a partir del stub del movimiento de camara
    with open(config.CAMERA_STUB_PATH, 'rb') as f:
        camera_movement = pickle.load(f)
    H_to_first = np.array(camera_movement[frame_num])
    H_ref_to_first = np.array(camera_movement[config.CALIBRATION_FRAME])
    return np.linalg.inv(H_ref_to_first) @ H_to_first


def draw_check(frame, frame_to_reference=np.eye(3)):
    H, errors = reprojection_errors(config.PITCH_POINTS_PX, config.PITCH_POINTS_M)
    # metros -> pixeles del frame de referencia -> pixeles de este frame
    meters_to_frame = np.linalg.inv(frame_to_reference) @ np.linalg.inv(H)
    out = frame.copy()
    thickness = max(1, round(frame.shape[0]/540))
    for line in pitch_lines_m(config.PITCH_LENGTH_M, config.PITCH_WIDTH_M):
        pts = cv2.perspectiveTransform(np.float32(line).reshape(-1, 1, 2), meters_to_frame)
        cv2.polylines(out, [pts.astype(np.int32)], False, (255, 0, 255), thickness)
    # Cuadricula cada 10 m
    for x in range(-50, 51, 10):
        pts = cv2.perspectiveTransform(np.float32([(x, 0), (x, 68)]).reshape(-1, 1, 2), meters_to_frame)
        cv2.polylines(out, [pts.astype(np.int32)], False, (0, 255, 255), 1)

    if np.allclose(frame_to_reference, np.eye(3)):
        for px in config.PITCH_POINTS_PX:
            cv2.circle(out, tuple(map(int, px)), 4, (0, 0, 255), -1)
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

    diagram = render_landmark_diagram(LANDMARKS, config.PITCH_LENGTH_M, config.PITCH_WIDTH_M)
    print("Marcas disponibles (tambien se ven numeradas en la ventana 'marcas de referencia'):")
    for landmark_id, (name, _) in LANDMARKS.items():
        print(f"  {landmark_id:2d}: {name}")

    cv2.namedWindow('calibracion', cv2.WINDOW_NORMAL)
    cv2.namedWindow('marcas de referencia', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('calibracion', on_click)
    print("\nHaz clic en al menos 4 puntos conocidos de la cancha. Enter para terminar, Esc para cancelar.")
    while True:
        cv2.imshow('calibracion', display)
        cv2.imshow('marcas de referencia', diagram)
        key = cv2.waitKey(20)
        if key == 13 and len(points) >= 4:
            break
        if key == 27:
            sys.exit(0)
    cv2.destroyAllWindows()

    used_ids = set()
    points_m = []
    for i, p in enumerate(points, 1):
        while True:
            raw = input(f"Punto {i} {p} - numero de marca (ver diagrama): ").strip()
            if not raw.isdigit() or int(raw) not in LANDMARKS:
                print(f"  Numero invalido. Elegi uno de: {sorted(LANDMARKS)}")
                continue
            landmark_id = int(raw)
            if landmark_id in used_ids:
                print(f"  Aviso: la marca {landmark_id} ya se uso en otro punto.")
            used_ids.add(landmark_id)
            points_m.append(LANDMARKS[landmark_id][1])
            break

    if len(used_ids) < 4:
        print(f"\nAdvertencia: solo hay {len(used_ids)} marcas distintas (se repitieron algunas)."
              " La calibracion puede no ser confiable.")

    H, errors = reprojection_errors(points, points_m)
    if H is None:
        print("\nNo se pudo calcular la homografia con estos puntos (¿estan muy alineados entre si?).")
    else:
        print("\nError de reproyeccion por punto (m):", np.round(errors, 2))
        if np.any(errors > 2):
            print("Aviso: algun punto tiene error > 2 m. Revisa si clickeaste o elegiste la marca"
                  " equivocada antes de usar esta calibracion.")

    print("\nCopia esto en config.py:\n")
    print("PITCH_POINTS_PX = " + repr(points))
    print("PITCH_POINTS_M = " + repr(points_m))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=config.VIDEO_PATH)
    parser.add_argument('--check', action='store_true', help='Dibujar la cancha proyectada con config.py')
    parser.add_argument('--frame', type=int, default=config.CALIBRATION_FRAME,
                        help='Frame a revisar (distinto de CALIBRATION_FRAME usa el stub de camara)')
    parser.add_argument('--output', default='output_videos/calibration_check.jpg')
    args = parser.parse_args()

    frame = read_frame(args.video, args.frame)
    if args.check:
        to_ref = np.eye(3) if args.frame == config.CALIBRATION_FRAME else camera_to_reference(args.frame)
        cv2.imwrite(args.output, draw_check(frame, to_ref))
        print(f"Imagen guardada en {args.output}")
    else:
        click_points(frame)
