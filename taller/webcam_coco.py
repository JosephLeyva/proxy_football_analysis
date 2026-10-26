"""Demo en vivo: YOLO preentrenado en COCO (80 clases) sobre la webcam.

Uso (desde la raiz del repo):
    python taller/webcam_coco.py                      # webcam 0, yolo11n.pt
    python taller/webcam_coco.py --source 1           # otra camara
    python taller/webcam_coco.py --source input_videos/clip.mp4
    python taller/webcam_coco.py --classes person "sports ball"
    python taller/webcam_coco.py --list-classes # lista las 80 clases y sale
    python taller/webcam_coco.py --model yolo11s.pt --conf 0.4
    python taller/webcam_coco.py --source input_videos/clip.mp4 --all-frames  # sin saltar cuadros

Teclas: q / ESC salir · espacio pausa · s captura en output_videos/
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="YOLO COCO en tiempo real")
    p.add_argument("--model", default=str(ROOT / "yolo11n.pt"),
                   help="pesos COCO (yolo11n/s/m/l/x.pt se descargan solos)")
    p.add_argument("--source", default="0",
                   help="indice de camara (0, 1, ...), ruta de video o URL")
    p.add_argument("--conf", type=float, default=0.25, help="confianza minima")
    p.add_argument("--imgsz", type=int, default=640, help="tamano de entrada")
    p.add_argument("--device", default=None, help="cpu, 0 (GPU), mps... por defecto auto")
    p.add_argument("--classes", nargs="+", default=None,
                   help='solo estas clases por nombre, p. ej. person "sports ball"')
    p.add_argument("--list-classes", action="store_true", help="imprime las 80 clases y sale")
    p.add_argument("--max-width", type=int, default=1280,
                   help="reduce cuadros mas anchos que esto antes de procesar (0 = no reducir)")
    p.add_argument("--all-frames", action="store_true",
                   help="en videos, procesa todos los cuadros (se ve en camara lenta) "
                        "en vez de saltar cuadros para ir a velocidad real")
    return p.parse_args()


def open_source(source):
    if source.isdigit():
        # En Windows DirectShow abre la webcam mucho mas rapido que el backend por defecto
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(source), backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir la fuente '{source}'. Prueba otro --source (0, 1, ...).")
    return cap


def class_ids_from_names(model, names):
    name_to_id = {v: k for k, v in model.names.items()}
    unknown = [n for n in names if n not in name_to_id]
    if unknown:
        sys.exit(f"Clases desconocidas: {unknown}. Usa --list-classes para ver las validas.")
    return [name_to_id[n] for n in names]


def limit_width(frame, max_width):
    h, w = frame.shape[:2]
    if not max_width or w <= max_width:
        return frame
    return cv2.resize(frame, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)


def draw_panel(frame, fps, counts):
    lines = [f"FPS: {fps:.1f}", f"Objetos: {sum(counts.values())}"]
    lines += [f"{name}: {n}" for name, n in counts.most_common(8)]
    h = 28 * len(lines) + 12
    # Oscurecer solo el recuadro del panel (55 % negro), no una copia del cuadro entero
    frame[8:8 + h, 8:260] = cv2.convertScaleAbs(frame[8:8 + h, 8:260], alpha=0.45)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (18, 38 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    model = YOLO(args.model)

    if args.list_classes:
        for k, v in model.names.items():
            print(f"{k:2d}  {v}")
        return

    classes = class_ids_from_names(model, args.classes) if args.classes else None
    # La primera inferencia tarda varios segundos (inicializacion de PyTorch); hacerla
    # antes de empezar, para que no se coma el inicio del video en modo tiempo real
    model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=args.imgsz,
                  device=args.device, verbose=False)
    cap = open_source(args.source)
    out_dir = ROOT / "output_videos"
    window = "YOLO COCO - q para salir"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Un video se lee tan rapido como YOLO lo procese: si YOLO va a 10 FPS y el video
    # es de 30 FPS, se veria a 1/3 de velocidad. En modo tiempo real se saltan los cuadros
    # que "pasaron" mientras se procesaba el anterior, como ocurre de forma natural con la webcam.
    realtime = not args.source.isdigit() and not args.all_frames
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    next_idx = 0  # indice del proximo cuadro del video por leer
    t0 = time.perf_counter()  # instante en que el video "empezo" a reproducirse

    fps, prev, paused, paused_at, annotated = 0.0, time.perf_counter(), False, 0.0, None
    while True:
        if not paused or annotated is None:
            if realtime:
                due = int((time.perf_counter() - t0) * video_fps)
                while next_idx < due and cap.grab():  # grab descarta sin convertir a imagen
                    next_idx += 1
            ok, frame = cap.read()
            if not ok:
                break
            next_idx += 1
            frame = limit_width(frame, args.max_width)
            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                   device=args.device, classes=classes, verbose=False)[0]
            annotated = result.plot()
            counts = Counter(model.names[int(c)] for c in result.boxes.cls)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))  # FPS suavizado
            prev = now
            draw_panel(annotated, fps, counts)

        cv2.imshow(window, annotated)
        delay = 1
        if realtime and not paused:
            # Si YOLO va mas rapido que el video (p. ej. con GPU), esperar para no adelantarse
            delay = max(1, int((next_idx / video_fps - (time.perf_counter() - t0)) * 1000))
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            if paused:
                paused_at = time.perf_counter()
            else:
                t0 += time.perf_counter() - paused_at  # el tiempo en pausa no cuenta
            prev = time.perf_counter()
        if key == ord("s"):
            out_dir.mkdir(exist_ok=True)
            path = out_dir / f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(path), annotated)
            print(f"Captura guardada: {path}")
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
