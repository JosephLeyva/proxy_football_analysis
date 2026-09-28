# Guía de usuario — Football Analysis

Guía práctica para instalar, configurar y correr el pipeline de análisis de video de fútbol de este repositorio.

## 1. ¿Qué hace este proyecto?

A partir de un clip de video de un partido de fútbol, el pipeline:

1. Detecta jugadores, árbitros y balón con YOLO.
2. Sigue (trackea) cada objeto entre frames con ByteTrack.
3. Asigna cada jugador a un equipo según el color de su camiseta (K-Means).
4. Interpola la posición del balón cuando no se detecta y calcula qué equipo tiene la posesión.
5. Estima el movimiento de la cámara (optical flow) para separar el movimiento real del jugador del movimiento de la toma.
6. Convierte posiciones de píxeles a metros con una homografía calibrada de la cancha.
7. Calcula velocidad (km/h) y distancia recorrida por jugador, usando el FPS real del video.
8. Dibuja todo (cajas, IDs, posesión, velocidad) sobre el video de salida.

## 2. Requisitos e instalación

```bash
pip install -r requirements.txt
```

Dependencias principales: `ultralytics` (YOLO), `supervision` (ByteTrack), `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `gdown`, `scenedetect`.

## 3. Estructura relevante del repo

| Ruta | Para qué sirve |
|---|---|
| `main.py` | Punto de entrada; corre todo el pipeline en orden. |
| `config.py` | **Toda la configuración**: rutas de video/modelo/stubs, calibración de la cancha, umbrales, overlays. |
| `input_videos/` | Videos de entrada. |
| `output_videos/` | Video anotado de salida. |
| `models/` | Pesos del modelo YOLO (`.pt`). |
| `stubs/` | Cachés `.pkl` de tracking y movimiento de cámara ya calculados (evitan recorrer YOLO/optical flow otra vez). |
| `trackers/` | Detección + tracking (YOLO + ByteTrack), dibujo de anotaciones. |
| `camera_movement_estimator/` | Movimiento de cámara por optical flow. |
| `view_transformer/` | Homografía píxeles → metros. |
| `speed_and_distance_estimator/` | Velocidad y distancia por jugador. |
| `team_assigner/` | Asignación de equipo por color de camiseta (K-Means). |
| `player_ball_assigner/` | A qué jugador pertenece el balón en cada frame. |
| `utils/` | Lectura/escritura de video, utilidades de bounding boxes, máscara de la cancha. |
| `tools/calibrate_pitch.py` | Calibra los puntos de la cancha (píxeles ↔ metros) para un video nuevo. |
| `tools/find_wide_shots.py` | Ayuda a encontrar tomas abiertas (wide shots) útiles para calibrar. |
| `taller/` | Material del taller "Visión con YOLO aplicada al fútbol" (notebook + guía en Colab). |
| `training/` | Notebook de entrenamiento de un YOLO propio. |

## 4. Uso básico

Correr el pipeline con la configuración actual (usa los stubs si existen, para ir más rápido):

```bash
python main.py
```

Forzar que se vuelva a correr YOLO y el optical flow (ignora los `.pkl` en `stubs/`):

```bash
python main.py --recompute
```

El video anotado queda en la ruta definida por `OUTPUT_PATH` en `config.py` (por defecto `output_videos/fifa_31s_output.mp4`).

## 5. Cómo usar tu propio video

1. Copia tu video a `input_videos/`.
2. En `config.py`, actualiza:
   - `VIDEO_PATH`: ruta al nuevo video.
   - `TRACK_STUB_PATH` y `CAMERA_STUB_PATH`: dale nombres nuevos (o bórralos) para que no reutilice cachés de otro video.
   - `OVERLAY_BOXES`: ajusta o vacía si tu transmisión tiene marcador/logo en otra posición (o ninguno).
3. **Recalibra la cancha** (obligatorio si cambia el ángulo de cámara — ver sección 6).
4. Corre `python main.py --recompute` la primera vez (no hay stubs todavía para este video).

## 6. Calibración de la cancha (píxeles → metros)

La conversión de posiciones a metros depende de puntos de referencia específicos de **ese** ángulo de cámara. Si cambias de video hay que recalibrar.

**Elegir puntos y generar la calibración:**

```bash
python tools/calibrate_pitch.py
```

Esto abre el primer frame del video, permite hacer clic en al menos 4 puntos conocidos de la cancha (por ejemplo las esquinas del círculo central o la línea de medio campo) e ingresar sus coordenadas reales en metros. Al final imprime los valores listos para copiar en `config.py` (`PITCH_POINTS_PX` y `PITCH_POINTS_M`).

**Verificar que la calibración quedó bien:**

```bash
python tools/calibrate_pitch.py --check
```

Genera una imagen (`output_videos/calibration_check.jpg` por defecto) con las líneas de la cancha proyectadas y una cuadrícula cada 10 m sobre el primer frame, además de imprimir el error de reproyección (en metros) de cada punto. Si las líneas no coinciden con la cancha real, o el error es grande, vuelve a calibrar con puntos más precisos.

Sistema de coordenadas en metros usado en `config.py`:
- `x`: distancia a la línea de medio campo (positivo hacia la derecha).
- `y`: distancia a la banda superior.
- Cancha estándar: 105 x 68 m, círculo central de radio 9.15 m.

## 7. Parámetros de configuración más usados (`config.py`)

| Parámetro | Qué controla |
|---|---|
| `MAX_VIDEO_SECONDS` | Cuántos segundos del video se cargan en memoria (todo el video se guarda en RAM). Bajar este valor si falta memoria. |
| `DETECTION_CONF` | Confianza mínima de YOLO para aceptar una detección. |
| `MAX_PLAYER_SPEED_KMH` | Velocidades por encima de este valor se consideran error de medición y no se dibujan. |
| `MAX_PLAYER_BALL_DISTANCE` | Distancia máxima (px) entre el pie del jugador y el balón para considerarlo en posesión. |
| `BALL_CONTROL_BOX`, `CAMERA_MOVEMENT_BOX` | Posición de los cuadros de texto (posesión, movimiento de cámara) dibujados sobre el video. |
| `OVERLAY_BOXES` | Zonas del frame (marcador, logo) que se ignoran para detección y optical flow. |

## 8. Sobre los stubs (caché)

Correr YOLO y el optical flow sobre el video completo es lento. Los resultados se cachean en `stubs/*.pkl`:

- `TRACK_STUB_PATH`: tracks de YOLO/ByteTrack.
- `CAMERA_STUB_PATH`: movimiento de cámara por frame.

Por defecto `python main.py` reutiliza estos archivos si existen. Usa `--recompute` cuando:
- Cambiaste de video.
- Cambiaste el modelo (`MODEL_PATH`) o `DETECTION_CONF`.
- Sospechas que el `.pkl` quedó corrupto o desactualizado.

## 9. Notas y limitaciones conocidas

- Todo el video se carga en memoria como lista de frames; videos largos pueden agotar la RAM (por eso existe `MAX_VIDEO_SECONDS`).
- La calibración de la cancha (`PITCH_POINTS_PX`) es válida solo mientras la cámara no cambie de posición/zoom dentro del clip.
- El equipo del arquero no se asigna automáticamente por color (queda como reto pendiente, ver `taller/README.md`).
- El video de salida se codifica en H.264.

## 10. Material del taller

Si estás siguiendo el taller "Visión con YOLO aplicada al fútbol", el notebook y la agenda están en [`taller/taller_futbol_yolo.ipynb`](../taller/taller_futbol_yolo.ipynb) y [`taller/README.md`](../taller/README.md).
