# Configuracion del pipeline para el clip del Mundial 2026 (CZE vs RSA),
# minuto 3:11-4:30 de la transmision (79 s, 1280x720, 30 fps).
# Si cambias de video, recalibra con: python tools/calibrate_pitch.py

# Rutas
VIDEO_PATH = 'input_videos/wc2026_clip.mp4'
OUTPUT_PATH = 'output_videos/wc2026_output.mp4'
MODEL_PATH = 'models/best.pt'
TRACK_STUB_PATH = 'stubs/wc2026_track_stubs.pkl'
CAMERA_STUB_PATH = 'stubs/wc2026_camera_movement_stub.pkl'

# Maximo de segundos que se cargan en memoria (read_video guarda todos los frames en RAM:
# ~2.7 MB por frame a 720p, ~6 MB a 1080p). 79 s a 720p son ~6.5 GB; Colab tiene ~12 GB.
MAX_VIDEO_SECONDS = 90

# Todas las cajas (x1, y1, x2, y2) estan en pixeles de 1920x1080 y se escalan a la resolucion real.
# Overlays de la transmision:
# marcador de arriba a la izquierda y logo de DSports. Se ignoran para deteccion y optical flow.
OVERLAY_BOXES = [
    (85, 50, 705, 112),
    (1565, 40, 1840, 112),
]

# Deteccion
DETECTION_CONF = 0.1

# Perspectiva: puntos de la cancha en el frame CALIBRATION_FRAME (pixeles reales del video) y su
# posicion real (metros). El movimiento de camara lleva cada frame a las coordenadas de ese frame.
# x = metros desde la linea de medio campo (positivo a la derecha), y = metros desde la banda superior.
# Cancha de 105 x 68 m, circulo central de radio 9.15 m.
CALIBRATION_FRAME = 150
PITCH_POINTS_PX = [
    (572, 115),   # medio campo / banda superior
    (577, 662),   # medio campo / banda inferior
    (573, 219),   # circulo central, arriba
    (574, 339),   # circulo central, abajo
    (327, 280),   # circulo central, izquierda
    (817, 274),   # circulo central, derecha
]
PITCH_POINTS_M = [
    (0, 0),
    (0, 68),
    (0, 34 - 9.15),
    (0, 34 + 9.15),
    (-9.15, 34),
    (9.15, 34),
]
PITCH_LENGTH_M = 105
PITCH_WIDTH_M = 68
# Se dibuja/analiza solo a quien este dentro de la cancha + este margen (m). Los jueces de linea
# andan a ~1.5 m de la banda; stewards y fotografos detras de la valla quedan a 4 m o mas
PITCH_MARGIN_M = 3

# Velocidades por encima de esto se consideran error de medicion y no se dibujan
MAX_PLAYER_SPEED_KMH = 40

# Posesion: distancia maxima (px del video) entre el pie del jugador y el balon
MAX_PLAYER_BALL_DISTANCE = 50

# Cuadros de texto sobre el video (en pixeles de 1920x1080)
BALL_CONTROL_BOX = (1350, 870, 1900, 990)
CAMERA_MOVEMENT_BOX = (0, 125, 640, 225)

# Un track de "arbitro" pasa a jugador si su camiseta esta mucho mas cerca de un equipo que del otro:
# distancia al equipo mas cercano < REFEREE_TEAM_COLOR_RATIO * distancia al otro equipo
REFEREE_TEAM_COLOR_RATIO = 0.3

# Que dibujar en el video de salida (se puede cambiar con --ids/--no-ids y --speed/--no-speed)
DRAW_TRACK_IDS = True
DRAW_SPEED_DISTANCE = True

# Balon: huecos sin deteccion mas largos que esto (frames) no se interpolan; el balon no se dibuja
BALL_MAX_GAP_FRAMES = 20
