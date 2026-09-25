# Configuracion del pipeline para el clip del Mundial 2026 (CZE vs RSA),
# minuto 3:11-4:30 de la transmision (79 s, 1280x720, 30 fps).
# Si cambias de video, recalibra con: python tools/calibrate_pitch.py

# Rutas
VIDEO_PATH = 'input_videos/wc2026_clip.mp4'
OUTPUT_PATH = 'output_videos/wc2026_clip_output.mp4'
MODEL_PATH = 'models/best.pt'
TRACK_STUB_PATH = 'stubs/wc2026_clip_track_stubs.pkl'
CAMERA_STUB_PATH = 'stubs/wc2026_clip_camera_movement_stub.pkl'

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

# Valla LED perimetral (publicidad con texto que cambia/scrollea, ej. BYJU'S, Hisense): no es un
# overlay de TV, es parte de la escena, pero su contenido se mueve sin que se mueva la camara y
# corrompe el tracking de optical flow para movimiento de camara. Se excluye solo de ese calculo
# (no de la deteccion de jugadores, que ya se filtra aparte con la mascara de pasto).
CAMERA_EXCLUDE_BOXES = OVERLAY_BOXES + [
    (0, 320, 1920, 400),
]

# Deteccion
DETECTION_CONF = 0.1

# Perspectiva: puntos de la cancha en el frame CALIBRATION_FRAME (pixeles reales del video) y su
# posicion real (metros). El movimiento de camara lleva cada frame a las coordenadas de ese frame.
# x = metros desde la linea de medio campo (positivo a la derecha), y = metros desde la banda superior.
# Cancha de 105 x 68 m, circulo central de radio 9.15 m.
CALIBRATION_FRAME = 0
PITCH_POINTS_PX = [(902, 472), (902, 258), (904, 1026), (900, 397), (904, 569), (552, 482), (1250, 474)]
PITCH_POINTS_M = [(0, 34.0), (0, 0), (0, 68), (0, 24.85), (0, 43.15), (-9.15, 34.0), (9.15, 34.0)]
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
DRAW_TRACK_IDS = False
DRAW_SPEED_DISTANCE = False

# Balon: huecos sin deteccion mas largos que esto (frames) no se interpolan; el balon no se dibuja
BALL_MAX_GAP_FRAMES = 20
