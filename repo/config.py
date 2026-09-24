# Configuracion del pipeline para el clip del Mundial 2026 (CZE vs RSA).
# Si cambias de video, recalibra con: python tools/calibrate_pitch.py

# Rutas
VIDEO_PATH = 'input_videos/wc2026_clip.mp4'
OUTPUT_PATH = 'output_videos/wc2026_output.mp4'
MODEL_PATH = 'models/best.pt'
TRACK_STUB_PATH = 'stubs/wc2026_track_stubs.pkl'
CAMERA_STUB_PATH = 'stubs/wc2026_camera_movement_stub.pkl'

# Maximo de segundos que se cargan en memoria (read_video guarda todos los frames en RAM)
MAX_VIDEO_SECONDS = 60

# Overlays de la transmision (x1, y1, x2, y2) en pixeles de un frame de 1920x1080:
# marcador de arriba a la izquierda y logo de DSports. Se ignoran para deteccion y optical flow.
OVERLAY_BOXES = [
    (85, 50, 705, 112),
    (1565, 40, 1840, 112),
]

# Deteccion
DETECTION_CONF = 0.1

# Perspectiva: puntos de la cancha en el primer frame del clip (pixeles) y su posicion real (metros).
# x = metros desde la linea de medio campo (positivo a la derecha), y = metros desde la banda superior.
# Cancha de 105 x 68 m, circulo central de radio 9.15 m.
PITCH_POINTS_PX = [
    (901, 251),   # medio campo / banda superior
    (905, 1029),  # medio campo / banda inferior
    (902, 397),   # circulo central, arriba
    (903, 567),   # circulo central, abajo
    (551, 485),   # circulo central, izquierda
    (1249, 478),  # circulo central, derecha
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

# Velocidades por encima de esto se consideran error de medicion y no se dibujan
MAX_PLAYER_SPEED_KMH = 40

# Posesion: distancia maxima (px) entre el pie del jugador y el balon
MAX_PLAYER_BALL_DISTANCE = 70

# Cuadros de texto sobre el video (x1, y1, x2, y2)
BALL_CONTROL_BOX = (1350, 870, 1900, 990)
CAMERA_MOVEMENT_BOX = (0, 125, 500, 225)
