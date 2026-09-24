# Football Analysis Project

> **Versión adaptada para el taller "Visión con YOLO aplicada al fútbol"** (clip del Mundial 2026, CZE vs RSA).
> Notebook del taller: [`taller/taller_futbol_yolo.ipynb`](taller/taller_futbol_yolo.ipynb) · Guía: [`taller/README.md`](taller/README.md)
>
> ```bash
> pip install -r requirements.txt
> python main.py              # usa los stubs de stubs/ si existen
> python main.py --recompute  # vuelve a correr YOLO y optical flow
> python tools/calibrate_pitch.py --check   # verifica la homografia
> ```
>
> Cambios respecto al repo original:
> - `config.py` centraliza rutas, calibración de la cancha y overlays de la transmisión (marcador, logo)
> - Detecciones fuera de la cancha se filtran con una máscara de pasto; la clase jugador/árbitro se decide por mayoría en cada track
> - Equipos: K-Means con jugadores de 10 frames y voto por jugador (se quitó el caso fijo `player_id == 91`)
> - Movimiento de cámara: mediana del optical flow, acumulado respecto al frame 0 (el original usaba el desplazamiento de un solo frame)
> - Perspectiva: homografía con `cv2.findHomography` a partir de puntos de la cancha configurables (`tools/calibrate_pitch.py`)
> - Velocidad con el FPS real del video (antes fijo en 24) y descarte de velocidades imposibles
> - Los dibujos se hacen sobre los frames sin copiarlos (el original triplicaba la RAM) y el video sale en H.264

## Introduction
The goal of this project is to detect and track players, referees, and footballs in a video using YOLO, one of the best AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, we can measure a team's ball acquisition percentage in a match. We will also use optical flow to measure camera movement between frames, enabling us to accurately measure a player's movement. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, allowing us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered. This project covers various concepts and addresses real-world problems, making it suitable for both beginners and experienced machine learning engineers.

![Screenshot](output_videos/screenshot.png)

## Modules Used
The following modules are used in this project:
- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## Trained Models
- [Trained Yolo v5](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

## Sample video
-  [Sample input video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

## Requirements
To run this project, you need to have the following requirements installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas