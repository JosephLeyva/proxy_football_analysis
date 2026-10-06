# Taller: Visión por computadora con YOLO aplicada al fútbol ⚽

Analizamos una toma abierta continua de ~15 s (387 frames a 1080p) de la **final del Mundial Qatar 2022, Argentina vs Francia**. Detectamos jugadores, árbitros y balón con YOLO, los seguimos con ByteTrack, asignamos equipos con K-Means, calculamos la posesión y, como bonus, medimos la velocidad de cada jugador en km/h.

**Duración:** ~2 h · **Nivel:** intermedio (saber Python; no hace falta experiencia en visión)

## Qué necesitas
- Una cuenta de Google (para usar Colab)
- Navegador; no hay que instalar nada

## Dos versiones del notebook
- `taller_futbol_yolo_alumno.ipynb`: la que usan los alumnos, con 7 huecos ✏️ (pista, verificación y solución desplegable en cada uno).
- `taller_futbol_yolo.ipynb`: la del instructor, completa. Después de editarla, regenera la del alumno con `python tools/make_student_notebook.py`.

## Cómo empezar
1. Abre el notebook en Colab: `<link al notebook en Colab>`
2. `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4`
3. Ejecuta las celdas en orden (`Shift + Enter`)

Si la GPU va lenta o no hay GPU disponible, deja `USE_STUBS = True`: el notebook carga los resultados de YOLO precalculados y todo lo demás corre igual.

## Agenda

| # | Tema | Tiempo |
|---|---|---|
| 0 | Preparación | 10 min |
| 1 | YOLO preentrenado (COCO): qué ve y qué no | 15 min |
| 2 | YOLO especializado en fútbol (fine-tuning) | 15 min |
| 3 | Tracking con ByteTrack | 20 min |
| 4 | Equipos con K-Means | 20 min |
| 5 | Balón: interpolación y posesión | 15 min |
| 6 | Bonus: movimiento de cámara, homografía y velocidad | 15 min |
| 7 | Retos y cierre | 10 min |

## Retos para después
- Mapa de calor de un jugador
- Radar 2D animado junto a la transmisión
- Analizar otro clip (`tools/find_wide_shots.py` + `tools/calibrate_pitch.py`)
- Entrenar tu propio YOLO11 con el dataset de Roboflow
- Asignar los porteros al equipo correcto

## Créditos
- Código base: [abdullahtarek/football_analysis](https://github.com/abdullahtarek/football_analysis)
- Dataset: [Roboflow football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
- [Ultralytics YOLO](https://docs.ultralytics.com) · [supervision](https://supervision.roboflow.com)
- Video: transmisión de TVP, final de la Copa Mundial de la FIFA Qatar 2022, usado con fines educativos
