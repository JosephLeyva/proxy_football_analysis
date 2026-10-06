"""Genera la version del alumno del notebook del taller, con huecos para llenar.

    python tools/make_student_notebook.py

Lee taller/taller_futbol_yolo.ipynb (version del instructor, la "solucion") y escribe
taller/taller_futbol_yolo_alumno.ipynb. Cada hueco reemplaza unas lineas de una celda por `_____`
(5 guiones bajos: `___` no sirve porque Jupyter lo usa para guardar la antepenúltima salida)
y agrega: instrucciones + pista antes, una celda de verificacion y la solucion desplegable despues.

Si cambias el notebook del instructor, vuelve a correr este script. Si una celda cambio tanto que
un hueco ya no se encuentra, el script se detiene diciendo cual: ajusta su `solucion` aqui abajo.
"""
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "taller", "taller_futbol_yolo.ipynb")
DST = os.path.join(ROOT, "taller", "taller_futbol_yolo_alumno.ipynb")

# Cada hueco: `solucion` debe existir tal cual en una celda de codigo del notebook del instructor;
# `hueco` es lo que ve el alumno en su lugar.
HUECOS = [
    dict(
        titulo="Carga YOLO y haz tu primera detección",
        instrucciones="Carga el modelo **nano** de YOLO11 (entrenado con COCO) y pásale el **primer frame** del video.",
        pista="El modelo se llama `\"yolo11n.pt\"` (con comillas: es un nombre de archivo). Los frames están en la lista `frames`; el primero es el índice 0.",
        solucion='coco_model = YOLO("yolo11n.pt")          # n = nano, el mas pequeño y rapido\n'
                 'result = coco_model.predict(frames[0], conf=0.25, verbose=False)[0]',
        hueco='coco_model = YOLO(_____)          # ✏️ HUECO 1a: el modelo nano de YOLO11\n'
              'result = coco_model.predict(_____, conf=0.25, verbose=False)[0]   # ✏️ HUECO 1b: el primer frame del video',
        verificacion='lambda: len(result.boxes) > 0 and "person" in result.names.values()',
        si_falla="¿Usaste el nombre del modelo entre comillas y frames[0]?",
    ),
    dict(
        titulo="Carga el modelo especializado en fútbol",
        instrucciones="Ahora carga nuestro YOLO reentrenado con imágenes de fútbol. Su ruta ya está guardada en la configuración del proyecto.",
        pista="Busca en `config` una variable con la ruta del modelo: `config.MODEL_PATH`.",
        solucion="model = YOLO(config.MODEL_PATH)",
        hueco="model = YOLO(_____)   # ✏️ HUECO 2: la ruta del modelo de fútbol",
        verificacion='lambda: set(model.names.values()) == {"ball", "goalkeeper", "player", "referee"}',
        si_falla="El modelo correcto tiene 4 clases: ball, goalkeeper, player y referee.",
    ),
    dict(
        titulo="Cuenta jugadores e IDs",
        instrucciones=("`tracks[\"players\"]` es una **lista** con un elemento por frame. Cada elemento es un "
                       "**diccionario** `{track_id: datos_del_jugador}`. Calcula:\n"
                       "- `unique_ids`: el conjunto de todos los IDs que aparecen en el video.\n"
                       "- `per_frame`: una lista con cuántos jugadores hay en cada frame."),
        pista="Recorrer un diccionario con `for tid in frame` te da sus llaves (los IDs). `len(f)` dice cuántas llaves tiene `f`.",
        solucion='unique_ids = {tid for frame in tracks["players"] for tid in frame}\n'
                 'per_frame = [len(f) for f in tracks["players"]]',
        hueco='unique_ids = {_____ for frame in tracks["players"] for tid in frame}   # ✏️ HUECO 3a: el ID de cada track\n'
              'per_frame = [_____ for f in tracks["players"]]                          # ✏️ HUECO 3b: cuántos jugadores hay en el frame f',
        verificacion=('lambda: len(per_frame) == len(frames) and per_frame[0] == len(tracks["players"][0]) '
                      'and unique_ids == {t for fr in tracks["players"] for t in fr}'),
        si_falla="unique_ids debe tener los IDs (tid) y per_frame un número por frame.",
    ),
    dict(
        titulo="Recorta la camiseta y agrupa sus colores",
        instrucciones=("Recorta la caja del jugador del frame, quédate con la **mitad superior** (la camiseta) "
                       "y agrupa sus píxeles con K-Means."),
        pista=("Una imagen en NumPy se indexa `imagen[filas, columnas]`, es decir `[y, x]`: `frames[0][y1:y2, x1:x2]`. "
               "La altura del recorte es `crop.shape[0]`; la mitad es `crop.shape[0]//2`. "
               "¿Cuántos colores esperas en la mitad superior? Camiseta y pasto."),
        solucion="crop = frames[0][y1:y2, x1:x2]\n"
                 "top_half = crop[: crop.shape[0]//2]\n"
                 "\n"
                 "kmeans = KMeans(n_clusters=2, n_init=1).fit(top_half.reshape(-1, 3))",
        hueco="crop = frames[0][_____, _____]     # ✏️ HUECO 4a: recorta la caja: primero filas (y), luego columnas (x)\n"
              "top_half = crop[: _____]         # ✏️ HUECO 4b: solo la mitad superior (la camiseta)\n"
              "\n"
              "kmeans = KMeans(n_clusters=_____, n_init=1).fit(top_half.reshape(-1, 3))   # ✏️ HUECO 4c: ¿cuántos grupos de color?",
        verificacion=("lambda: crop.shape[:2] == (y2 - y1, x2 - x1) and top_half.shape[0] == crop.shape[0] // 2 "
                      "and kmeans.n_clusters == 2"),
        si_falla="Revisa el orden [y, x] del recorte, la mitad con // y el número de clusters.",
    ),
    dict(
        titulo="Guarda el equipo de cada jugador",
        instrucciones=("`get_player_team` ya te dice el equipo (1 o 2) del jugador en ese frame. "
                       "Guárdalo en el track junto con el color del equipo, para poder dibujarlo."),
        pista="`team_assigner.team_colors` es un diccionario `{equipo: color}`: el color del equipo `team` es `team_assigner.team_colors[team]`.",
        solucion='        track["team"] = team\n'
                 '        track["team_color"] = team_assigner.team_colors[team]',
        hueco='        track["team"] = _____                                   # ✏️ HUECO 5a: el equipo del jugador\n'
              '        track["team_color"] = team_assigner.team_colors[_____]  # ✏️ HUECO 5b: el color de ese equipo',
        verificacion=('lambda: all(p["team"] in (1, 2) and (p["team_color"] == team_assigner.team_colors[p["team"]]).all() '
                      'for f in tracks["players"] for p in f.values())'),
        si_falla="Cada jugador debe tener team (1 o 2) y el team_color de ese equipo.",
    ),
    dict(
        titulo="¿Quién tiene el balón?",
        instrucciones=("Completa la lógica de posesión. Para cada frame agregamos a `control` el equipo que tiene el balón:\n"
                       "- Si hay un jugador cerca del balón (`assigned != -1`), es **su** equipo.\n"
                       "- Si nadie está cerca (pase en el aire), se queda el **último** equipo que lo tuvo; "
                       "si todavía no lo ha tenido nadie, ponemos `0`."),
        pista=("El equipo del jugador asignado está en `player_track[assigned][\"team\"]`. "
               "El último elemento de una lista es `control[-1]`; para el caso vacío usa `control[-1] if control else 0`."),
        solucion='            control.append(player_track[assigned]["team"])\n'
                 '        else:\n'
                 '            control.append(control[-1] if control else 0)',
        hueco='            control.append(_____)   # ✏️ HUECO 6a: el equipo del jugador que tiene el balón\n'
              '        else:\n'
              '            control.append(_____)   # ✏️ HUECO 6b: el último equipo que lo tuvo (o 0 si la lista está vacía)',
        verificacion='lambda: _check_posesion()',
        si_falla="Cuando nadie tiene el balón, el valor debe ser el del frame anterior.",
        preparacion='''def _check_posesion():
    c = list(team_ball_control)
    if len(c) != len(frames) or not set(c) <= {0, 1, 2}:
        return False
    for i, f in enumerate(tracks["players"]):
        dueño = [p["team"] for p in f.values() if p.get("has_ball")]
        esperado = dueño[0] if dueño else (c[i - 1] if i else 0)
        if c[i] != esperado:
            return False
    return True''',
    ),
    dict(
        titulo="Ciencia de datos: estadísticas por jugador",
        instrucciones=("Con pandas, agrupa las filas **por jugador** (`id`) y calcula su **velocidad máxima**. "
                       "La distancia ya está calculada: como es *acumulada*, su máximo es el total recorrido."),
        pista="`groupby` recibe el nombre de la columna por la que agrupas (`\"id\"`). Para el máximo, `agg` acepta el texto `\"max\"`.",
        solucion='stats = pd.DataFrame(rows).groupby("id").agg(equipo=("equipo", "first"),\n'
                 '                                             vel_max=("km/h", "max"),',
        hueco='stats = pd.DataFrame(rows).groupby(_____).agg(equipo=("equipo", "first"),   # ✏️ HUECO 7a: agrupa por jugador\n'
              '                                             vel_max=("km/h", _____),     # ✏️ HUECO 7b: la velocidad máxima',
        verificacion=('lambda: stats.index.name == "id" and '
                      '(stats["vel_max"] == pd.DataFrame(rows).groupby("id")["km/h"].max()).all()'),
        si_falla="Agrupa por \"id\" y usa \"max\" para la velocidad.",
    ),
]

AYUDA_REVISAR = '''

# Verificación de los huecos ✏️ (no la modifiques)
def revisar(n, condicion, si_falla=""):
    try:
        ok = bool(condicion())
    except NameError as e:
        print(f"⚠️ Hueco {n}: falta ejecutar la celda del hueco (o algo quedó sin llenar): {e}")
        return
    except Exception as e:
        ok = False
        si_falla = f"{si_falla} (error: {type(e).__name__}: {e})"
    print(f"✅ Hueco {n}: ¡correcto!" if ok else f"❌ Hueco {n}: todavía no. {si_falla}")'''

INTRO = """## ✏️ Cómo usar esta versión

Este notebook tiene **7 huecos** marcados con ✏️ donde tú escribes el código (son de 1 a 3 líneas).

1. Lee la instrucción y reemplaza cada `_____` por tu respuesta.
2. Ejecuta la celda y luego la de **verificación** que está debajo: te dirá ✅ o ❌.
3. ¿Atorado? Abre la **💡 pista**. Si sigue sin salir, abre **👀 Ver solución** y cópiala: lo importante es entenderla.

> Si ves `NameError: name '_____' is not defined`, te falta llenar un hueco de esa celda.

> 💾 **Antes de empezar:** `Archivo → Guardar una copia en Drive`. Así te quedas con tu versión al terminar."""


def lines(text):
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(text)}


def main():
    with open(SRC, encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]

    # Ubicar cada hueco
    for n, h in enumerate(HUECOS, 1):
        matches = [i for i, c in enumerate(cells)
                   if c["cell_type"] == "code" and h["solucion"] in "".join(c["source"])]
        if len(matches) != 1:
            sys.exit(f"Hueco {n} ({h['titulo']}): la solución aparece en {len(matches)} celdas (debe ser 1). "
                     "¿Cambió el notebook del instructor? Ajusta 'solucion' en este script.")
        h["celda"] = matches[0]
    por_celda = {h["celda"]: (n, h) for n, h in enumerate(HUECOS, 1)}

    helper_cell = next(i for i, c in enumerate(cells) if "def show_video" in "".join(c["source"]))

    out = []
    for i, cell in enumerate(cells):
        cell = copy.deepcopy(cell)
        src = "".join(cell["source"])
        if i == 0:
            src = src.replace("\n", " — versión del alumno ✏️\n", 1)
            cell["source"] = lines(src)
            out += [cell, md(INTRO)]
            continue
        if i == helper_cell:
            cell["source"] = lines(src.rstrip("\n") + AYUDA_REVISAR)
            out.append(cell)
            continue
        if i not in por_celda:
            out.append(cell)
            continue

        n, h = por_celda[i]
        cell["source"] = lines(src.replace(h["solucion"], h["hueco"]))
        out.append(md(f"### ✏️ Hueco {n}: {h['titulo']}\n\n{h['instrucciones']}\n\n"
                      f"<details><summary>💡 Pista</summary>\n\n{h['pista']}\n\n</details>"))
        out.append(cell)
        check = (h["preparacion"] + "\n\n") if "preparacion" in h else ""
        check += f"# Verificación del hueco {n}\nrevisar({n}, {h['verificacion']}, {h['si_falla']!r})"
        out.append(code(check))
        out.append(md(f"<details><summary>👀 Ver solución del hueco {n}</summary>\n\n"
                      f"```python\n{h['solucion'].strip(chr(10))}\n```\n\n</details>"))

    for c in out:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    nb["cells"] = out
    with open(DST, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{len(HUECOS)} huecos -> {os.path.relpath(DST, ROOT)} ({len(out)} celdas)")


if __name__ == "__main__":
    main()
