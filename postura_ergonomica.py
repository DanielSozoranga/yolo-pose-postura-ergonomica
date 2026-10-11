"""Monitor de ergonomia postural con YOLO Pose (usuario sentado frente al computador).

Flujo: 'c' calibra tu postura correcta -> el sistema compara cada frame contra ella
y avisa si la mala postura se mantiene varios segundos. Salir: 'q'.

Uso (en Windows, con la camara conectada):
    python postura_ergonomica.py
"""
import math
import time

import cv2
from ultralytics import YOLO

try:
    import winsound  # beep de alerta (solo Windows)
except ImportError:
    winsound = None

MODELO = "yolo26n-pose.pt"   # modelo de pose pre-entrenado en COCO Keypoints
CONF_MIN = 0.5               # confianza minima para creerle a un keypoint
MUESTRAS_CALIBRACION = 30    # frames promediados para fijar la postura correcta
SEGUNDOS_ALERTA = 3          # tiempo de mala postura antes de alertar

# Tolerancias respecto a TU postura calibrada
TOL_HOMBROS = 6      # grados de inclinacion de hombros
TOL_CABEZA = 10      # grados de inclinacion de cabeza (linea de los ojos)
TOL_ENCORVADO = 0.80 # la cabeza no debe hundirse por debajo del 80% de la altura calibrada
TOL_CERCA = 1.20     # los hombros no deben verse mas de 20% mas anchos (te acercaste)

# Indices de keypoints del formato COCO (17 puntos)
NARIZ, OJO_IZQ, OJO_DER, HOMBRO_IZQ, HOMBRO_DER = 0, 1, 2, 5, 6
CLAVES = (NARIZ, OJO_IZQ, OJO_DER, HOMBRO_IZQ, HOMBRO_DER)

VERDE, ROJO, AMARILLO = (0, 200, 0), (0, 0, 255), (0, 220, 255)


def inclinacion(a, b):
    """Angulo en grados de la recta a-b respecto a la horizontal, en (-90, 90)."""
    if a[0] > b[0]:  # siempre de izquierda a derecha en la imagen
        a, b = b, a
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def medir(xy, conf):
    """Convierte los keypoints de una persona en 4 metricas posturales (o None)."""
    if min(conf[i] for i in CLAVES) < CONF_MIN:
        return None
    ancho = math.dist(xy[HOMBRO_IZQ], xy[HOMBRO_DER])
    if ancho < 1:
        return None
    y_hombros = (xy[HOMBRO_IZQ][1] + xy[HOMBRO_DER][1]) / 2
    return {
        "hombros": inclinacion(xy[HOMBRO_IZQ], xy[HOMBRO_DER]),
        "cabeza": inclinacion(xy[OJO_IZQ], xy[OJO_DER]),
        "altura": (y_hombros - xy[NARIZ][1]) / ancho,  # normalizada por el ancho de hombros
        "ancho": ancho,
    }


def evaluar(m, base):
    """Compara las metricas actuales con la postura calibrada; devuelve los problemas."""
    problemas = []
    if abs(m["hombros"] - base["hombros"]) > TOL_HOMBROS:
        problemas.append("Hombros desnivelados")
    if abs(m["cabeza"] - base["cabeza"]) > TOL_CABEZA:
        problemas.append("Cabeza ladeada")
    if m["altura"] < base["altura"] * TOL_ENCORVADO:
        problemas.append("Encorvado / cabeza baja")
    if m["ancho"] > base["ancho"] * TOL_CERCA:
        problemas.append("Muy cerca de la pantalla")
    return problemas


def persona_principal(resultado):
    """Indice de la persona con la caja mas grande (la mas cercana) o None."""
    if len(resultado.boxes) == 0:
        return None
    areas = resultado.boxes.xywh[:, 2] * resultado.boxes.xywh[:, 3]
    return int(areas.argmax())


def main():
    modelo = YOLO(MODELO)
    camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW: backend mas estable en Windows

    base = None          # metricas de la postura correcta (tras calibrar)
    muestras = None      # lista de metricas mientras se calibra; None si no se calibra
    inicio_mala = None   # instante en que empezo la mala postura actual
    alerta = False
    t_total = t_bueno = 0.0
    t_prev = time.time()

    while True:
        ok, frame = camara.read()
        if not ok:
            break
        ahora = time.time()
        dt, t_prev = ahora - t_prev, ahora

        resultado = modelo(frame, verbose=False)[0]
        imagen = resultado.plot(boxes=False)  # esqueleto y keypoints, sin caja

        m, xy, problemas = None, None, []
        i = persona_principal(resultado)
        if i is not None:
            xy = resultado.keypoints.xy[i].cpu().numpy()
            conf = resultado.keypoints.conf[i].cpu().numpy()
            m = medir(xy, conf)

        if m and muestras is not None:  # --- calibrando ---
            muestras.append(m)
            if len(muestras) >= MUESTRAS_CALIBRACION:
                base = {k: sum(s[k] for s in muestras) / len(muestras) for k in m}
                muestras = None
        elif m and base:                # --- monitoreando ---
            problemas = evaluar(m, base)
            t_total += dt
            if not problemas:
                t_bueno += dt
            if problemas:
                inicio_mala = inicio_mala or ahora
                nueva = ahora - inicio_mala >= SEGUNDOS_ALERTA
                if nueva and not alerta and winsound:
                    winsound.MessageBeep()  # suena solo al iniciar la alerta
                alerta = nueva
            else:
                inicio_mala, alerta = None, False
        else:                           # --- sin datos confiables ---
            inicio_mala, alerta = None, False

        # Estado y color
        if muestras is not None:
            estado, color = f"Calibrando... {len(muestras)}/{MUESTRAS_CALIBRACION}", AMARILLO
        elif base is None:
            estado, color = "Sientate derecho y presiona 'c' para calibrar", AMARILLO
        elif m is None:
            estado, color = "No te veo bien (cara y hombros)", AMARILLO
        elif alerta:
            estado, color = "ALERTA: corrige tu postura", ROJO
        elif problemas:
            estado, color = "Atencion...", AMARILLO
        else:
            estado, color = "Postura correcta", VERDE

        # Lineas de hombros y ojos, coloreadas segun el estado
        if m:
            for a, b in ((HOMBRO_IZQ, HOMBRO_DER), (OJO_IZQ, OJO_DER)):
                cv2.line(imagen, tuple(map(int, xy[a])), tuple(map(int, xy[b])), color, 3)

        cv2.putText(imagen, estado, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        for n, p in enumerate(problemas):
            cv2.putText(imagen, "- " + p, (10, 70 + 28 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROJO, 2)
        if m and base:  # angulos en vivo (diferencia contra tu postura calibrada)
            h, c = m["hombros"] - base["hombros"], m["cabeza"] - base["cabeza"]
            y = imagen.shape[0] - 70
            cv2.putText(imagen, f"Hombros {h:+.1f} deg | Cabeza {c:+.1f} deg", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if t_total > 0:
            puntaje = 100 * t_bueno / t_total
            cv2.putText(imagen, f"Puntaje de sesion: {puntaje:.0f}%", (10, imagen.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Pose YOLO - ergonomia postural", imagen)

        tecla = chr(cv2.waitKey(1) & 0xFF).lower()
        if tecla == "c":
            muestras = []  # (re)inicia la calibracion
        elif tecla == "q":
            break

    camara.release()
    cv2.destroyAllWindows()
    if t_total > 0:
        print(f"Sesion: {t_total:.0f} s monitoreados, {100 * t_bueno / t_total:.0f}% con buena postura")


if __name__ == "__main__":
    main()
