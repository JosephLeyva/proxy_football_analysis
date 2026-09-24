import cv2
import numpy as np

def get_pitch_mask(frame):
    # Mascara de la cancha: pixeles verdes (pasto), cerrando huecos de lineas y jugadores
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)

    # Quedarse con la region verde mas grande y rellenarla con su envolvente convexa
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.fillPoly(mask, [cv2.convexHull(largest)], 1)
    return mask
