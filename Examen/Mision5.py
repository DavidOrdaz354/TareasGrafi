import cv2
import numpy as np
import math

# 1. crear lienzo negro de 500x500

lienzo = np.zeros((500,500,3), dtype=np.uint8)


# 2. recorrer t desde 0 hasta 2pi

t = 0

while t <= 6.28:

    # calcular las ecuaciones parametricas
    x = 250 + 150 * math.sin(3*t)
    y = 250 + 150 * math.sin(2*t)

    # convertir a enteros para usar como coordenadas
    x = int(x)
    y = int(y)

    # dibujar un punto blanco en esa posicion
    cv2.circle(lienzo, (x,y), 1, (255,255,255), -1)

    # incrementar t
    t = t + 0.01


# mostrar resultado

cv2.imshow("antena parabolica", lienzo)

cv2.waitKey(0)
cv2.destroyAllWindows()