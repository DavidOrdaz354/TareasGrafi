import cv2
import numpy as np

# 1. Crear el lienzo azul oscuro de 500x500
lienzo = np.zeros((500,500,3), dtype=np.uint8)

# color base BGR(50,20,20)
lienzo[:] = (50,20,20)


# -----------------------------
# dibujar el circulo central
# -----------------------------

# centro (250,250), radio 100, color amarillo
cv2.circle(lienzo, (250,250), 100, (0,255,255), 3)


# -----------------------------
# dibujar el rectangulo rojo
# -----------------------------

# de (200,200) a (300,300)
cv2.rectangle(lienzo, (200,200), (300,300), (0,0,255), -1)


# -----------------------------
# dibujar las lineas diagonales
# -----------------------------

# linea de esquina superior izquierda a esquina inferior derecha
cv2.line(lienzo, (0,0), (500,500), (255,255,255), 2)

# linea de esquina superior derecha a esquina inferior izquierda
cv2.line(lienzo, (500,0), (0,500), (255,255,255), 2)


# -----------------------------
# guardar imagen final
# -----------------------------

cv2.imwrite("m3_sello_forjado.png", lienzo)


# mostrar resultado
cv2.imshow("Sello", lienzo)
cv2.waitKey(0)
cv2.destroyAllWindows()