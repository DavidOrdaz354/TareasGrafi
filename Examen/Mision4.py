import cv2
import numpy as np

# cargar la imagen con ruido
img = cv2.imread(r"C:\Users\death\Desktop\entorno\TareasGrafi\Examen\m4_ruido.png")

# 1. convertir la imagen a HSV

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 2. crear mascara para el color cyan


# rango sugerido en la consigna
bajo = np.array([80,100,100])
alto = np.array([100,255,255])

# aplicar filtro de color
mascara = cv2.inRange(hsv, bajo, alto)

# 3. mostrar la mascara

cv2.imshow("imagen original", img)
cv2.imshow("mascara cyan", mascara)

cv2.waitKey(0)
cv2.destroyAllWindows()