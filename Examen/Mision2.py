import cv2
import numpy as np

# Cargar las dos mitades del QR
mitad1 = cv2.imread(r"C:\Users\death\Desktop\entorno\TareasGrafi\Examen\m2_mitad1.png")
mitad2 = cv2.imread(r"C:\Users\death\Desktop\entorno\TareasGrafi\Examen\m2_mitad2.png")

# 1. Crear un lienzo en blanco de 400x400
lienzo = np.zeros((400,400,3), dtype=np.uint8)

# -----------------------------
# 2. Trasladar la mitad superior
# -----------------------------

# matriz de traslación para mover la imagen a la parte superior del lienzo
M1 = np.float32([
    [1,0,-100],   # mover en eje X
    [0,1,-50]     # mover en eje Y
])

# aplicar la traslación
mitad1_movida = cv2.warpAffine(mitad1, M1, (400,400))

# pegar la primera mitad en el lienzo
lienzo = cv2.add(lienzo, mitad1_movida)


# -----------------------------
# 3. Rotar la mitad inferior
# -----------------------------

# obtener el tamaño de la imagen
h, w = mitad2.shape[:2]

# calcular el centro de la imagen para rotarla
centro = (w//2, h//2)

# matriz de rotación de 180 grados
Mrot = cv2.getRotationMatrix2D(centro, 180, 1)

# aplicar la rotación
mitad2_rotada = cv2.warpAffine(mitad2, Mrot, (w,h))


# -----------------------------
# 4. Trasladar la mitad rotada hacia abajo
# -----------------------------

# matriz para mover la imagen a la parte inferior del lienzo
M2 = np.float32([
    [1,0,100],   # mover en X
    [0,1,200]    # mover en Y
])

# aplicar la traslación
mitad2_movida = cv2.warpAffine(mitad2_rotada, M2, (400,400))

# unir la segunda mitad con el lienzo
lienzo = cv2.add(lienzo, mitad2_movida)


# -----------------------------
# Mostrar el QR reconstruido
# -----------------------------

cv2.imshow("QR reconstruido", lienzo)

cv2.waitKey(0)
cv2.destroyAllWindows()