import cv2
import numpy as np

# Crear una imagen en blanco
img = np.ones((500, 500, 3), dtype=np.uint8) * 255

# Color negro
color = (0, 0, 0)
grosor = 2

# =========================
# BASE (cubo isométrico)
# =========================

# Punto frontal inferior izquierdo
A = (150, 300)
# Punto frontal inferior derecho
B = (300, 300)

# Desplazamiento isométrico
dx = 80
dy = 50

# Puntos traseros
C = (B[0] + dx, B[1] - dy)
D = (A[0] + dx, A[1] - dy)

# Dibujar base
cv2.line(img, A, B, color, grosor)
cv2.line(img, B, C, color, grosor)
cv2.line(img, C, D, color, grosor)
cv2.line(img, D, A, color, grosor)

# =========================
# ALTURA (paredes)
# =========================

altura = 120

A1 = (A[0], A[1] - altura)
B1 = (B[0], B[1] - altura)
C1 = (C[0], C[1] - altura)
D1 = (D[0], D[1] - altura)

# Líneas verticales
cv2.line(img, A, A1, color, grosor)
cv2.line(img, B, B1, color, grosor)
cv2.line(img, C, C1, color, grosor)
cv2.line(img, D, D1, color, grosor)

# Parte superior (techo base)
cv2.line(img, A1, B1, color, grosor)
cv2.line(img, B1, C1, color, grosor)
cv2.line(img, C1, D1, color, grosor)
cv2.line(img, D1, A1, color, grosor)

# =========================
# TECHO (tipo casa)
# =========================

# Punto central del techo
E = ((A1[0] + B1[0]) // 2, A1[1] - 60)
F = ((D1[0] + C1[0]) // 2, D1[1] - 60)

# Dibujar techo
cv2.line(img, A1, E, color, grosor)
cv2.line(img, B1, E, color, grosor)
cv2.line(img, D1, F, color, grosor)
cv2.line(img, C1, F, color, grosor)
cv2.line(img, E, F, color, grosor)

# =========================
# Mostrar imagen
# =========================

cv2.imshow("Casa Isometrica", img)
cv2.waitKey(0)
cv2.destroyAllWindows()