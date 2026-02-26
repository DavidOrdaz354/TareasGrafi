import cv2 as cv
import numpy as np 

# Crear el fondo (el '3' es clave para poder usar colores BGR)
img = np.ones((500, 500, 3), np.uint8) * 220 

# --- Colores (BGR) ---
cafe = (42, 42, 165)      
cafe_oscuro = (19, 69, 139)
negro = (0, 0, 0)
blanco = (255, 255, 255)

# --- 1. Parte trasera ---
# Colita (Línea)
cv.line(img, (300, 310), (360, 260), cafe_oscuro, 10)

# Patitas traseras (Rectángulos)
cv.rectangle(img, (190, 380), (220, 430), cafe_oscuro, -1)
cv.rectangle(img, (280, 380), (310, 430), cafe_oscuro, -1)

# --- 2. Cuerpo ---
# Cuerpo principal (Rectángulo)
cv.rectangle(img, (200, 290), (300, 400), cafe, -1)

# Patitas delanteras (Rectángulos)
cv.rectangle(img, (210, 400), (240, 450), cafe_oscuro, -1)
cv.rectangle(img, (260, 400), (290, 450), cafe_oscuro, -1)

# --- 3. Cabeza ---
# Orejas (Círculos)
cv.circle(img, (175, 130), 50, cafe_oscuro, -1)
cv.circle(img, (325, 130), 50, cafe_oscuro, -1)

# Cara (Círculo)
cv.circle(img, (250, 200), 85, cafe, -1)

# Ojos (Círculos)
cv.circle(img, (220, 185), 15, negro, -1)
cv.circle(img, (280, 185), 15, negro, -1)
# Brillo de los ojos (Círculos pequeños)
cv.circle(img, (225, 180), 5, blanco, -1)
cv.circle(img, (285, 180), 5, blanco, -1)

# Nariz (Círculo)
cv.circle(img, (250, 220), 12, negro, -1)

# Boca (Líneas)
cv.line(img, (250, 232), (225, 255), negro, 3)
cv.line(img, (250, 232), (275, 255), negro, 3)

# Mostrar la imagen
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()




