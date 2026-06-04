import cv2 as cv
import numpy as np 

# Creamos un lienzo de 500x500 con 3 canales (RGB) para tener colores
img = np.ones((500, 500, 3), np.uint8) * 255 

# Partes del Oso
# Orejas
cv.circle(img, (180, 110), 60, (0, 0, 0), -1)  # Oreja Izq
cv.circle(img, (320, 110), 60, (0, 0, 0), -1)  # Oreja Der

# Cabeza (Círculo principal)
cv.circle(img, (250, 170), 108, (203, 192, 255), -1) # Color rosa claro
cv.circle(img, (250, 170), 108, (0, 0, 0), 2)       # Contorno

# Hocico
cv.circle(img, (250, 230), 40, (255, 255, 255), -1)
cv.circle(img, (250, 230), 20, (0, 0, 0), -1)       # Nariz

# Ojos
cv.circle(img, (215, 180), 15, (0, 0, 0), -1)       # Ojo Izq
cv.circle(img, (285, 180), 15, (0, 0, 0), -1)       # Ojo Der

# Cuerpo (Rectángulo redondeado o base)
#cv.rectangle(img, (180, 320), (320, 450), (203, 192, 255), -1)
#cv.rectangle(img, (180, 320), (320, 450), (0, 0, 0), 2)
#cuerpo (circulo)
cv.circle(img, (250, 367), 90, (203, 192, 255), -1)
cv.circle(img, (250, 367), 90, (0, 0, 0), 2)

#Mano izq
cv.circle(img, (136, 320), 30, (203,192, 255), -1)
cv.circle(img, (136, 320), 30, (0, 0, 0), 2)

#Mano der
cv.circle(img, (360, 320), 30, (203,192, 255), -1)
cv.circle(img, (360, 320), 30, (0, 0, 0), 2)

#Pata Izq
cv.circle(img, (353, 430), 30, (203,192, 255), -1)
cv.circle(img, (353, 430), 30, (0, 0, 0), 2)

#Pata Der
cv.circle(img, (145, 430), 30, (203,192, 255), -1)
cv.circle(img, (145, 430), 30, (0, 0, 0), 2)

#Cicatriz en la cabeza
cv.line(img, (180,180), (200,100), (0, 0, 0), 2)

#Hombligo
cv.circle(img, (250, 430), 10, (0, 0, 0), -1)  



cv.imshow('Oso', img)
cv.waitKey(0)
cv.destroyAllWindows()