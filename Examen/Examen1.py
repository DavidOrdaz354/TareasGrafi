import cv2
import numpy as np

imagen = cv2.imread(r"C:\Users\death\Desktop\entorno\TareasGrafi\Examen\m1_oscura.png", cv2.IMREAD_GRAYSCALE)

# --- MODO RAW ---

h, w = imagen.shape
img_raw = np.zeros((h, w), dtype=np.uint8)

for i in range(h):
    for j in range(w):
        
        nuevo_pixel = int(imagen[i, j]) * 50
        
        if nuevo_pixel > 255:
            nuevo_pixel = 255
            
        img_raw[i, j] = nuevo_pixel

cv2.imwrite("m1_resuelto_raw.png", img_raw)

# --- MODO OPENCV / NUMPY ---

img_numpy = np.clip(imagen.astype(np.int32) * 50, 0, 255).astype(np.uint8)

cv2.imwrite("m1_resuelto_numpy.png", img_numpy)

cv2.imshow("Original Oscura", imagen)
cv2.imshow("Mision 1 - Raw", img_raw)
cv2.imshow("Mision 1 - OpenCV/Numpy", img_numpy)

cv2.waitKey(0)
cv2.destroyAllWindows()