import numpy as np
import cv2

width, height = 1000, 1000
img = np.ones((height, width, 3), dtype=np.uint8) * 255

center_x, center_y = width // 2, height // 2

R = 250
k = 5   # número de pétalos si es impar

theta_increment = 0.01
theta = 0
max_theta = 6 * np.pi

while theta < max_theta:
    r = R * np.cos(k * theta)
    x = int(center_x + r * np.cos(theta))
    y = int(center_y + r * np.sin(theta))

    cv2.circle(img, (x, y), 1, (0, 0, 0), -1)
    theta += theta_increment

cv2.imshow("Rose Curve", img)
cv2.waitKey(0)
cv2.destroyAllWindows()