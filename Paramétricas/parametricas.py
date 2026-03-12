import numpy as np
import cv2

width, height = 1000, 1000
img = np.ones((height, width, 3), dtype=np.uint8) * 255

center_x, center_y = width // 2, height // 2

R = 250
k = 5

theta_increment = 0.0001
theta = 0
max_theta = 6 * np.pi

while True:

    r = R * np.cos(k * theta)
    x = int(center_x + r * np.cos(theta))
    y = int(center_y + r * np.sin(theta))

    cv2.circle(img, (x, y), 1, (0, 0, 0), -1)

    cv2.imshow("Rosa Animada", img)

    theta += theta_increment

    if theta >= max_theta:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()