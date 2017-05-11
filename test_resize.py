import numpy as np
import cv2
img = cv2.imread('261_892_130_143.png', 0)
print(img.shape)
img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
cv2.imwrite('testouttt.png', img)
