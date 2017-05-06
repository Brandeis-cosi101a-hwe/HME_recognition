import numpy as np
import cv2
train32 = np.load('train32_raw.npz')
x = train32['x']
y = train32['y']
x = x.reshape(y.shape[0], 32, 32, 1)
# cv2.imshow('img',x[6])
# cv2.waitKey(0)
print(y[6])
