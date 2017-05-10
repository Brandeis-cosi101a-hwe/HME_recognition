import numpy as np
import cv2
train32 = np.load('train32_raw.npz')
x = train32['x']
y = train32['y']
x = x.reshape(y.shape[0], 32, 32, 1)
i=0
for e in x:
    cv2.imwrite('./raw/'+ str(i)+'.png', e)
    i+=1
