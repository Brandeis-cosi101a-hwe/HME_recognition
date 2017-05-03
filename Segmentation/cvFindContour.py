import cv2
import numpy as np
im_gray = cv2.imread("test2out.png", 0)

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
im = 255 - im_gray
# Threshold the image
ret, im_th = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)


im2, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
for rect in rects:
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
im = 255 - im

cv2.imwrite('test2out.png', im)