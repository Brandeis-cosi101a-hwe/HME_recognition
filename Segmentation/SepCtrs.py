"""
Separate each contour and save as images
"""
import cv2
import numpy as np
from skimage import measure

im_gray = cv2.imread("test.png", 0)

def getName(rect):
    return str(rect[0])+'_'+str(rect[0]+rect[2])+'_'+str(rect[1])+"_"+str(rect[1] + rect[3])+'.png'

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
im = 255 - im_gray
# Threshold the image
ret, im_th = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)

im2, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
for rect in rects:
    # create new img file
    im_temp = im2[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    blobs_labels = measure.label(im_temp, connectivity=5, neighbors=8, background=0, return_num=False)
    im_temp = (blobs_labels == 1) * 255
    cv2.imwrite(getName(rect), im_temp)
    # draw rectangle
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

im = 255 - im

cv2.imwrite('out.png', im)