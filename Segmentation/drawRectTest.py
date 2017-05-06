import cv2
import numpy as np
import glob
import ntpath
import os
import sys

path = os.path.join(os.getcwd(), 'Segmentation', 'annotated', 'eqs')
# all paths of pics
pics = glob.glob(path + '/*.png')

kernel = np.ones((3, 3), np.uint8)
i=0
t=len(pics)
for pic in pics:
    img = cv2.imread(pic, 0)
    # Convert to grayscale and apply Gaussian filtering
    img = cv2.GaussianBlur(img, (11, 11), 0)
    # Threshold the image
    ret, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)

    im2, ctrs, hier = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    imgc = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for rect in rects:
        cv2.rectangle(imgc, (rect[0], rect[1]), (rect[0] +
                                                 rect[2], rect[1] + rect[3]), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(os.getcwd(), 'output',
                             ntpath.basename(pic)), imgc)
    sys.stdout.write("\r%d/%d" % (i,t))
    sys.stdout.flush()
    i+=1
