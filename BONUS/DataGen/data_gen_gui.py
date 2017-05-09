import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pylab


def nothing(x):
    pass

path = os.path.join(os.getcwd(), 'test', 'eqs')
# all paths of pics
pics = glob.glob(path + '/*.png')

pic = pics[0]
img = cv2.imread(pic, 0)
imgplot = plt.imshow(img)
pylab.show()
