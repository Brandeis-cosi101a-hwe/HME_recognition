import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pylab
import ntpath


def nothing(x):
    pass

path = os.path.join(os.getcwd(), 'test', 'eqs')
sym_path = os.path.join(os.getcwd(), 'test', 'syms')
# all paths of pics
pics = glob.glob(path + '\\*.png')

pic = pics[0]
img = cv2.imread(pic, 0)
syms = glob.glob(sym_path + '\\'+ ntpath.basename(pic).split('.')[0] + '_*.png')
for sym in syms:
    boudingRect = []
    
imgplot = plt.imshow(img)
pylab.show()
