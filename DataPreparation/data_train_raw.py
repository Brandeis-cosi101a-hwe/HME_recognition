import numpy as np
import cv2
import os
import glob
import ntpath

categories = ['c', 's', 'y', 'pi', '3', '0', '6', '1', 'mul', 'p', 'cos', '2',
 'n', 'A', '-', 'pm', 'a', '+', '(', 'sqrt', 'sin', 'x', 'div', 'dots', 'tan',
 'bar', 'b', 'o', 't', 'h', 'delta', 'f', 'm', ')', 'd', '4', 'i', 'k', '=', 'frac']

path = os.path.join(os.getcwd(), 'DataPreparation', 'annotated', 'syms')
pics = glob.glob(path + '/*.png')
for pic in pics:
    label = ntpath.basename(pic).split('_')[3]
    img = cv2.imread(pic, 0)
    
