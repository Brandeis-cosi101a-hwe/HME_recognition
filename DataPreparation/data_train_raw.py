import numpy
import numpy as np
import cv2
import os
import glob
import ntpath
import skimage
from PIL import Image
from skimage.exposure import exposure
import skimage.transform
import shutil

# Overall: multiprocessing is not used, but mem is pre-allocated for better performance


# All available labels generated from pic names (using set)


categories = ['c', 's', 'y', 'pi', '3', '0', '6', '1', 'mul', 'p', 'cos', '2',
              'n', 'A', '-', 'pm', 'a', '+', '(', 'sqrt', 'sin', 'x', 'div', 'dots', 'tan',
              'bar', 'b', 'o', 't', 'h', 'delta', 'f', 'm', ')', 'd', '4', 'i', 'k', '=', 'frac']

def padding_square(img):
    """
    :type img: ndarray
    :rtype: ndarray
    """
    # TODO: add padding to right/bottom side of img and make it a square
    img = exposure.adjust_gamma(img, 0.15)
    (vertical_pixel, horizontal_pixel) = img.shape
    if vertical_pixel > horizontal_pixel:
        vertical_padding = int(round(vertical_pixel * 0.3))
        horizontal_padding = int(round(vertical_pixel * 1.3 - horizontal_pixel))
        padding = ((0, vertical_padding), (0, horizontal_padding))
    else:
        horizontal_padding = int(round(horizontal_pixel * 3))
        vertical_padding = int(round(horizontal_pixel * 1.3 - vertical_pixel))
        padding = ((0, vertical_padding), (0, horizontal_padding))
    img = skimage.util.pad(img, padding, 'constant', constant_values=0)
    return img

def padding_32(img):
    """
    :type img: ndarray
    :rtype: ndarray
    """
    # TODO: add padding to 32*32
    padding = ((2, 2), (2, 2))
    img = skimage.util.pad(img, padding, 'constant', constant_values=0)
    return img


# path of dataset
path = os.path.join(os.getcwd(), 'annotated', 'syms')

# all paths of pics
pics = glob.glob(path + '/*.png')

# length of pics
pic_nums = len(pics)

# init x, y and pre-alloc mem
x = np.empty((pic_nums * 32 * 32))
y = np.empty((pic_nums), dtype=str)

pxl = 0  # pointer to next slot to insert img32 into x (pointer_x_left)
py = 0 # pointer to append label

for pic in pics:
    # get label from each pic name
    label = ntpath.basename(pic).split('_')[3]

    #print(pic)
    #print(label)
    #print(py)

    # Readin pic
    img = cv2.imread(pic, 0) # 0 for grayscale 1 channel

    # padding to square
    img = padding_square(img)

    # resize to 28*28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # padding to 32*32
    img = padding_32(img)

    # test
    #img = Image.fromarray(img, 'L')
    #img.save('my.png')
    #img.show()

    # flatten img to fill np array
    flatten = img.flatten()

    # insert flatten to x
    pxr = pxl + 32 * 32
    x[pxl:pxr] = flatten
    pxl = pxr

    # append label to y
    y[py] = label
    py+=1

# save to npz
np.savez_compressed('train32_raw.npz', x = x, y = y)
