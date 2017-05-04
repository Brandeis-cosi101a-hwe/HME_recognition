import numpy as np
import cv2
import os
import glob
import ntpath

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

    return img


def padding_32(img):
    """
    :type img: ndarray
    :rtype: ndarray
    """
    # TODO: add padding to 32*32

    return img


# path of dataset
path = os.path.join(os.getcwd(), 'annotated', 'syms')

# all paths of pics
pics = glob.glob(path + '/*.png')

# length of pics
pic_nums = len(pics)

# init x, y and pre-alloc mem
x = np.empty((pic_nums * 32 * 32))
y = np.empty((pic_nums))

pxl = 0  # pointer to next slot to insert img32 into x (pointer_x_left)
py = 0 #pointer to append label

for pic in pics:

    # get label from each pic name
    label = ntpath.basename(pic).split('_')[3]

    # Readin pic
    img = cv2.imread(pic, 0) # 0 for grayscale 1 channel

    # padding to square
    img = padding_square(img)

    # resize to 28*28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # padding to 32*32
    img = padding_32(img)

    # flatten img to fill np array
    flatten = img.flatten()

    #insert flatten to x
    pxr = pxl + 32 * 32
    x[pxl:pxr] = flatten
    pxl = pxr

    # append label to y
    y[py] = label
    py+=1

#save to npz
np.savez_compressed('train32_raw.npz', x = x, y = y)
