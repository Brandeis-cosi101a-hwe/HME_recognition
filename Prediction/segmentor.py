"""
Separate each contour and save as images
"""
import cv2
import numpy as np
import skimage
from skimage import measure
from skimage.exposure import exposure
import skimage.transform

def getName(rect):
    return str(rect[0])+'_'+str(rect[0]+rect[2])+'_'+str(rect[1])+"_"+str(rect[1] + rect[3])+'.png'

def targetNum(blobs_labels):
    (h, w) = blobs_labels.shape
    edge1 = blobs_labels[0]
    edge2 = blobs_labels[h-1]
    edge3 = blobs_labels[:,0]
    edge4 = blobs_labels[:,w-1]
    edges = []
    edges.extend(edge1)
    edges.extend(edge2)
    edges.extend(edge3)
    edges.extend(edge4)
    edges = [x for x in edges if x != 0]
    nums = list(set(edges))
    for num in nums:
        if (num in edge1) and (num in edge2) and (num in edge3) and (num in edge4):
            #print(num)
            return num
    return 0

def padding_square(img):
    """
    :type img: ndarray
    :rtype: ndarray
    """
    # TODO: add padding to right/bottom side of img and make it a square
    img = exposure.adjust_gamma(img, 0.15)
    (vertical_pixel, horizontal_pixel) = img.shape
    if vertical_pixel > horizontal_pixel:
        horizontal_padding = int(round(vertical_pixel  - horizontal_pixel))
        padding = ((0, 0), (0, horizontal_padding))
    else:
        vertical_padding = int(round(horizontal_pixel - vertical_pixel))
        padding = ((0, vertical_padding), (0, 0))
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


def process(image_path):
    im_gray = cv2.imread(image_path, 0)
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
        im_temp = (blobs_labels == targetNum(blobs_labels)) * 255
        im_temp = padding_square(im_temp)
        im_temp = im_temp/255
        im_temp = cv2.resize(im_temp, (28, 28), interpolation=cv2.INTER_AREA)

        ##TODO: list.append(results, im_temp)
        im_temp = padding_32(im_temp)
        # cv2.imwrite(getName(rect), im_temp)
    return ([x1, y1, x2, y2], results)
