"""
Separate each contour and save as images
"""
import cv2
import numpy as np
import skimage
from skimage import measure
from skimage.exposure import exposure
import skimage.transform

cropped_imgs = []
cropped_rects = []

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
        horizontal_padding = int(round(vertical_pixel - horizontal_pixel))
        horizontal_padding = int(horizontal_padding/1)
        padding = ((0, 0), (horizontal_padding, horizontal_padding))
    else:
        vertical_padding = int(round(horizontal_pixel - vertical_pixel))
        vertical_padding = int(vertical_padding/2)
        padding = ((vertical_padding, vertical_padding), (0, 0))
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

def isBar(img,rect):
    treshold1 = 0.5
    treshold2 = 3
    used = 0
    total = 0
    for i in img:
        total += 1
        if i==1:
            used += 1
    usage = used/total
    ratio = rect[2] / rect[3]
    return usage>treshold1 or ratio>treshold2

def overlap(bars_rect, i , j):
    r1 = bars_rect[i]
    r2 = bars_rect[j]
    if (r1[0]+r1[2])<r2[0] or r1[0]>(r2[0]+r2[2]):  # don't even touch horizontally
        return False
    temp_array = [r1[0], r1[0] + r1[2],
                  r2[0], r2[0] + r2[2]]
    temp_array.sort()
    vert_overlap_rate = (temp_array[2] - temp_array[1]) / (temp_array[3] - temp_array[0])
    return vert_overlap_rate>0.6

def cropImg(im_temp2, new_rect):
    rect = new_rect
    cv2.imwrite("./output/" + getName(rect), im_temp2)
    im_temp2 = im_temp2 * 255
    im_temp2 = padding_square(im_temp2)
    im_temp2 = im_temp2 / 255
    im_temp2 = cv2.resize(im_temp2, (28, 28), interpolation=cv2.INTER_AREA)
    im_temp2 = padding_32(im_temp2)
    # cv2.imwrite(getName(rect), im_temp2)
    cropped_imgs.append(im_temp2)
    cropped_rects.append([rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]])

def img_combine(img1, img2, r1, r2, base_rect):
    w = base_rect[2]
    h = base_rect[3]
    base_w = base_rect[0]
    base_h = base_rect[1]
    r1 = [r1[0]-base_w, r1[1]-base_h, r1[2], r1[3]]
    r2 = [r2[0]-base_w, r2[1]-base_h, r2[2], r2[3]]
    img1 = skimage.util.pad(img1, ((r1[1], h-r1[3]-r1[1]),(r1[0],w-r1[2]-r1[0])), 'constant', constant_values=0)
    img2 = skimage.util.pad(img2, ((r2[1], h-r2[3]-r2[1]),(r2[0],w-r2[2]-r2[0])), 'constant', constant_values=0)
    new_im = ((img1+img2)>0)*1
    return new_im

def process(image_path):
    im_gray = cv2.imread(image_path, 0)
    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    im = 255 - im_gray
    # Threshold the image
    ret, im_th = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)

    im2, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    bars_rect = []
    bars_im = []

    for rect in rects:
        # create new img file
        im_temp = im2[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        blobs_labels = measure.label(im_temp, connectivity=5, neighbors=8, background=0, return_num=False)
        im_temp = (blobs_labels == targetNum(blobs_labels))*1
        if isBar(im_temp.flatten(), rect):
            bars_rect.append(rect)
            bars_im.append(im_temp)
        else:
            cropImg(im_temp, rect)
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)


    change_flag = True
    while (change_flag):
        change_flag = False
        i = -1
        j = -1
        for i in range(-1, len(bars_rect)-1):
            i += 1
            j = i
            if (i >= len(bars_rect)):
                break
            for j in range(i, len(bars_rect)-1):
                j += 1
                if (j>=len(bars_rect)):
                    break
                if overlap(bars_rect, i, j):
                    change_flag = True
                    new_rect = (min(bars_rect[i][0], bars_rect[j][0]),
                                min(bars_rect[i][1], bars_rect[j][1]),
                                max(bars_rect[i][0]+bars_rect[i][2], bars_rect[j][0]+bars_rect[j][2]) - min(bars_rect[i][0], bars_rect[j][0]),
                                max(bars_rect[i][1]+bars_rect[i][3], bars_rect[j][1]+bars_rect[j][3]) - min(bars_rect[i][1], bars_rect[j][1]))
                    new_im = img_combine(bars_im[i], bars_im[j], bars_rect[i], bars_rect[j], new_rect)
                    del bars_rect[j]
                    del bars_rect[i]
                    del bars_im[j]
                    del bars_im[i]
                    bars_rect.append(new_rect)
                    bars_im.append(new_im)

    for i in range(0, len(bars_rect)):
        rect = bars_rect[i]
        cropImg(bars_im[i], rect)
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    return cropped_imgs, cropped_rects
