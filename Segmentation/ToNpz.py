import numpy as np
import cv2
import glob
import sys

x = np.empty((0))
y = np.empty((0))

# class names to extract
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'x', 'y', 'A', 'a', 'b', 'c', 'd',
              'm', 'n', 'p', 'f', 'h', 'k']

c_map = {
    '0':'30',
    '1':'31',
    '2':'32',
    '3':'33',
    '4':'34',
    '5':'35',
    '6':'36',
    '7':'37',
    '8':'38',
    '9':'39',
    'A':'41',
    'a':'61',
    'b':'62',
    'c':'63',
    'd':'64',
    'h':'68',
    'k':'6b',
    'f':'66',
    'm':'6d',
    'n':'6e',
    'p':'70',
    'x':'78',
    'y':'79'
}


# dataset path
path = './annotated/'


for label in categories:
    # c_path = path + c_map[label] + '/train_' + c_map[label] + '/'
    c_pics = glob.glob(path + '*')

    # samples size for debug
    print(label, len(c_pics))

    i = 0
    l = len(c_pics)
    t = min(l, 5000)

    for pic in c_pics:
        if i > 5000:
            break
        if pic.endswith('.png') or pic.endswith('.jpg'):

            # read in
            img = cv2.imread(pic, 0)

            # write for debug
            #  cv2.imwrite('./output/' + str(t) + '.png', img)

            img = img/255

            x = np.append(x, img.flatten())
            y = np.append(y, label)

            p = i*100.0/t
            sys.stdout.write("\r%d%%" % p)
            sys.stdout.flush()
            i+=1
    print("%s done." % label)

# save to file
np.savez_compressed('nist_annotated.npz', x = x, y = y)