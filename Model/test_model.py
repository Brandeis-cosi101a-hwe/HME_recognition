from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
from sklearn import preprocessing
import h5py
import ntpath
import sys
import glob
import cv2
import skimage
from skimage import measure
from skimage.exposure import exposure
import skimage.transform

#data = np.load('nist_annotated.npz')
data = np.load('train32_raw.npz')
x = data['x']
y = data['y']

le = preprocessing.LabelEncoder()
le.fit(y)
t_y = le.transform(y)

batch_size = 128
num_classes = len(le.classes_)
epochs = 12

# input image dimensions
img_rows, img_cols = 32, 32


if K.image_data_format() == 'channels_first':
    x= x.reshape(y.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x = x.reshape(y.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x = x.astype('float32')
print('x shape:', x.shape)
print(x.shape[0], 'train samples')

# convert class vectors to binary class matrices
y = keras.utils.to_categorical(t_y, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# serialize weights to HDF5
model.load_weights("model_final32_1.h5")

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

i=0
t=len(y)
c=0
if(len(sys.argv)>1):
    folder_name = sys.argv[1]
    all_pics = glob.glob("./"+folder_name+"/*.png")
    for pic in all_pics:
        img = cv2.imread(pic, 0)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img = padding_square(img)
        label = ntpath.basename(pic).split('_')[3]
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = padding_32(img)
        # cv2.imwrite('./pre/'+ str(i)+'.png', img)
        result = le.inverse_transform(np.argmax(model.predict(img.reshape([-1,32,32,1]))))
        if label == result:
            c+=1
        i+=1
print(c, t)
