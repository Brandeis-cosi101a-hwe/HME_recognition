from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
from sklearn import preprocessing
import keras_buildresnet as resnet

alignmnist = np.load('train32_raw.npz')
x = alignmnist['x']
y = alignmnist['y']

le = preprocessing.LabelEncoder()
le.fit(y)
t_y = le.transform(y)

batch_size = 128
num_classes = len(le.classes_)
epochs = 20

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

model = resnet.ResnetBuilder.build_resnet_34((1, img_rows, img_cols), num_classes)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x, y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          validation_split=0.20)


# serialize weights to HDF5
model.save_weights("model_final32_res34.h5")
