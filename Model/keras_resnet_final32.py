from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
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

# model.fit(x, y,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           shuffle=True,
#           validation_split=0.20)
print('Using real-time data augmentation.')
# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x)

# Fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(x, y, batch_size=batch_size),
                    steps_per_epoch=x.shape[0],
                    epochs=epochs, verbose=1, max_q_size=100)

# serialize weights to HDF5
model.save_weights("model_final32_res34.h5")
