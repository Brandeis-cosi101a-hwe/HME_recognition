import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
from sklearn import preprocessing
import h5py


class KerasModel(object):

    @staticmethod
    def load():
        # keras model
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
            x = x.reshape(y.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x = x.reshape(y.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x = x.astype('float32')

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

        model.load_weights("model_final32_1.h5")
        return (model, le)
