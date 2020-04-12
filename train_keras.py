"""Trains simple keras model.

Used dataset: MNIST
@author: kaliv
"""

import os
import time
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, Input
from keras.layers import BatchNormalization, Activation, Concatenate
from keras.layers import AveragePooling2D


def train():
    """
    Trains a simple convolutional net keras model on MNIST dataset.

    Returns
    -------
    None.

    """
    # initial parameters
    batch_size = 1024
    num_classes = 10
    epochs = 12
    dataset_path = 'mnist.h5'

    # create dirs for result files
    path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
        __file__)), 'temp', str(time.time())))
    os.makedirs(path_wd)
    # load dataset
    dataset = mnist.load_data(path=dataset_path)
    (x_train, y_train), (x_test, y_test) = dataset
    
    # preprocess data
    x_train = x_train / 255
    x_test = x_test / 255
    axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    x_train = np.expand_dims(x_train, axis)
    x_test = np.expand_dims(x_test, axis)
    y_train = keras.utils.np_utils.to_categorical(y_train)
    y_test = keras.utils.np_utils.to_categorical(y_test)
    input_shape = x_train.shape[1:]
    
    # create the ANN model
    input_layer = Input(input_shape)
    layer = Conv2D(filters=16,
               kernel_size=(5, 5),
               strides=(2, 2))(input_layer)
    layer = BatchNormalization(axis=axis)(layer)
    layer = Activation('relu')(layer)
    layer = AveragePooling2D()(layer)
    branch1 = Conv2D(filters=32,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu')(layer)
    branch2 = Conv2D(filters=8,
                 kernel_size=(1, 1),
                 activation='relu')(layer)
    layer = Concatenate(axis=axis)([branch1, branch2])
    layer = Conv2D(filters=10,
               kernel_size=(3, 3),
               activation='relu')(layer)
    layer = Flatten()(layer)
    layer = Dropout(0.01)(layer)
    layer = Dense(units=10,
              activation='softmax')(layer)
    model = Model(input_layer, layer)
    model.summary()

    # train and test the model
    loss_fn = 'categorical_crossentropy'
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(x_test, y_test)
        )
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # save trained model and test data
    model_name = 'keras_model'
    keras.models.save_model(
        model,
        os.path.join(path_wd, model_name + '.h5')
        )
    np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
    np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)
    np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_train[::10])
    print('Model and dataset saved to {}'.format(path_wd))
    
    return path_wd
