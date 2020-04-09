"""Trains simple keras model.

Used dataset: MNIST
@author: kaliv
"""

import os
import time
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D


def train():
    """
    Trains a simple convolutional net keras model on MNIST dataset.

    Returns
    -------
    None.

    """
    # initial parameters
    batch_size = 128
    num_classes = 10
    epochs = 1

    # create dirs for result files
    path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
        __file__)), 'temp', str(time.time())))
    os.makedirs(path_wd)

    dataset_path = 'mnist.h5'
    # load dataset
    dataset = mnist.load_data(path=dataset_path)
    (x_train, y_train), (x_test, y_test) = dataset

    # preprocess data
    x_train = x_train / 255
    x_test = x_test / 255

    axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1

    x_train = np.expand_dims(x_train, axis)
    x_test = np.expand_dims(x_test, axis)
    input_shape = x_train.shape[1:]

    model = Sequential()
    model.add(Conv2D(
        32,
        kernel_size=(3, 3),
        activation=keras.activations.relu,
        input_shape=input_shape
        ))
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation=keras.activations.relu))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation=keras.activations.relu))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test)
        )
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # save trained model
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
