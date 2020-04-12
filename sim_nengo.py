from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import nengo
import nengo_dl

def load_and_run(path):
  model = tf.keras.models.load_model(join(path, 'keras_model.h5'), compile=True)
  num_classes = 10
  n_steps = 50
  (test_images, test_labels) = (np.load(join(path, 'x_test.npz'))['arr_0'],
   np.load(join(path, 'y_test.npz'))['arr_0'])
  test_images = test_images.reshape(test_images.shape[0], -1)
  print('test_labels shape: {}'.format(test_labels.shape))
  test_labels = np.tile(test_labels[:, None, :], (1, n_steps, 1))
  test_images = np.tile(test_images[:, None, :], (1, n_steps, 1))
  print('test_images shape: {}'.format(test_images.shape))
  print('test_labels shape: {}'.format(test_labels.shape))

  converter = nengo_dl.Converter(model)

  with nengo_dl.Simulator(converter.net, dt = 0.1) as sim:
    # the Converter will copy the parameters from the Keras model, so we don't
    # need to do any further training (although we could if we wanted)
    sim.compile(loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])

    print("Test accuracy:", sim.evaluate(
        test_images, test_labels, verbose=0, n_steps = n_steps)["probe_accuracy"])
