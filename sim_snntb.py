"""Configures SNN-Toolbox and runs simulation of converted keras trained model.

@author: kalivoda
coding: utf-8
"""
import os
import time
from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser


def config_and_sim(path_wd=None):
    """
    Create configuration for SNN-Toolbox and executes the simulation.

    Returns
    -------
    None.

    """
    if path_wd is None:
      print('Please specify path to working directory.')
      return

    filename_ann = 'keras_model'

    dt = 0.1  # Time resolution of simulator.

    # Create a config file with experimental setup for SNN Toolbox.
    configparser = import_configparser()
    config = configparser.ConfigParser()

    config['paths'] = {
        'path_wd': path_wd,  # Path to model.
        'dataset_path': path_wd,  # Path to dataset.
        'filename_ann': filename_ann  # Name of input model.
    }

    config['tools'] = {
        'evaluate_ann': True,  # Test ANN on dataset before conversion.
        'normalize': True,  # Normalize weights for full dynamic range.
    }

    config['simulation'] = {
        'simulator': 'INI',  # Chooses execution backend of SNN toolbox.
        'duration': 50,  # Number of time steps to run each sample.
        'num_to_test': 10000,  # How many test samples to run.
        'batch_size': 500,
        'top_k': 3,
        'keras_backend': 'tensorflow',
        'dt': dt   # Time resolution for ODE solving.
    }

    # config['cell'] = {
    #     'tau_refrac': dt,  # Refractory period and delay must be at
    #     'delay': dt,  # least one time step.
    #     'v_thresh': 0.01  # Reducing default value (1) for higher spikerates.
    # }

    """
        Various plots (slows down simulation).
        Leave section empty to turn off plots.
    """
    config['output'] = {
        'plot_vars': {
            'spiketrains',
            #'spikerates',
            'activations',
            #'correlation',
            #'v_mem',
            'error_t'
            }
    }

    # Store config file.
    config_filepath = os.path.join(path_wd, 'config')
    with open(config_filepath, 'w') as configfile:
        config.write(configfile)

    # RUN SNN TOOLBOX #
    ###################

    main(config_filepath)
