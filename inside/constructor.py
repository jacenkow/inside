# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Grzegorz Jacenków.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Experiment constructor to setup an environment."""

from datetime import datetime
import os
from shutil import copy

from comet_ml import Experiment
import yaml

from inside import config


def _load_yaml(configuration_name):
    """Load .yaml configuration file."""
    try:
        with open(os.path.join(config.CONFIGURATION_YAMLS,
                               configuration_name + ".yml"), "r") as stream:
            return yaml.safe_load(stream)
    except FileNotFoundError:
        raise ValueError("The configuration file does not exist.")


def _parse_configuration(configuration_name):
    """Load the selected configuration and inject into the project."""
    _configuration = _load_yaml(configuration_name)

    # Comet.ml.
    config.COMET_MONITOR = _configuration['cometml']['monitor']
    config.COMET_PROJECT_NAME = _configuration['cometml']['project']

    # Early stopping.
    config.EARLY_STOPPING_DELTA = _configuration['early_stopping']['delta']
    config.EARLY_STOPPING_PATIENCE = \
        _configuration['early_stopping']['patience']

    # Experiment.
    if "alpha" in _configuration['experiment'].keys():  # Loss coefficient.
        config.EXPERIMENTS_ALPHA = _configuration['experiment']['alpha']

    config.EXPERIMENT_BATCH_SIZE = _configuration['experiment']['batch_size']
    config.EXPERIMENT_DATASET = _configuration['experiment']['dataset']
    config.EXPERIMENT_EPOCHS = _configuration['experiment']['epochs']
    config.EXPERIMENT_NAME = _configuration['experiment']['name']
    config.EXPERIMENT_NETWORK = _configuration['experiment']['network']

    if "z" in _configuration['experiment'].keys():  # Conditioning `z`.
        config.EXPERIMENT_Z = _configuration['experiment']['z']
    else:
        config.EXPERIMENT_Z = None

    # GPUs.
    config.ENVIRONMENT_GPUS = _configuration['enviornment']['gpus']
    config.ENVIRONMENT_SEED = _configuration['enviornment']['seed']


def _set_gpus():
    """Set available GPUs for the experiment."""
    os.environ["CUDA_VISIBLE_DEVICES"] = config.ENVIRONMENT_GPUS


def _set_seed():
    """Set the seed."""
    import numpy as np
    np.random.seed(config.ENVIRONMENT_SEED)


def _setup_experiment_folder(configuration_name):
    """Create a new folder for the experiment."""
    _experiment_folder = os.path.join(
        config.EXPERIMENTS_FOLDER,
        config.EXPERIMENT_NAME + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S"))

    if not os.path.exists(_experiment_folder):
        os.mkdir(_experiment_folder)

    # Copy the configuration file.
    copy(os.path.join(config.CONFIGURATION_YAMLS,
                      configuration_name + ".yml"), _experiment_folder)

    config.EXPERIMENT_FOLDER = _experiment_folder


def _setup_tensorflow():
    """Initialise TensorFlow and override the default configuration."""
    import tensorflow as tf
    tf.keras.backend.set_floatx(config.TF_DEFAULT_DTYPE)


def setup_comet_ml():
    """Initialise Experiment object."""
    experiment = Experiment(
        api_key=config.COMET_API_KEY,
        disabled=True if not config.COMET_MONITOR else False,
        log_code=False,
        project_name=config.COMET_PROJECT_NAME,
        workspace=config.COMET_WORKSPACE,
    )

    experiment.set_name(config.EXPERIMENT_NAME)
    experiment.log_others({
        "conditioning": config.EXPERIMENT_Z,
        "dataset": config.EXPERIMENT_DATASET,
    })
    experiment.log_parameters({
        "batch_size": config.EXPERIMENT_BATCH_SIZE,
        "epochs": config.EXPERIMENT_EPOCHS,
    })

    return experiment


def setup_experiment(configuration_name):
    """Setup experiment environment."""
    _parse_configuration(configuration_name)
    _set_gpus()
    _set_seed()
    _setup_experiment_folder(configuration_name)
    _setup_tensorflow()


def setup_model():
    """Return model based on the configuration file."""
    from inside.models.clevr import (
        EncoderDecoderFiLM,
        EncoderDecoderGuideMe,
        EncoderDecoderINSIDE,
        SimpleCNN,
    )

    if config.EXPERIMENT_NETWORK == "film":
        return EncoderDecoderFiLM()
    elif config.EXPERIMENT_NETWORK == "guideme":
        return EncoderDecoderGuideMe()
    elif config.EXPERIMENT_NETWORK == "inside":
        return EncoderDecoderINSIDE()
    elif config.EXPERIMENT_NETWORK == "simple":
        return SimpleCNN()
    else:
        raise ValueError("Wrong network selected.")
