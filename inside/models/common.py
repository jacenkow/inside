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

"""Common patterns."""

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

from inside import config


class ProjectModel(Model):
    """Abstract class for the project-related models."""
    def __init__(self):
        super().__init__()
        self.stop_training = False  # Callbacks' monitoring.
        self.z = config.EXPERIMENT_Z

        # Optimiser and losses.
        self.optimiser = Adam()
        self.model_losses = {"focal": SigmoidFocalCrossEntropy()}

        # Segmentation layers.
        self.encoder = ShallowEncoder()
        self.decoder = ShallowDecoder()

    def inference(self, x):
        return self.call(x)

    def loss(self, y_true, y_pred):
        return self.model_losses['focal'](y_true, y_pred)


class ShallowDecoder(Model):
    """Shallow decoder for binary prediction."""
    def __init__(self):
        super().__init__()

        self.conv2a = Conv2D(64, 3, activation="relu", padding="same")
        self.conv2b = Conv2D(32, 3, activation="relu", padding="same")
        self.conv2c = Conv2D(16, 3, activation="relu", padding="same")
        self.conv2d = Conv2D(1, 1, activation="sigmoid", padding="same")

    def call(self, x):
        x = self.conv2a(x)
        x = UpSampling2D()(x)
        x = self.conv2b(x)
        x = UpSampling2D()(x)
        x = self.conv2c(x)

        return self.conv2d(x)


class ShallowEncoder(Model):
    """Shallow encoder."""
    def __init__(self):
        super().__init__()

        self.conv2a = Conv2D(16, 3, activation="relu", padding="same")
        self.conv2b = Conv2D(32, 3, activation="relu", padding="same")
        self.conv2c = Conv2D(64, 3, activation="relu", padding="same")

    def call(self, x):
        x = self.conv2a(x)
        x = MaxPooling2D()(x)
        x = self.conv2b(x)
        x = MaxPooling2D()(x)

        return self.conv2c(x)
