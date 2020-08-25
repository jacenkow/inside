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

"""Callbacks for training."""

import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
)

from inside import config


class PlotPredictions(Callback):
    """Generate segmentation masks."""
    def __init__(self, folder):
        super().__init__()
        self.comet_ml = None
        self.folder = folder
        self.model = None
        self.network = config.EXPERIMENT_NETWORK

    def on_train_begin(self):
        """Setup folder in case does not exist."""
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

    def on_epoch_end(self, epoch, images, labels):
        """Predict a batch and plot image, predictions and confidence maps."""
        if not epoch % 10:  # Log every 10th epoch.
            fig = self.plot(images, labels)
            self.comet_ml.log_figure(
                figure_name="epoch.{:0>3d}.png".format(epoch),
                figure=fig)
            fig.savefig(os.path.join(self.folder,
                                     "epoch.{:0>3d}.png".format(epoch)))

    def on_test_end(self, images, labels):
        """Predict on the first mini-batch of the test set."""
        fig = self.plot(images, labels)
        self.comet_ml.log_figure(figure_name="test.png", figure=fig)
        fig.savefig(os.path.join(self.folder, "test.png"))

    def plot(self, images, labels):
        """Predict and plot first five instances."""
        try:
            x = {"x": images['x'][:5], "z": images['z'][:5]}
        except TypeError:
            x = images[:5]

        predictions = self.model(x)

        if self.network in ["guideme", "inside"]:
            attentions = self.model.attention(x)
            fig, axs = plt.subplots(ncols=5, nrows=5, figsize=(15, 15))
        else:
            fig, axs = plt.subplots(ncols=4, nrows=5, figsize=(10, 15))

        for index in range(5):
            try:  # Image and conditioning information.
                axs[index, 0].imshow(images['x'][index, :, :])
            except TypeError:  # Image-only input.
                axs[index, 0].imshow(images[index, :, :])

            axs[index, 0].set_title("Input")

            axs[index, 1].imshow(np.rint(predictions[index, :, :, 0]))
            axs[index, 1].set_title("Prediction")

            axs[index, 2].imshow(predictions[index, :, :, 0])
            axs[index, 2].set_title("Confidence Map")

            axs[index, 3].imshow(labels[index, :, :, 0])
            axs[index, 3].set_title("Ground Truth")

            if self.network in ["guideme", "inside"]:
                axs[index, 4].imshow(np.mean(
                    attentions[index, :, :, :], axis=-1))
                axs[index, 4].set_title("Attention")

        fig.tight_layout()

        return fig


def setup_callbacks():
    """Get standard set of preconfigured callbacks."""
    _weights_folder = os.path.join(config.EXPERIMENT_FOLDER, "weights")

    if not os.path.exists(_weights_folder):
        os.mkdir(_weights_folder)

    cl = CSVLogger(os.path.join(config.EXPERIMENT_FOLDER, "logs.csv"))
    es = EarlyStopping(min_delta=config.EARLY_STOPPING_DELTA,
                       patience=config.EARLY_STOPPING_PATIENCE,
                       monitor="validation_loss")
    mc = ModelCheckpoint(os.path.join(_weights_folder, "epoch.{epoch:03d}.h5"),
                         mode="min", monitor="validation_loss",
                         save_best_only=True,
                         save_weights_only=True,
                         verbose=1)
    pp = PlotPredictions(os.path.join(config.EXPERIMENT_FOLDER, "images"))

    return cl, es, mc, pp
