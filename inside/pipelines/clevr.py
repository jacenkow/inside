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

"""Training and evaluation pipeline for the networks."""

import csv
import os

import tensorflow as tf
from tensorflow.keras.metrics import Mean

from inside import config
from inside.callbacks import setup_callbacks
from inside.constructor import setup_comet_ml, setup_model
from inside.loaders import CLEVR
from inside.metrics import DiceScore


def _write_results(logs):
    """Write final logs to a CSV file."""
    w = csv.writer(open(os.path.join(
        config.EXPERIMENT_FOLDER, "results.csv"), "w"))
    for key, val in logs.items():
        w.writerow([key, val])


class Pipeline:
    def __init__(self):
        # Model.
        self.model = setup_model()

        # Comet.ml experiment.
        self.comet_ml = setup_comet_ml()

        # Testing metrics.
        self.test_dice = DiceScore(name="testing_dice")
        self.test_loss = Mean(name="testing_loss")

        # Training metrics.
        self.training_dice = DiceScore(name="training_dice")
        self.training_loss = Mean(name="training_loss")

        # Callbacks.
        self.cl, self.es, self.mc, self.pp = setup_callbacks()
        self.cl.model, self.es.model, self.mc.model = \
            self.model, self.model, self.model

        self.pp.model = self.model
        self.pp.comet_ml = self.comet_ml

    def fit(self):
        """Train the model."""
        # Toy dataset.
        loader = CLEVR()
        train_ds, valid_ds, test_ds = loader.load()

        with self.comet_ml.train():
            self.cl.on_train_begin()
            self.es.on_train_begin()
            self.mc.on_train_begin()
            self.pp.on_train_begin()

            for epoch in range(config.EXPERIMENT_EPOCHS):
                self.comet_ml.set_epoch(epoch)

                for images, labels in train_ds:
                    self.train_step(images, labels)

                for batch, (images, labels) in enumerate(valid_ds):
                    self.test_step(images, labels)

                    if not batch:  # Log only first mini-batch from an epoch.
                        self.pp.on_epoch_end(epoch, images, labels)

                # Get results.
                logs = {
                    "dice": self.training_dice.result().numpy(),
                    "loss": self.training_loss.result().numpy(),
                    "validation_dice": self.test_dice.result().numpy(),
                    "validation_loss": self.test_loss.result().numpy(),
                }

                template = ("Epoch {}. Training Loss: {}. Training Dice: {}. "
                            "Validation Loss: {}. Validation Dice: {}.")

                print(template.format(epoch + 1,
                                      logs['loss'],
                                      logs['dice'],
                                      logs['validation_loss'],
                                      logs['validation_dice']))

                # Log metrics.
                self.comet_ml.log_metrics(logs, epoch=epoch)
                self.cl.on_epoch_end(epoch, logs)
                self.es.on_epoch_end(epoch, logs)
                self.mc.on_epoch_end(epoch, logs)

                # Reset the metrics for the next epoch.
                self.training_dice.reset_states()
                self.training_loss.reset_states()
                self.test_dice.reset_states()
                self.test_loss.reset_states()

                # Early stopping criterion.
                if self.es.model.stop_training:
                    self.cl.on_train_end()
                    self.es.on_train_end()
                    self.mc.on_train_end()
                    break

        with self.comet_ml.test():
            for batch, (images, labels) in enumerate(test_ds):
                self.test_step(images, labels)

                if not batch:
                    self.pp.on_test_end(images, labels)

            # Get results.
            logs = {
                "dice": self.test_dice.result().numpy(),
                "loss": self.test_loss.result().numpy(),
            }

            print("Test Loss: {}. Test Dice: {}.".format(
                logs['loss'], logs['dice']))

            # Log metrics.
            self.comet_ml.log_metrics(logs)
            _write_results(logs)

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model.inference(images)
            loss = self.model.loss(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimiser.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        self.training_loss(loss)
        self.training_dice(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model.inference(images)
        t_loss = self.model.loss(labels, predictions)

        self.test_loss(t_loss)
        self.test_dice(labels, predictions)
