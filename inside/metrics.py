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

"""Additional metrics."""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric

from inside import config


class DiceScore(Metric):
    """Dice score for evaluation."""
    def __init__(self, name="dice_score", **kwargs):
        super(DiceScore, self).__init__(name=name, **kwargs)
        self.count = self.add_weight(initializer="zeros", name="count")
        self.dices = self.add_weight(initializer="zeros", name="dices")

    def update_state(self, y_true, y_pred):
        if config.EXPERIMENT_NETWORK in ["guideme", "inside"]:
            y_pred = y_pred[1]

        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
        result = numerator / (denominator + K.epsilon())

        self.count.assign_add(tf.size(result, out_type=tf.float32))
        self.dices.assign_add(tf.reduce_sum(result))

    def result(self):
        return tf.divide(self.dices, self.count)
