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

"""Additional losses."""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss


class DiceLoss(Loss):
    """Implements binary Dice loss function."""
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE,
                 name="dice_loss"):
        super().__init__(name=name, reduction=reduction)

    @tf.function
    def call(self, y_true, y_pred):
        import pdb; pdb.set_trace()
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))

        return 1 - (numerator / (denominator + K.epsilon()))


class MeanPrediction(Loss):
    """Implements mean of predictions as a loss function."""
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE,
                 name="mean_pred_loss"):
        super().__init__(name=name, reduction=reduction)

    @tf.function
    def call(self, y_true, y_pred):
        return tf.reduce_mean(y_pred, axis=[0, 2])
