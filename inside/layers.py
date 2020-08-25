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

"""Custom layers for the project."""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Lambda, Layer
from tensorflow.keras.regularizers import l2

from inside import config


@tf.function
def _film_reshape(gamma, beta, x):
    """Reshape gamma and beta for FiLM."""
    gamma = tf.tile(
        tf.reshape(gamma, (tf.shape(gamma)[0], 1, 1, tf.shape(gamma)[-1])),
        (1, tf.shape(x)[1], tf.shape(x)[2], 1))
    beta = tf.tile(
        tf.reshape(beta, (tf.shape(beta)[0], 1, 1, tf.shape(beta)[-1])),
        (1, tf.shape(x)[1], tf.shape(x)[2], 1))

    return gamma, beta


class FiLM(Layer):
    """General Conditioning Layer (FiLM) by (Perez et al., 2017)."""
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.channels = input_shape[0][-1]  # input_shape: [x, z].

        self.fc_1 = Dense(int(self.channels / 2))
        self.fc_2 = Dense(int(self.channels / 2))
        self.fc_3 = Dense(int(2 * self.channels))

    def call(self, inputs):
        x, z = inputs

        gamma, beta = self.hypernetwork(z)
        gamma, beta = _film_reshape(gamma, beta, x)

        return gamma * x + beta

    def hypernetwork(self, inputs):
        x = tf.nn.tanh(self.fc_1(inputs))
        x = tf.nn.tanh(self.fc_2(x))
        x = self.fc_3(x)

        return x[..., :self.channels], x[..., self.channels:]


class GuidingBlock(Layer):
    """Guiding Block by (Rupprecht et al., 2018)."""
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.height, self.width, self.channels = input_shape[0][1:]  # NHWC.

        self.fc_1 = Dense(int(self.channels / 2))
        self.fc_2 = Dense(int(self.channels / 2))
        self.fc_3 = Dense(int(2 * self.channels + self.height + self.width))

    def call(self, inputs):
        x, z = inputs

        gamma_b, gamma_s, alpha, beta = self.hypernetwork(z)

        # Spatial attention shared across feature maps.
        alpha = tf.tile(tf.reshape(
            alpha, (tf.shape(alpha)[0], tf.shape(alpha)[-1], 1, 1)),
            (1, 1, tf.shape(x)[2], 1))
        beta = tf.tile(tf.reshape(
            beta, (tf.shape(beta)[0], 1, tf.shape(beta)[-1], 1)),
            (1, tf.shape(x)[1], 1, 1))

        # FiLM-like conditioning.
        gamma_b, gamma_s = _film_reshape(gamma_b, gamma_s, x)

        return alpha + beta, (1 + alpha + beta + gamma_s) * x + gamma_b

    def compute_output_shape(self, input_shape):
        return [input_shape, input_shape]

    def hypernetwork(self, inputs):
        x = tf.nn.tanh(self.fc_1(inputs))
        x = tf.nn.tanh(self.fc_2(x))
        x = self.fc_3(x)

        # FiLM.
        gamma_b = x[..., :self.channels]
        gamma_s = x[..., self.channels:self.channels * 2]

        # Attention.
        alpha = x[..., self.channels * 2:self.channels * 2 + self.height]
        beta = x[..., self.channels * 2 + self.height:]

        return gamma_b, gamma_s, alpha, beta


class INSIDE(Layer):
    """INstance modulation with Spatial DEpendency (INSIDE)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def attention(self, mu, sigma, shape, index):
        """1-D Gaussian Attention; one attention per channel.

        :param mu:
            (batch_size, channels) of Gaussian means.

        :param sigma:
            (batch_size, channels) of Gaussian standard deviations (std).

        :param shape:
            Shape (4, ) of input feature maps (batch_size, X, Y, channels).

        :param index:
            Index (int) of the axis to create an attention for.
        """
        _arrange = (1, shape[index], 1, 1) if index == 1 else \
            (1, 1, shape[index], 1)

        x = tf.cast(K.tile(K.reshape(K.arange(0, shape[index]), _arrange),
                    (shape[0], 1, 1, shape[-1])),
                    tf.float32)

        # Calculate relative coordinates.
        shape = tf.cast(shape, tf.float32)
        mu = mu * shape[index] / 2 + shape[index] / 2 - 0.5
        sigma = sigma * 3.5 + K.epsilon()  # We set the max. width here.

        # 1-D Attention.
        mask = K.exp(-.5 * K.square(
            (x - K.reshape(mu, (K.shape(mu)[0], 1, 1, K.shape(mu)[1]))) /
            K.reshape(sigma, (K.shape(sigma)[0], 1, 1, (K.shape(sigma)[1])))))

        return mask

    def build(self, input_shape):
        self.channels = input_shape[0][-1]  # input_shape: [x, z].

        self.fc_1 = Dense(int(self.channels / 2))
        self.fc_2 = Dense(int(self.channels / 2))
        self.fc_3 = Dense(int(4 * self.channels))  # scale, shift, mean.
        self.fc_4 = Dense(
            int(2 * self.channels),
            activity_regularizer=l2(config.INSIDE_PENELTY))  # std.

    def call(self, inputs):
        x, z = inputs

        scale, shift, mu_a, sigma_a, mu_b, sigma_b = self.hypernetwork(z)

        # Gaussian attention.
        a_x = self.attention(mu_a, sigma_a, K.shape(x), 1)
        a_y = self.attention(mu_b, sigma_b, K.shape(x), 2)

        a_x = tf.transpose(a_x, perm=[0, 3, 1, 2])  # tf.matmul.
        a_y = tf.transpose(a_y, perm=[0, 3, 1, 2])

        a = tf.transpose(tf.matmul(a_x, a_y), perm=[0, 2, 3, 1])

        # FiLM.
        gamma_s, gamma_b = _film_reshape(scale, shift, x)

        return a, gamma_s * (x * a) + gamma_b

    def compute_output_shape(self, input_shape):
        return [input_shape, input_shape]

    def hypernetwork(self, inputs):
        x = tf.nn.tanh(self.fc_1(inputs))
        x = tf.nn.tanh(self.fc_2(x))

        p = self.fc_3(x)  # scale, shift, mean.
        s = tf.nn.tanh(self.fc_4(x))  # std.

        # Scale and shift.
        scale = p[..., :self.channels]
        shift = p[..., self.channels:self.channels * 2]

        # Attention. A_x.
        mu_a = tf.nn.tanh(p[..., self.channels * 2:self.channels * 3])
        sigma_a = s[..., :self.channels]

        # Attention. A_y.
        mu_b = tf.nn.tanh(p[..., self.channels * 3:])
        sigma_b = s[..., self.channels:]

        return scale, shift, mu_a, sigma_a, mu_b, sigma_b
