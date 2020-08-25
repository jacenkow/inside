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

import numpy as np
import tensorflow as tf

from inside.layers import FiLM, GuidingBlock, INSIDE


def test_film():
    """Test FiLM layer.

    FiLM(x, gamma, beta) = gamma * x + beta.
    """
    x = tf.convert_to_tensor(np.ones((1, 3, 3, 1)), dtype=tf.int32)
    gamma = tf.Variable([[2]], dtype=tf.int32)
    beta = tf.Variable([[1]], dtype=tf.int32)

    np.testing.assert_allclose(x * 3, FiLM()([x, gamma, beta]))


def test_guiding_block():
    """Test Guiding Block from Guide Me.

    GuidingBlock(x, gamma_s, gamma_n, alpha, beta) = \
        (1 + alpha + beta + gamma_s) * x + gamma_b
    """
    x = tf.convert_to_tensor(np.ones((1, 3, 4, 1)), dtype=tf.int32)

    gamma_s = tf.Variable([[8]], dtype=tf.int32)
    gamma_b = tf.Variable([[9]], dtype=tf.int32)
    alpha = tf.Variable([[1, 2, 3]], dtype=tf.int32)
    beta = tf.Variable([[4, 5, 6, 7]], dtype=tf.int32)

    result = [[[[23], [24], [25], [26]],
               [[24], [25], [26], [27]],
               [[25], [26], [27], [28]]]]

    np.testing.assert_allclose(result, GuidingBlock()(
        [x, gamma_s, gamma_b, alpha, beta]))


def test_inside():
    """Test INSIDE layer."""
    x = tf.convert_to_tensor(np.ones((1, 5, 5, 2)), dtype=tf.float32)

    gamma_s = tf.Variable([[1]], dtype=tf.float32)
    gamma_b = tf.Variable([[1]], dtype=tf.float32)

    mu_a = tf.Variable([[-1, 1]], dtype=tf.float32)
    sigma_a = tf.Variable([[0.1, 0.1]], dtype=tf.float32)
    mu_b = tf.Variable([[-1, 1]], dtype=tf.float32)
    sigma_b = tf.Variable([[0.1, 0.1]], dtype=tf.float32)

    result = [[[[1.5, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
               [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
               [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
               [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
               [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1.5]]]]

    np.testing.assert_allclose(result, INSIDE()(
        [x, gamma_s, gamma_b, mu_a, sigma_a, mu_b, sigma_b])[1], rtol=0.1)
