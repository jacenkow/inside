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

"""Implementation of a simple CNN for CLEVR-Seg experiments."""

from inside.layers import FiLM, GuidingBlock, INSIDE
from inside.models.common import ProjectModel


class EncoderDecoderFiLM(ProjectModel):
    """Encoder-Decoder network with a FiLM layer in the bottleneck."""
    def __init__(self):
        super().__init__()
        self.conditioning = FiLM()

    def call(self, x):
        x = self.encoder(x['x'])
        x = self.conditioning([x, x['z']])

        return self.decoder(x)


class EncoderDecoderGuideMe(ProjectModel):
    """Encoder-Decoder network with a single Guiding Block layer."""
    def __init__(self):
        super().__init__()
        self.conditioning = GuidingBlock()

    def call(self, x):
        x = self.encoder(x['x'])
        x = self.conditioning([x, x['z']])

        return self.decoder(x)


class EncoderDecoderINSIDE(ProjectModel):
    """Encoder-Decoder network with an INSIDE layer in the bottleneck."""
    def __init__(self):
        super().__init__()
        self.conditioning = INSIDE()

    def attention(self, x):
        """Return attention map and Gaussians' sigmas."""
        return self.encoder_call(x)[0:3]

    def call(self, x):
        """Predict segmentation masks."""
        return self.decoder(self.encoder_call(x)[3])

    def encoder_call(self, x):
        # Segmentation branch.
        x = self.encoder(x['x'])
        a, x = self.conditioning([x, x['z']])

        return a, x

    def inference(self, x):
        attentions = self.attention(x)[1:3]  # Sigmas.
        predictions = self.call(x)  # Segmentation masks.

        return attentions, predictions

    def loss(self, y_true, y_pred):
        return self.model_losses['focal'](y_true, y_pred[1])


class SimpleCNN(ProjectModel):
    """Simple CNN."""
    def __init__(self):
        super().__init__()

    def call(self, x):
        return self.decoder(self.encoder(x))
