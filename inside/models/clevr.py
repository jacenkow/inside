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
from inside.models.common import ProjectModel, ProjectModelAttention


class EncoderDecoderFiLM(ProjectModel):
    """Encoder-Decoder network with a FiLM layer in the bottleneck."""
    def __init__(self):
        super().__init__()
        self.conditioning = FiLM()

    def call(self, x):
        z = self.encoder(x['x'])
        z = self.conditioning([z, x['z']])

        return self.decoder(z)


class EncoderDecoderGuideMe(ProjectModelAttention):
    """Encoder-Decoder network with a single Guiding Block layer."""
    def __init__(self):
        super().__init__()
        self.conditioning = GuidingBlock()

    def call(self, x):
        """Predict segmentation masks."""
        return self.decoder(self.encoder_call(x)[1])


class EncoderDecoderINSIDE(ProjectModelAttention):
    """Encoder-Decoder network with an INSIDE layer in the bottleneck."""
    def __init__(self):
        super().__init__()
        self.conditioning = INSIDE()

    def call(self, x):
        """Predict segmentation masks."""
        return self.decoder(self.encoder_call(x)[1])


class SimpleCNN(ProjectModel):
    """Simple CNN."""
    def __init__(self):
        super().__init__()

    def call(self, x):
        return self.decoder(self.encoder(x))
