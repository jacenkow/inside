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

"""Configuration file."""

# Comet.ml
COMET_API_KEY = ":)"
COMET_WORKSPACE = ":)"

# Datasets.
CLEVR_ENCODING = {
    "colour": {
        "alizarin": [1, 0, 0],
        "nephritis": [0, 1, 0],
        "orange": [0, 0, 1],
    },
    "coordinates": {
        "top_left": [1, 0, 0, 0],
        "top_right": [0, 1, 0, 0],
        "bottom_right": [0, 0, 1, 0],
        "bottom_left": [0, 0, 0, 1],
    },
    "shape": {
        "cube": [1, 0, 0],
        "sphere": [0, 1, 0],
        "triangle": [0, 0, 1],
    },
    "size": {
        "small": [1, 0, 0],
        "medium": [0, 1, 0],
        "large": [0, 0, 1],
    }
}

# Project.
CONFIGURATION_YAMLS = "inside/configurations"
DATASETS_FOLDER = "inside/datasets"
EXPERIMENTS_FOLDER = "inside/experiments"

# TensorFlow.
TF_DEFAULT_DTYPE = "float32"
