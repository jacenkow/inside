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

"""Data loader for CLEVR-Seg dataset."""

from copy import deepcopy
from glob import glob
import json
import os

import numpy as np
from PIL import Image
from tensorflow import data
from tqdm import tqdm

from inside import config


def _cast(x):
    """Cast to float32 as `Mul` operation requires this type."""
    return np.asarray(x, np.float32)


class CLEVR:
    def __init__(self):
        self.conditioning = config.EXPERIMENT_Z

        if self.conditioning and self.conditioning not in \
                config.CLEVR_ENCODING.keys():
            raise ValueError("Wrong conditioning scenario selected.")

        self.dataset = os.path.join(config.DATASETS_FOLDER, "clevr")
        self.shape = (4000, 200, 200, 3)  # RGB.
        self.splits = (0.2, 0.72, 0.08)  # Test, training, validation sets.

    def combine(self, filename):
        """Combine image with segmentation mask given the conditioning."""
        image = np.array(Image.open(
            os.path.join(self.dataset, "images", filename + ".png")))[:, :, 0:3]
        image = np.divide(image, 255)

        mask = np.zeros((self.shape[1], self.shape[2], 1))

        _counter = 0
        _masks = self.load_masks(filename).values()

        while True:
            # Randomly select a single colour, shape or size.
            choice = np.bincount(np.random.randint(0, len(
                config.CLEVR_ENCODING[self.conditioning].items()),
                    150)).argmax()

            z = list(config.CLEVR_ENCODING[self.conditioning].values())[choice]

            # Combine all masks that meet the condition.
            for m in _masks:
                if m[self.conditioning] == list(config.CLEVR_ENCODING[
                        self.conditioning].keys())[choice]:
                    mask = np.add(mask, m['mask'])

            if mask.any() or _counter == 100:
                break
            else:
                _counter += 1

        return image, mask, z

    def generate(self):
        """Generate `.npy` file with the datasets."""
        _template = {"images": [], "masks": [], "z": []}
        testing_filenames, training_filenames, validation_filenames = \
            self.split()
        testing_set, training_set, validation_set = \
            deepcopy(_template), deepcopy(_template), deepcopy(_template)

        for filenames, dataset in zip(
                [testing_filenames, training_filenames, validation_filenames],
                [testing_set, training_set, validation_set]):
            for filename in tqdm(filenames):
                _image, _mask, _z = self.combine(filename)
                dataset['images'].append(_image)
                dataset['masks'].append(_mask)
                dataset['z'].append(_z)

        np.savez_compressed(os.path.join(self.dataset, "dataset.npz"), {
            "conditioning": self.conditioning,
            "testing": testing_set,
            "training": training_set,
            "validation": validation_set
        })

    def load(self):
        """Load the dataset."""
        if not os.path.exists(self.dataset):
            self.generate()

        dataset = np.load(os.path.join(self.dataset, "dataset.npz"),
                          allow_pickle=True)['arr_0'][()]

        if config.EXPERIMENT_Z:
            assert dataset['conditioning'] == config.EXPERIMENT_Z
            modality = "z"

        # Testing images.
        x_test = _cast(dataset['testing']['images'])
        if config.EXPERIMENT_Z:
            x_test = {"x": x_test, "z": _cast(dataset['testing'][modality])}

        test_ds = data.Dataset.from_tensor_slices((
            x_test, _cast(dataset['testing']['masks']))).batch(
            config.EXPERIMENT_BATCH_SIZE)

        # Training images.
        x_train = _cast(dataset['training']['images'])
        if config.EXPERIMENT_Z:
            x_train = {"x": x_train, "z": _cast(dataset['training'][modality])}

        train_ds = data.Dataset.from_tensor_slices((
            x_train, _cast(dataset['training']['masks'])))
        train_ds = train_ds.shuffle(10000).batch(config.EXPERIMENT_BATCH_SIZE)

        # Validation images.
        x_valid = _cast(dataset['validation']['images'])
        if config.EXPERIMENT_Z:
            x_valid = {"x": x_valid,
                       "z": _cast(dataset['validation'][modality])}

        valid_ds = data.Dataset.from_tensor_slices((
            x_valid, _cast(dataset['validation']['masks'])))
        valid_ds = valid_ds.shuffle(10000).batch(config.EXPERIMENT_BATCH_SIZE)

        return train_ds, valid_ds, test_ds

    def load_json(self, filename):
        """Load description of the scene."""
        with open(os.path.join(
                self.dataset, "scenes", filename + ".json"), "r") as file:
            scene = json.load(file)

        description = {}

        for i, object in enumerate(scene['objects']):
            _center = object['pixel_coords'][0:2]
            _coordinates = "top_" if _center[1] < (self.shape[2] / 2) \
                else "bottom_"
            _coordinates += "left" if _center[0] < (self.shape[1] / 2) \
                else "right"

            description[i] = {
                "coordinates": _coordinates,
                "colour": object['color'],
                "shape": object['shape'],
                "size": object['size'],
            }

        return description

    def load_masks(self, filename):
        """Load segmentation mask with scene description."""
        scene = self.load_json(filename)

        for i, path in enumerate(sorted(glob(
                os.path.join(self.dataset, "segmentations", filename, "*")))):
            # Skip alpha channel and reduce dimensionality.
            _mask = np.array(Image.open(path))[:, :, 0:3]
            _mask = np.apply_along_axis(lambda x: x[0], -1, _mask)

            # Binarise.
            assert np.any(_mask == 64)  # Background.
            assert np.any(_mask == 255)  # Object.

            _mask[_mask == 64] = 0
            _mask[_mask == 255] = 1

            scene[i]['mask'] = _mask[..., np.newaxis]

        return scene

    def split(self):
        """Split filenames into testing. training, validation datasets."""
        testing_set, training_set, validation_set = [], [], []

        for i, filename in enumerate(sorted(
                glob(os.path.join(self.dataset, "images", "*")))):
            _name = filename.split("/")[-1][:-4]
            _training_slice = int(self.shape[0] * self.splits[1])
            _validation_slice = _training_slice + int(
                self.shape[0] * self.splits[2])

            if i < _training_slice:
                training_set.append(_name)
            elif _training_slice <= i < _validation_slice:
                validation_set.append(_name)
            else:
                testing_set.append(_name)

        assert len(testing_set) == int(self.shape[0] * self.splits[0])
        assert len(training_set) == int(self.shape[0] * self.splits[1])
        assert len(validation_set) == int(self.shape[0] * self.splits[2])

        return testing_set, training_set, validation_set
