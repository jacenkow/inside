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

from argparse import ArgumentParser

from inside.constructor import setup_experiment


def run():
    """Setup experiment environment and train the selected model."""
    parser = ArgumentParser()
    parser.add_argument("-c", "--configuration", type=str, required=True,
                        help="Configuration filename.")
    args = parser.parse_args()
    setup_experiment(args.configuration)

    from inside.pipelines import Pipeline  # Comet.ml and seed first.
    experiment = Pipeline()
    experiment.fit()


if __name__ == "__main__":
    run()
