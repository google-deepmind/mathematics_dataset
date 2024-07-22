# Copyright 2018 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The various mathematics modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mathematics_dataset.modules import algebra
from mathematics_dataset.modules import arithmetic
from mathematics_dataset.modules import calculus
from mathematics_dataset.modules import comparison
from mathematics_dataset.modules import measurement
from mathematics_dataset.modules import md_numbers
from mathematics_dataset.modules import polynomials
from mathematics_dataset.modules import probability
import six


all_ = {
    "algebra": algebra,
    "arithmetic": arithmetic,
    "calculus": calculus,
    "comparison": comparison,
    "measurement": measurement,
    "numbers": md_numbers,
    "polynomials": polynomials,
    "probability": probability,
}


def train(entropy_fn):
    """Returns dict of training modules."""
    return {name: module.train(entropy_fn) for name, module in six.iteritems(all_)}


def test():
    """Returns dict of testing modules."""
    return {name: module.test() for name, module in six.iteritems(all_)}


def test_extra():
    """Returns dict of extrapolation testing modules."""
    return {name: module.test_extra() for name, module in six.iteritems(all_)}
