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

"""Utility for train/test split based on hash value."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib


def is_train(value):
    """Returns whether `value` should be used in a training question."""
    value_as_string = str(value).encode("utf-8")
    return int(hashlib.md5(value_as_string).hexdigest(), 16) % 2 == 0
