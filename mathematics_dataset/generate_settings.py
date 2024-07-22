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

"""Settings for generation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import string

MAX_QUESTION_LENGTH = 160
MAX_ANSWER_LENGTH = 30
QUESTION_CHARS = ["", " "] + list(
    string.ascii_letters + string.digits + string.punctuation
)
EMPTY_INDEX = QUESTION_CHARS.index("")
NUM_INDICES = len(QUESTION_CHARS)
CHAR_TO_INDEX = {char: index for index, char in enumerate(QUESTION_CHARS)}
INDEX_TO_CHAR = {index: char for index, char in enumerate(QUESTION_CHARS)}
