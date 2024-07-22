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

"""Tests for mathematics_dataset.sample.arithmetic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# Dependency imports
from absl.testing import absltest
from absl.testing import parameterized
from mathematics_dataset.sample import arithmetic
from mathematics_dataset.sample import number
from mathematics_dataset.sample import ops
from six.moves import range
import sympy


class ArithmeticTest(parameterized.TestCase):

    def testArithmetic(self):
        for _ in range(1000):
            target = number.integer_or_rational(4, signed=True)
            entropy = 8.0
            expression = arithmetic.arithmetic(target, entropy)
            self.assertEqual(sympy.sympify(expression), target)

    def testArithmeticLength(self):
        """Tests that the generated arithmetic expressions have given length."""
        for _ in range(1000):
            target = number.integer_or_rational(4, signed=True)
            entropy = 8.0
            length = random.randint(2, 10)
            expression = arithmetic.arithmetic(target, entropy, length)
            # Note: actual length is #ops = #numbers - 1.
            actual_length = len(ops.number_constants(expression)) - 1
            self.assertEqual(actual_length, length)


if __name__ == "__main__":
    absltest.main()
