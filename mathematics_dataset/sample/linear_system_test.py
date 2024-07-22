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

"""Tests for mathematics_dataset.sample.linear_system."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# Dependency imports
from absl.testing import absltest
from absl.testing import parameterized
from mathematics_dataset.sample import linear_system
from six.moves import range
import sympy


class ExpressionWithValueTest(parameterized.TestCase):

    def testIsTrivialIn(self):
        self.assertEqual(linear_system._is_trivial_in([[1]], 0), False)
        self.assertEqual(linear_system._is_trivial_in([[1, 2], [3, 4]], 0), False)
        self.assertEqual(linear_system._is_trivial_in([[1, 2], [3, 0]], 0), True)
        self.assertEqual(linear_system._is_trivial_in([[1, 2], [3, 0]], 1), False)
        self.assertEqual(linear_system._is_trivial_in([[1, 2], [0, 3]], 0), False)
        self.assertEqual(linear_system._is_trivial_in([[1, 2], [0, 3]], 1), True)

    @parameterized.parameters([1, 2, 3])
    def testLinearSystem(self, degree):
        for _ in range(100):  # test a few times
            target = [random.randint(-100, 100) for _ in range(degree)]
            variables = [sympy.Symbol(chr(ord("a") + i)) for i in range(degree)]
            system = linear_system.linear_system(
                variables=variables, solutions=target, entropy=10.0
            )
            solved = sympy.solve(system, variables)
            solved = [solved[symbol] for symbol in variables]
            self.assertEqual(target, solved)


if __name__ == "__main__":
    absltest.main()
