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

"""Tests for mathematics_dataset.modules.arithmetic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import absltest
from mathematics_dataset.modules import arithmetic
import sympy


class ArithmeticTest(absltest.TestCase):

    def testSurdCoefficients(self):
        exp = sympy.sympify("1")
        self.assertEqual(arithmetic._surd_coefficients(exp), (1, 0))

        exp = sympy.sympify("1/2")
        self.assertEqual(arithmetic._surd_coefficients(exp), (1 / 2, 0))

        exp = sympy.sympify("sqrt(2)")
        self.assertEqual(arithmetic._surd_coefficients(exp), (0, 1))

        exp = sympy.sympify("3*sqrt(2)")
        self.assertEqual(arithmetic._surd_coefficients(exp), (0, 3))

        exp = sympy.sympify("3*sqrt(5)/2")
        self.assertEqual(arithmetic._surd_coefficients(exp), (0, 3 / 2))

        exp = sympy.sympify("1 + 3 * sqrt(2)")
        self.assertEqual(arithmetic._surd_coefficients(exp), (1, 3))

        exp = sympy.sympify("1/2 + 3 * sqrt(5) / 2")
        self.assertEqual(arithmetic._surd_coefficients(exp), (1 / 2, 3 / 2))

        exp = sympy.sympify("sqrt(2)/(-1 + 2*sqrt(2))**2")
        self.assertEqual(arithmetic._surd_coefficients(exp), (8 / 49, 9 / 49))


if __name__ == "__main__":
    absltest.main()
