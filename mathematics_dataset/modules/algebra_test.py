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

"""Tests for mathematics_dataset.modules.algebra."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# Dependency imports
from absl.testing import absltest
from mathematics_dataset.modules import algebra
from mathematics_dataset.sample import polynomials
from six.moves import range
import sympy


class AlgebraTest(absltest.TestCase):

    def testPolynomialCoeffsWithRoots(self):
        coeffs = algebra._polynomial_coeffs_with_roots([1, 2], scale_entropy=0.0)
        self.assertEqual(coeffs, [2, -3, 1])

    def testPolynomialRoots(self):
        variable = sympy.Symbol("x")
        for _ in range(10):
            roots = random.sample(list(range(-9, 10)), 3)
            coeffs = algebra._polynomial_coeffs_with_roots(roots, scale_entropy=10.0)
            polynomial = polynomials.coefficients_to_polynomial(coeffs, variable)
            calc_roots = sympy.polys.polytools.real_roots(polynomial)
            self.assertEqual(calc_roots, sorted(roots))


if __name__ == "__main__":
    absltest.main()
