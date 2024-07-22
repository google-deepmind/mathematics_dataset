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

"""Tests for mathematics_dataset.sample.polynomials."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# Dependency imports
from absl.testing import parameterized
from mathematics_dataset.sample import polynomials
import numpy as np
from six.moves import range
import sympy
import tensorflow as tf


class ExpressionWithValueTest(tf.test.TestCase, parameterized.TestCase):

    def testSplitValueEqually(self):
        split = polynomials._split_value_equally(3, 2)
        self.assertEqual(split, [1, 2])
        split = polynomials._split_value_equally(sympy.sympify("3/4"), 2)
        self.assertEqual(split, [sympy.sympify("1/4"), sympy.sympify("1/2")])

    def testIntegersWithSum(self):
        value = 13
        count = 10
        terms = polynomials.integers_with_sum(value=value, count=count, entropy=4.0)
        self.assertLen(terms, count)
        self.assertEqual(sum(terms), value)

    def testMonomial(self):
        x, y = sympy.symbols("x y")
        self.assertEqual(str(polynomials.monomial(1, [x, y], [2, 3])), "x**2*y**3")
        # TODO(b/124038530): how handle rational coefficients; are they even used?
        # self.assertEqual(
        #     str(polynomials.monomial(sympy.Rational(2, 3), [x], [1])), '2*x/3')
        # self.assertEqual(
        #     str(polynomials.monomial(sympy.Rational(1, 3), [x], [1])), 'x/3')
        self.assertEqual(str(polynomials.monomial(x, [y], [4])), "x*y**4")

    def testExpandCoefficients(self):
        for _ in range(10):
            num_variables = np.random.randint(1, 4)
            degrees = np.random.randint(0, 4, [num_variables])
            coefficients = np.random.randint(-3, 3, degrees + 1)
            entropy = np.random.uniform(0, 10)
            expanded = polynomials.expand_coefficients(coefficients, entropy)
            collapsed = np.vectorize(sum)(expanded)
            self.assertAllEqual(coefficients, collapsed)

    def testCoefficientsToPolynomial(self):
        coeffs = [3, 2, 1]
        x = sympy.Symbol("x")
        polynomial = polynomials.coefficients_to_polynomial(coeffs, [x])
        polynomial = sympy.sympify(polynomial)
        self.assertEqual(polynomial, x * x + 2 * x + 3)

    def testUnivariate(self):
        # Test generation for: x**2 + 2*x + 1
        x = sympy.Symbol("x")
        coeffs = [1, 2, 3]
        for _ in range(10):
            expanded = polynomials.expand_coefficients(coeffs, 5.0)
            polynomial = polynomials.coefficients_to_polynomial(expanded, [x])
            sympified = sympy.sympify(polynomial)
            self.assertEqual(sympified, 1 + 2 * x + 3 * x * x)

    def testMultivariate(self):
        # Test generation for: x**2 + 2*x*y + 3*y**2 - x + 5
        x, y = sympy.symbols("x y")
        coeffs = [[5, 0, 3], [-1, 2, 0], [1, 0, 0]]
        for _ in range(10):
            expanded = polynomials.expand_coefficients(coeffs, 5.0, length=10)
            polynomial = polynomials.coefficients_to_polynomial(expanded, [x, y])
            sympified = sympy.sympify(polynomial)
            self.assertEqual(sympified, x * x + 2 * x * y + 3 * y * y - x + 5)

    def testAddCoefficients(self):
        # Add x**2 + 2*y and 3*x + 4*y**3.
        coeffs1 = [[0, 2], [0, 0], [1, 0]]
        coeffs2 = [[0, 0, 0, 4], [3, 0, 0, 0]]
        target = [[0, 2, 0, 4], [3, 0, 0, 0], [1, 0, 0, 0]]
        actual = polynomials.add_coefficients(coeffs1, coeffs2)
        self.assertAllEqual(target, actual)

    def testCoefficientsLinearSplit(self):
        for degree in range(3):
            for ndims in range(3):
                for _ in range(10):
                    coefficients = np.random.randint(-5, 5, [degree + 1] * ndims)
                    entropy = random.uniform(1, 4)
                    c1, c2, coeffs1, coeffs2 = polynomials.coefficients_linear_split(
                        coefficients, entropy
                    )
                    c1 = int(c1)
                    c2 = int(c2)
                    coeffs1 = np.asarray(coeffs1, dtype=np.int32)
                    coeffs2 = np.asarray(coeffs2, dtype=np.int32)
                    sum_ = c1 * coeffs1 + c2 * coeffs2
                    self.assertAllEqual(sum_, coefficients)

    def testSampleWithBrackets(self):
        x, y = sympy.symbols("x y")
        for _ in range(100):
            degrees = np.random.randint(1, 4, [2])
            entropy = random.uniform(0, 4)
            polynomial = polynomials.sample_with_brackets(
                variables=[x, y], degrees=degrees, entropy=entropy
            )
            self.assertIn("(", str(polynomial))
            poly = sympy.poly(sympy.sympify(polynomial).expand())
            self.assertEqual(poly.degree(x), degrees[0])
            self.assertEqual(poly.degree(y), degrees[1])

    def testTrim(self):
        self.assertAllEqual(polynomials.trim([1]), [1])
        self.assertAllEqual(polynomials.trim([1, 0]), [1])
        self.assertAllEqual(polynomials.trim([0, 1]), [0, 1])
        self.assertAllEqual(polynomials.trim([0]), [])
        self.assertAllEqual(polynomials.trim([0, 0]), [])

    def testDifferentiate_univariate(self):
        coeffs = [5, 3, 2]
        expected = [3, 4]
        actual = polynomials.differentiate(coeffs, 0)
        self.assertAllEqual(expected, actual)

    def testDifferentiate_multivariate(self):
        coeffs = [[0, 3, 1], [5, 0, 0], [0, 2, 0]]
        expected = [[5, 0], [0, 4]]
        actual = polynomials.differentiate(coeffs, 0)
        self.assertAllEqual(expected, actual)

    def testIntegrate_univariate(self):
        coeffs = [5, 3, 2]
        expected = [0, 5, sympy.Rational(3, 2), sympy.Rational(2, 3)]
        actual = polynomials.integrate(coeffs, 0)
        self.assertAllEqual(expected, actual)

    def testIntegrate_multivariate(self):
        coeffs = [[0, 1], [1, 0]]
        expected = [[0, 0, sympy.Rational(1, 2)], [0, 1, 0]]
        actual = polynomials.integrate(coeffs, 1)
        self.assertAllEqual(expected, actual)


if __name__ == "__main__":
    tf.test.main()
