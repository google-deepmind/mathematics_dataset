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

"""Tests for mathematics_dataset.sample.number."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# Dependency imports
from absl.testing import absltest
from absl.testing import parameterized
from mathematics_dataset.sample import number
from six.moves import range
import sympy


class NumberTest(parameterized.TestCase):

    def testCoprimeDensity(self):
        self.assertEqual(number._coprime_density(1), 1.0)
        self.assertEqual(number._coprime_density(2), 0.5)
        self.assertLess(abs(number._coprime_density(3) - 2 / 3), 1e-6)
        self.assertLess(abs(number._coprime_density(6) - 1 / 3), 1e-6)

    @parameterized.parameters(False, True)
    def testInteger_allowZero(self, signed):
        saw_zero = False
        saw_nonzero = False
        for _ in range(1000):
            sample = number.integer(1, signed=signed)
            if sample == 0:
                saw_zero = True
            else:
                saw_nonzero = True
            if saw_zero and saw_nonzero:
                break
        self.assertTrue(saw_zero)
        self.assertTrue(saw_nonzero)

    def testNonIntegerRational(self):
        for _ in range(1000):
            entropy = random.uniform(0, 10)
            signed = random.choice([False, True])
            sample = number.non_integer_rational(entropy, signed)
            self.assertNotEqual(sympy.denom(sample), 1)

    @parameterized.parameters(False, True)
    def testIntegerOrRational(self, signed):
        # Tests we can call it. Do it a few times so both code paths get executed.
        for _ in range(10):
            number.integer_or_rational(2, signed)

    def testNonIntegerDecimal(self):
        for _ in range(1000):
            sample = number.non_integer_decimal(1, False)
            self.assertNotEqual(sympy.denom(sample), 1)
            self.assertLen(str(sample), 3)  # should be of form "0.n"
            self.assertGreater(sample, 0)  # positive

    def testNonIntegerDecimal_size(self):
        saw_bigger_one = False
        saw_smaller_one = False
        for _ in range(1000):
            sample = number.non_integer_decimal(2, False)
            if sample > 1:
                saw_bigger_one = True
            else:
                saw_smaller_one = True
            if saw_bigger_one and saw_smaller_one:
                break
        self.assertTrue(saw_bigger_one)
        self.assertTrue(saw_smaller_one)

    @parameterized.parameters(
        lambda: number.integer(0, True),
        lambda: number.integer(1, True),
        lambda: number.non_integer_rational(2, True),
        lambda: number.non_integer_decimal(1, True),
    )
    def testGenerate_signed(self, generator):
        saw_positive = False
        saw_negative = False
        for _ in range(1000):
            sample = generator()
            saw_positive |= sample > 0
            saw_negative |= sample < 0
            if saw_positive and saw_negative:
                break

        self.assertTrue(saw_positive)
        self.assertTrue(saw_negative)

    @parameterized.parameters(
        lambda: number.integer(2, False), lambda: number.non_integer_rational(2, False)
    )
    def testIntegerRational_distinctCount(self, generator):
        seen = set()
        for _ in range(3000):
            seen.add(generator())
        self.assertGreaterEqual(len(seen), 10**2)

    @parameterized.parameters(number.integer, number.non_integer_decimal)
    def testEntropyOfValue(self, generator):
        for entropy in [1, 2, 4, 8, 16]:
            sum_entropy = 0.0
            count = 2000
            for _ in range(count):
                value = generator(entropy, signed=True)
                sum_entropy += number.entropy_of_value(value)
            avg_entropy = sum_entropy / count
            error = abs(entropy - avg_entropy) / entropy
            self.assertLess(error, 0.2)


if __name__ == "__main__":
    absltest.main()
