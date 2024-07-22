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

"""Tests for mathematics_dataset.util.display."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import absltest
from mathematics_dataset.util import display
import sympy


class DecimalTest(absltest.TestCase):

    def testBasic_integer(self):
        decimal = display.Decimal(123)
        self.assertEqual(str(decimal), "123")
        self.assertEqual(sympy.sympify(decimal), sympy.Integer(123))
        self.assertEqual(decimal.decimal_places(), 0)

    def testBasic_ten(self):
        decimal = display.Decimal(10)
        self.assertEqual(str(decimal), "10")
        self.assertEqual(sympy.sympify(decimal), sympy.Integer(10))
        self.assertEqual(decimal.decimal_places(), 0)

    def testBasic(self):
        decimal = display.Decimal(sympy.Rational(123, 100))
        self.assertEqual(str(decimal), "1.23")
        self.assertEqual(sympy.sympify(decimal), sympy.Rational(123, 100))
        self.assertEqual(decimal.decimal_places(), 2)

    def testStr(self):
        self.assertEqual(str(display.Decimal(sympy.Rational(0, 10))), "0")
        self.assertEqual(str(display.Decimal(sympy.Rational(-1, 10))), "-0.1")
        self.assertEqual(str(display.Decimal(sympy.Rational(-11, 10))), "-1.1")
        self.assertEqual(str(display.Decimal(sympy.Rational(11, 10))), "1.1")
        self.assertEqual(str(display.Decimal(sympy.Rational(101, 1))), "101")
        self.assertEqual(
            str(display.Decimal(sympy.Rational(20171, 1000000))), "0.020171"
        )

    def testStr_verySmall(self):
        # Tests it doesn't display in "scientific" notation 1E-9.
        decimal = display.Decimal(sympy.Rational(1, 1000000000))
        self.assertEqual(str(decimal), "0.000000001")

    def testAdd(self):
        self.assertEqual((display.Decimal(2) + display.Decimal(3)).value, 5)

    def testSub(self):
        self.assertEqual((display.Decimal(2) - display.Decimal(3)).value, -1)

    def testMul(self):
        self.assertEqual((display.Decimal(2) * display.Decimal(3)).value, 6)

    def testRound(self):
        decimal = display.Decimal(sympy.Rational(2675, 1000))  # 2.675
        self.assertEqual(sympy.sympify(decimal.round()), sympy.Integer(3))
        self.assertEqual(sympy.sympify(decimal.round(1)), sympy.Rational(27, 10))
        self.assertEqual(sympy.sympify(decimal.round(2)), sympy.Rational(268, 100))
        self.assertEqual(sympy.sympify(decimal.round(3)), sympy.Rational(2675, 1000))

    def testInt(self):
        decimal = display.Decimal(123)
        self.assertEqual(int(decimal), 123)

    def testInt_errorIfNonInt(self):
        decimal = display.Decimal(sympy.Rational(1, 2))
        with self.assertRaisesRegex(self, TypeError, "Cannot represent"):
            int(decimal)

    def testComparison(self):
        decimal = display.Decimal(sympy.Rational(-1, 2))
        # pylint: disable=g-generic-assert
        self.assertFalse(decimal != -0.5)
        self.assertTrue(decimal != 0)
        self.assertFalse(decimal < -0.5)
        self.assertTrue(decimal < 0)
        self.assertTrue(decimal <= -0.5)
        self.assertTrue(decimal <= 0)
        self.assertFalse(decimal > -0.5)
        self.assertTrue(decimal > -1)
        self.assertTrue(decimal >= -0.5)
        self.assertFalse(decimal >= 0)
        self.assertFalse(decimal == 0)
        self.assertTrue(decimal == -0.5)

    def testNegation(self):
        decimal = display.Decimal(sympy.Rational(1, 2))
        decimal = -decimal
        self.assertNotEqual(decimal, 0.5)
        self.assertEqual(decimal, -0.5)


class PercentageTest(absltest.TestCase):

    def testPercentage(self):
        percentage = display.Percentage(1.5)
        self.assertEqual(str(percentage), "150%")

        percentage = display.Percentage(sympy.Rational(67, 100))
        self.assertEqual(str(percentage), "67%")

        percentage = display.Percentage(sympy.Rational(67, 1000))
        self.assertEqual(str(percentage), "6.7%")


class NonSimpleRationalTest(absltest.TestCase):

    def testBasic(self):
        frac = display.NonSimpleRational(4, 6)
        self.assertEqual(frac.numer, 4)
        self.assertEqual(frac.denom, 6)
        self.assertEqual(str(frac), "4/6")


class StringNumberTest(absltest.TestCase):

    def testIntegerToWords(self):
        words = display.StringNumber(0)
        self.assertEqual(str(words), "zero")
        self.assertEqual(sympy.sympify(words), 0)

        words = display.StringNumber(8)
        self.assertEqual(str(words), "eight")
        self.assertEqual(sympy.sympify(words), 8)

        words = display.StringNumber(12)
        self.assertEqual(str(words), "twelve")
        self.assertEqual(sympy.sympify(words), 12)

        words = display.StringNumber(30)
        self.assertEqual(str(words), "thirty")
        self.assertEqual(sympy.sympify(words), 30)

        words = display.StringNumber(100)
        self.assertEqual(str(words), "one-hundred")
        self.assertEqual(sympy.sympify(words), 100)

        words = display.StringNumber(103)
        self.assertEqual(str(words), "one-hundred-and-three")
        self.assertEqual(sympy.sympify(words), 103)

        words = display.StringNumber(15439822)
        self.assertEqual(
            str(words),
            "fifteen-million-four-hundred-and-thirty-nine"
            "-thousand-eight-hundred-and-twenty-two",
        )
        self.assertEqual(sympy.sympify(words), 15439822)

    def testRationalToWords(self):
        words = display.StringNumber(sympy.Rational(2, 3))
        self.assertEqual(str(words), "two thirds")


class StringOrdinalTest(absltest.TestCase):

    def testBasic(self):
        ordinal = display.StringOrdinal(0)
        self.assertEqual(str(ordinal), "zeroth")
        ordinal = display.StringOrdinal(10)
        self.assertEqual(str(ordinal), "tenth")

    def testCreate_errorIfNegative(self):
        with self.assertRaisesRegex(self, ValueError, "Unsupported ordinal"):
            display.StringOrdinal(-1)


class NumberListTest(absltest.TestCase):

    def testBasic(self):
        numbers = [2, 3, 1]
        number_list = display.NumberList(numbers)
        string = str(number_list)
        self.assertEqual(string, "2, 3, 1")


class NumberInBaseTest(absltest.TestCase):

    def testBasic(self):
        self.assertEqual(str(display.NumberInBase(1, 10)), "1")
        self.assertEqual(str(display.NumberInBase(-1, 10)), "-1")
        self.assertEqual(str(display.NumberInBase(1, 2)), "1")
        self.assertEqual(str(display.NumberInBase(-1, 2)), "-1")
        self.assertEqual(str(display.NumberInBase(2, 2)), "10")
        self.assertEqual(str(display.NumberInBase(-2, 2)), "-10")
        self.assertEqual(str(display.NumberInBase(10, 16)), "a")
        self.assertEqual(str(display.NumberInBase(16, 16)), "10")
        self.assertEqual(str(display.NumberInBase(256, 16)), "100")
        self.assertEqual(str(display.NumberInBase(-75483, 10)), "-75483")


if __name__ == "__main__":
    absltest.main()
