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

"""Tests for mathematics_dataset.sample.ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import absltest
from mathematics_dataset.sample import ops
from six.moves import range
import sympy


class OpsTest(absltest.TestCase):

    def testNeg(self):
        op = ops.Neg(2)
        self.assertEqual(str(op), "-2")
        self.assertEqual(op.sympy(), -2)

        op = ops.Add(ops.Neg(2), 3)
        self.assertEqual(str(op), "-2 + 3")
        self.assertEqual(op.sympy(), 1)

        op = ops.Add(3, ops.Neg(2))
        self.assertEqual(str(op), "3 - 2")
        self.assertEqual(op.sympy(), 1)

        op = ops.Add(ops.Add(ops.Neg(2), 5), 3)
        self.assertEqual(str(op), "-2 + 5 + 3")
        self.assertEqual(op.sympy(), 6)

        op = ops.Add(3, ops.Add(ops.Identity(ops.Neg(2)), 5))
        self.assertEqual(str(op), "3 - 2 + 5")
        self.assertEqual(op.sympy(), 6)

        op = ops.Add(3, ops.Add(2, ops.Neg(5)))
        self.assertEqual(str(op), "3 + 2 - 5")
        self.assertEqual(op.sympy(), 0)

    def testAdd(self):
        add = ops.Add()
        self.assertEqual(str(add), "0")
        self.assertEqual(add.sympy(), 0)

        add = ops.Add(2, 3)
        self.assertEqual(str(add), "2 + 3")
        self.assertEqual(add.sympy(), 5)

        add = ops.Add(ops.Add(1, 2), 3)
        self.assertEqual(str(add), "1 + 2 + 3")
        self.assertEqual(add.sympy(), 6)

    def testSub(self):
        sub = ops.Sub(2, 3)
        self.assertEqual(str(sub), "2 - 3")
        self.assertEqual(sub.sympy(), -1)

        sub = ops.Sub(ops.Sub(1, 2), 3)
        self.assertEqual(str(sub), "1 - 2 - 3")
        self.assertEqual(sub.sympy(), -4)

        sub = ops.Sub(1, ops.Sub(2, 3))
        self.assertEqual(str(sub), "1 - (2 - 3)")
        self.assertEqual(sub.sympy(), 2)

        sub = ops.Sub(ops.Neg(1), 2)
        self.assertEqual(str(sub), "-1 - 2")
        self.assertEqual(sub.sympy(), -3)

    def testMul(self):
        mul = ops.Mul()
        self.assertEqual(str(mul), "1")
        self.assertEqual(mul.sympy(), 1)

        mul = ops.Mul(2, 3)
        self.assertEqual(str(mul), "2*3")
        self.assertEqual(mul.sympy(), 6)

        mul = ops.Mul(ops.Identity(ops.Constant(-2)), 3)
        self.assertEqual(str(mul), "-2*3")
        self.assertEqual(mul.sympy(), -6)

        mul = ops.Mul(ops.Add(1, 2), 3)
        self.assertEqual(str(mul), "(1 + 2)*3")
        self.assertEqual(mul.sympy(), 9)

        mul = ops.Mul(ops.Mul(2, 3), 5)
        self.assertEqual(str(mul), "2*3*5")
        self.assertEqual(mul.sympy(), 30)

        # TODO(b/124038946): reconsider how we want brackets in these cases:

    #     mul = ops.Mul(ops.Div(2, 3), 5)
    #     self.assertEqual(str(mul), '(2/3)*5')
    #     self.assertEqual(mul.sympy(), sympy.Rational(10, 3))
    #
    #     mul = ops.Mul(sympy.Rational(2, 3), 5)
    #     self.assertEqual(str(mul), '(2/3)*5')
    #     self.assertEqual(mul.sympy(), sympy.Rational(10, 3))

    def testDiv(self):
        div = ops.Div(2, 3)
        self.assertEqual(str(div), "2/3")
        self.assertEqual(div.sympy(), sympy.Rational(2, 3))

        div = ops.Div(2, sympy.Rational(4, 5))
        self.assertEqual(str(div), "2/(4/5)")
        self.assertEqual(div.sympy(), sympy.Rational(5, 2))

        div = ops.Div(1, ops.Div(2, 3))
        self.assertEqual(str(div), "1/(2/3)")
        self.assertEqual(div.sympy(), sympy.Rational(3, 2))

        div = ops.Div(ops.Div(2, 3), 4)
        self.assertEqual(str(div), "(2/3)/4")
        self.assertEqual(div.sympy(), sympy.Rational(1, 6))

        div = ops.Div(2, ops.Mul(3, 4))
        self.assertEqual(str(div), "2/(3*4)")

        div = ops.Div(2, sympy.Function("f")(sympy.Symbol("x")))
        self.assertEqual(str(div), "2/f(x)")

    def testPow(self):
        pow_ = ops.Pow(2, 3)
        self.assertEqual(str(pow_), "2**3")
        self.assertEqual(pow_.sympy(), 8)

        pow_ = ops.Pow(4, sympy.Rational(1, 2))
        self.assertEqual(str(pow_), "4**(1/2)")
        self.assertEqual(pow_.sympy(), 2)

        pow_ = ops.Pow(sympy.Rational(1, 2), 3)
        self.assertEqual(str(pow_), "(1/2)**3")
        self.assertEqual(pow_.sympy(), 1 / 8)

        pow_ = ops.Pow(3, ops.Pow(2, 1))
        self.assertEqual(str(pow_), "3**(2**1)")
        self.assertEqual(pow_.sympy(), 9)

        pow_ = ops.Pow(ops.Pow(2, 3), 4)
        self.assertEqual(str(pow_), "(2**3)**4")
        self.assertEqual(pow_.sympy(), 4096)

        pow_ = ops.Pow(-5, 2)
        self.assertEqual(str(pow_), "(-5)**2")
        self.assertEqual(pow_.sympy(), 25)

    def testEq(self):
        op = ops.Eq(ops.Add(2, 3), 4)
        self.assertEqual(str(op), "2 + 3 = 4")
        self.assertEqual(op.sympy(), False)

    def testDescendants(self):
        constants = [ops.Constant(i) for i in range(6)]

        # (1 + 2*3**4) / 5 - 6
        expression = ops.Sub(
            ops.Div(
                ops.Add(
                    constants[0],
                    ops.Mul(constants[1], ops.Pow(constants[2], constants[3])),
                ),
                constants[4],
            ),
            constants[5],
        )
        descendants = expression.descendants()
        descendants = ops._flatten(descendants)

        for constant in constants:
            self.assertIn(constant, descendants)
            self.assertEqual(descendants.count(constant), 1)

        # Also test top-level.
        self.assertEqual(constants[0].descendants(), [constants[0]])

        # Also general structure.
        constant = ops.Constant(3)
        expression = ops.Neg(constant)
        self.assertEqual(set(expression.descendants()), set([constant, expression]))

    def testNumberConstants(self):
        constant = ops.Constant(3)
        expression = ops.Neg(constant)
        constants = ops.number_constants([expression])
        self.assertEqual(constants, [constant])


if __name__ == "__main__":
    absltest.main()
