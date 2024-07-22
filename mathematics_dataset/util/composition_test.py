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

"""Tests for mathematics_dataset.util.composition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import absltest
from mathematics_dataset.util import composition
import sympy


class FunctionHandleTest(absltest.TestCase):

    def testApply(self):
        handle = composition.FunctionHandle("f", "g")
        applied = handle.apply(*sympy.symbols("x y"))
        self.assertEqual(str(applied), "f(g(x, y))")
        applied = handle.apply(sympy.symbols("x"))
        self.assertEqual(str(applied), "f(g(x))")


class ContextTest(absltest.TestCase):

    def testPeel(self):
        sample_args = composition.SampleArgs(4, 3.0)
        entropy, new_sample_args = sample_args.peel()
        self.assertAlmostEqual(entropy, 0.75)
        self.assertEqual(new_sample_args.num_modules, 4)
        self.assertAlmostEqual(new_sample_args.entropy, 2.25)

    def testSplit(self):
        sample_args = composition.SampleArgs(4, 5.0)
        children = sample_args.split(2)
        self.assertLen(children, 2)
        self.assertEqual(sum([child.num_modules for child in children]), 3)
        self.assertAlmostEqual(sum([child.entropy for child in children]), 5.0)


class EntityTest(absltest.TestCase):

    def testInit_valueErrorIfSelfAndHandle(self):
        with self.assertRaisesRegex(self, ValueError, "Cannot specify handle"):
            composition.Entity(
                context=composition.Context(),
                value=0,
                description="Something with {self}. ",
                handle="additional",
            )


if __name__ == "__main__":
    absltest.main()
