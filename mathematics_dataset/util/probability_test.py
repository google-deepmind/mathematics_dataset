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

"""Tests for mathematics_dataset.util.probability."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import absltest
from mathematics_dataset.util import probability
import sympy


class FiniteProductEventTest(absltest.TestCase):

    def testAllSequences(self):
        event = probability.FiniteProductEvent(
            [probability.DiscreteEvent({1, 2}), probability.DiscreteEvent({3})]
        )
        all_sequences = [i for i in event.all_sequences()]
        self.assertEqual(all_sequences, [(1, 3), (2, 3)])


class CountLevelSetEventTest(absltest.TestCase):

    def testAllSequences(self):
        event = probability.CountLevelSetEvent({"a": 2, "b": 4, "c": 1})
        all_sequences = event.all_sequences()

        # Number of sequences should be 7! / (4! * 2! * 1!) = 105.
        self.assertLen(all_sequences, 105)
        # They should all be unique.
        self.assertEqual(len(all_sequences), len(set(all_sequences)))
        # And check contains one correctly generated tuple.
        self.assertIn(("a", "b", "c", "b", "b", "a", "b"), all_sequences)


class DiscreteProbabilitySpaceTest(absltest.TestCase):

    def testBasic(self):
        space = probability.DiscreteProbabilitySpace({0: 1, 1: 2, 2: 3})
        p = space.probability(probability.DiscreteEvent([0]))
        self.assertEqual(p, sympy.Rational(1, 6))
        p = space.probability(probability.DiscreteEvent([0, 1]))
        self.assertEqual(p, sympy.Rational(1, 2))
        p = space.probability(probability.DiscreteEvent([0, 1, 2]))
        self.assertEqual(p, 1)
        p = space.probability(probability.DiscreteEvent([0, 1, 2, 3]))
        self.assertEqual(p, 1)
        p = space.probability(probability.DiscreteEvent([3]))
        self.assertEqual(p, 0)


class FiniteProductSpaceTest(absltest.TestCase):

    def testProbability_FiniteProductEvent(self):
        # 5 coin flips of a biased coin with heads prob = 1/3.
        base_space = probability.DiscreteProbabilitySpace({"h": 1, "t": 2})
        space = probability.FiniteProductSpace([base_space] * 5)

        heads = probability.DiscreteEvent({"h"})
        tails = probability.DiscreteEvent({"t"})
        event = probability.FiniteProductEvent([heads, heads, tails, tails, heads])
        self.assertEqual(space.probability(event), sympy.Rational(4, 3**5))

    def testProbability_CountLevelSetEvent(self):
        base_space = probability.DiscreteProbabilitySpace({"a": 2, "b": 3, "c": 5})
        space = probability.FiniteProductSpace([base_space] * 12)
        event = probability.CountLevelSetEvent({"a": 7, "b": 2, "c": 3})

        # Probability should be (12 choose 7 2 3) * p(a)^7 p(b)^2 p(c)^3
        coeff = 7920
        p_a = sympy.Rational(1, 5)
        p_b = sympy.Rational(3, 10)
        p_c = sympy.Rational(1, 2)
        self.assertEqual(
            space.probability(event), coeff * pow(p_a, 7) * pow(p_b, 2) * pow(p_c, 3)
        )


class SampleWithoutReplacementSpaceTest(absltest.TestCase):

    def testBasic(self):
        space = probability.SampleWithoutReplacementSpace({0: 1, 1: 1}, 2)
        event_0_0 = probability.FiniteProductEvent(
            [probability.DiscreteEvent({0}), probability.DiscreteEvent({0})]
        )
        event_0_1 = probability.FiniteProductEvent(
            [probability.DiscreteEvent({0}), probability.DiscreteEvent({1})]
        )
        p_0_0 = space.probability(event_0_0)
        p_0_1 = space.probability(event_0_1)
        self.assertEqual(p_0_0, 0)
        self.assertEqual(p_0_1, sympy.Rational(1, 2))

        space = probability.SampleWithoutReplacementSpace({0: 1, 1: 0}, 1)
        event_0 = probability.FiniteProductEvent([probability.DiscreteEvent({0})])
        event_1 = probability.FiniteProductEvent([probability.DiscreteEvent({1})])
        event_2 = probability.FiniteProductEvent([probability.DiscreteEvent({2})])
        p_0 = space.probability(event_0)
        p_1 = space.probability(event_1)
        p_2 = space.probability(event_2)
        self.assertEqual(p_0, 1)
        self.assertEqual(p_1, 0)
        self.assertEqual(p_2, 0)


class DiscreteRandomVariableTest(absltest.TestCase):

    def testCall(self):
        random_variable = probability.DiscreteRandomVariable({1: 1, 2: 3, 3: 4})
        forwards = random_variable(probability.DiscreteEvent({1, 3}))
        self.assertEqual(forwards.values, {1, 4})

    def testInverse(self):
        random_variable = probability.DiscreteRandomVariable({1: 1, 2: 3, 3: 4})
        inverse = random_variable.inverse(probability.DiscreteEvent({1, 3}))
        self.assertEqual(inverse.values, {1, 2})

        random_variable = probability.DiscreteRandomVariable({1: 1, 2: 1})
        inverse = random_variable.inverse(probability.DiscreteEvent({1, 5}))
        self.assertEqual(inverse.values, {1, 2})


class FiniteProductRandomVariableTest(absltest.TestCase):

    def _random_variable(self):
        rv1 = probability.DiscreteRandomVariable({1: "a", 2: "b", 3: "c"})
        rv2 = probability.DiscreteRandomVariable({1: "x", 2: "y", 3: "x"})
        return probability.FiniteProductRandomVariable((rv1, rv2))

    def testCall_FiniteProductEvent(self):
        rv = self._random_variable()
        event1 = probability.DiscreteEvent({1, 2})
        event2 = probability.DiscreteEvent({1, 3})
        event = probability.FiniteProductEvent((event1, event2))
        result = rv(event)
        self.assertIsInstance(result, probability.FiniteProductEvent)
        self.assertLen(result.events, 2)
        self.assertEqual(result.events[0].values, {"a", "b"})
        self.assertEqual(result.events[1].values, {"x"})

    def testInverse_FiniteProductEvent(self):
        rv = self._random_variable()
        event1 = probability.DiscreteEvent({"a", "b"})
        event2 = probability.DiscreteEvent({"x"})
        event = probability.FiniteProductEvent((event1, event2))
        result = rv.inverse(event)
        self.assertIsInstance(result, probability.FiniteProductEvent)
        self.assertLen(result.events, 2)
        self.assertEqual(result.events[0].values, {1, 2})
        self.assertEqual(result.events[1].values, {1, 3})

    def testInverse_CountLevelSetEvent(self):
        rv = self._random_variable()
        event = probability.CountLevelSetEvent({"a": 1, "x": 1})
        result = rv.inverse(event)
        sequences = result.all_sequences()
        self.assertLen(sequences, 2)
        self.assertEqual(set(sequences), {(1, 1), (1, 3)})


if __name__ == "__main__":
    absltest.main()
