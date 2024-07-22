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

"""Sample arithmetic expressions with a given value."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random

# Dependency imports
from mathematics_dataset.sample import number
from mathematics_dataset.sample import ops
from mathematics_dataset.util import combinatorics
import numpy as np
import six
from six.moves import zip
import sympy


class _SampleArgs(collections.namedtuple("SampleArgs", ("count", "entropy"))):
    """For sampling mathematical expressions."""

    def peel(self, frac=1):
        """Peels one (or `frac`) of an op's entropy."""
        entropy = frac * self.entropy / self.count
        new_sample_args = _SampleArgs(self.count, self.entropy - entropy)
        return entropy, new_sample_args

    def split(self, args):
        """Splits the entropy and op counts up."""
        non_integer_count = sum(not arg.is_Integer for arg in args)
        assert non_integer_count <= self.count - 1
        count_split = combinatorics.uniform_non_negative_integers_with_sum(
            len(args), (self.count - 1) - non_integer_count
        )
        for i, arg in enumerate(args):
            if not arg.is_Integer:
                count_split[i] += 1
        if all(count == 0 for count in count_split):
            assert self.entropy == 0
            entropies = np.zeros(len(count_split))
        else:
            entropies = (
                np.random.dirichlet(np.maximum(1e-9, count_split)) * self.entropy
            )
        return [
            _SampleArgs(op_count, entropy)
            for op_count, entropy in zip(count_split, entropies)
        ]


def _add_sub_filter(value, sample_args):
    return sample_args.count >= 2 or value.is_Integer


def _add_op(value, sample_args, rationals_allowed):
    """Returns sampled args for `ops.Add`."""
    entropy, sample_args = sample_args.peel()
    if rationals_allowed and sample_args.count >= 3:
        x = number.integer_or_rational(entropy, True)
    else:
        x = number.integer(entropy, True)
    if random.choice([False, True]):
        op_args = [x, value - x]
    else:
        op_args = [value - x, x]
    return ops.Add, op_args, sample_args


def _sub_op(value, sample_args, rationals_allowed):
    """Returns sampled args for `ops.Sub`."""
    entropy, sample_args = sample_args.peel()
    if rationals_allowed and sample_args.count >= 3:
        x = number.integer_or_rational(entropy, True)
    else:
        x = number.integer(entropy, True)
    if random.choice([False, True]):
        op_args = [x, x - value]
    else:
        op_args = [value + x, x]
    return ops.Sub, op_args, sample_args


def _entropy_of_factor_split(integer):
    """Returns entropy (log base 10) of decomposing: integer = a * b."""
    assert integer.is_Integer
    if integer == 0:
        return 0
    # Gives dict of form {factor: multiplicity}
    factors = sympy.factorint(integer)
    return sum(math.log10(mult + 1) for mult in six.itervalues(factors))


def _split_factors(integer):
    """Randomly factors integer into product of two integers."""
    assert integer.is_Integer
    if integer == 0:
        return [1, 0]
    # Gives dict of form {factor: multiplicity}
    factors = sympy.factorint(integer)
    left = sympy.Integer(1)
    right = sympy.Integer(1)
    for factor, mult in six.iteritems(factors):
        left_mult = random.randint(0, mult)
        right_mult = mult - left_mult
        left *= factor**left_mult
        right *= factor**right_mult
    return left, right


def _mul_filter(value, sample_args):
    if sample_args.count >= 2:
        return True
    if not value.is_Integer:
        return False
    return sample_args.entropy <= _entropy_of_factor_split(value)


def _mul_op(value, sample_args, rationals_allowed):
    """Returns sampled args for `ops.Mul`."""
    if sample_args.count >= 3:
        _, op_args, sample_args = _div_op(value, sample_args, rationals_allowed)
        op_args = [op_args[0], sympy.Integer(1) / op_args[1]]
    elif sample_args.count == 1:
        entropy, sample_args = sample_args.peel()
        assert _entropy_of_factor_split(value) >= entropy
        op_args = _split_factors(value)
    else:
        assert sample_args.count == 2
        entropy, sample_args = sample_args.peel()
        numer = sympy.numer(value)
        denom = sympy.denom(value)
        p1, p2 = _split_factors(numer)
        entropy -= _entropy_of_factor_split(numer)
        mult = number.integer(entropy, signed=True, min_abs=1, coprime_to=p1)
        op_args = [p1 / (mult * denom), p2 * mult]

    if random.choice([False, True]):
        op_args = list(reversed(op_args))

    return ops.Mul, op_args, sample_args


def _div_filter(value, sample_args):
    del value  # unused
    del sample_args  # unused
    return True


def _div_op(value, sample_args, rationals_allowed):
    """Returns sampled args for `ops.Div`."""
    assert rationals_allowed  # should be True if this function gets invoked
    entropy, sample_args = sample_args.peel()

    numer = sympy.numer(value)
    denom = sympy.denom(value)

    if sample_args.count == 1:
        mult = number.integer(entropy, signed=True, min_abs=1)
        op_args = [numer * mult, denom * mult]
    elif sample_args.count == 2:
        if numer == 0 or random.choice([False, True]):
            x = number.integer(entropy, signed=True, min_abs=1, coprime_to=denom)
            op_args = [sympy.Rational(x * numer, denom), x]
        else:
            x = number.integer(entropy, signed=True, min_abs=1, coprime_to=numer)
            op_args = [x, sympy.Rational(x * denom, numer)]
    else:
        assert sample_args.count >= 3
        p2, p1 = _split_factors(numer)
        q1, q2 = _split_factors(denom)
        entropy -= _entropy_of_factor_split(numer) + _entropy_of_factor_split(denom)
        entropy_r = random.uniform(0, entropy)
        entropy_s = entropy - entropy_r
        r = number.integer(entropy_r, signed=True, min_abs=1, coprime_to=q1 * p2)
        s = number.integer(entropy_s, signed=False, min_abs=1, coprime_to=p1 * q2)
        op_args = [sympy.Rational(r * p1, s * q1), sympy.Rational(r * q2, s * p2)]

    return ops.Div, op_args, sample_args


def _arithmetic(value, sample_args, add_sub, mul_div):
    """Internal arithmetic thingy...."""
    assert sample_args.count >= 0
    if sample_args.count == 0:
        assert sample_args.entropy == 0
        return ops.Constant(value)

    allowed = []
    if add_sub and _add_sub_filter(value, sample_args):
        allowed.append(_add_op)
        allowed.append(_sub_op)
    if mul_div and _mul_filter(value, sample_args):
        allowed.append(_mul_op)
    if mul_div and _div_filter(value, sample_args):
        allowed.append(_div_op)
    if not allowed:
        raise ValueError(
            "No valid ops found, add_sub={} mul_div={} value={} sample_args={}".format(
                add_sub, mul_div, value, sample_args
            )
        )
    choice = random.choice(allowed)

    op, args, sample_args = choice(value, sample_args, rationals_allowed=mul_div)
    sample_args = sample_args.split(args)
    child_expressions = [
        _arithmetic(arg, child_sample_arg, add_sub, mul_div)
        for arg, child_sample_arg in zip(args, sample_args)
    ]

    return op(*child_expressions)


def length_range_for_entropy(entropy):
    """Returns length range to sample from for given entropy."""
    min_length = 3
    max_length = min_length + int(entropy / 2)
    return min_length, max_length


def arithmetic(value, entropy, length=None, add_sub=True, mul_div=True):
    """Generates an arithmetic expression with a given value.

    Args:
      value: Target value (integer or rational).
      entropy: Amount of randomness to use in generating expression.
      length: Number of ops to use. If `None` then suitable length will be picked
          based on entropy by sampling within the range
          `length_range_for_entropy`.
      add_sub: Whether to include addition and subtraction operations.
      mul_div: Whether to include multiplication and division operations.

    Returns:
      Instance of `ops.Op` containing expression.
    """
    assert isinstance(entropy, float)
    if length is None:
        min_length, max_length = length_range_for_entropy(entropy)
        length = random.randint(min_length, max_length)
        # Some entropy used up in sampling the length.
        entropy -= math.log10(max_length - min_length + 1)
    else:
        assert isinstance(length, int)

    # Entropy adjustment, because different binary trees (from sampling ops) can
    # lead to the same expression. This is the correct value when we use just
    # addition as the op, and is otherwise an an upper bound.
    entropy += combinatorics.log_number_binary_trees(length) / math.log(10)

    value = sympy.sympify(value)
    sample_args = _SampleArgs(length, entropy)
    return _arithmetic(value, sample_args, add_sub, mul_div)
