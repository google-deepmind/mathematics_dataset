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

"""Generate polynomials with given values."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random

# Dependency imports
from mathematics_dataset.sample import number
from mathematics_dataset.sample import ops
from mathematics_dataset.util import combinatorics
import numpy as np
import six
from six.moves import range
from six.moves import zip
import sympy
from sympy.solvers.diophantine import (
    base_solution_linear as diophantine_solve_linear_2d,
)


def expanded_coefficient_counts(length, is_zero):
    """Generates list of integers for number of terms of given power.

    Args:
      length: Integer >= `sum(is_zero)`.
      is_zero: List of booleans.

    Returns:
      List of non-negative integers of length `is_zero`, summing to `length`,
      such that if `is_zero[i]` then `return_value[i] != 1`.

    Raises:
      ValueError: If assignment not possible.
    """
    if length == 1 and all(is_zero):
        raise ValueError("length=1 and all zero")

    counts = np.asarray([0 if zero else 1 for zero in is_zero])
    extra_needed = length - sum(counts)

    if extra_needed < 0:
        raise ValueError("length={} cannot handle is_zero={}".format(length, is_zero))

    extra = combinatorics.uniform_non_negative_integers_with_sum(
        count=len(is_zero), sum_=extra_needed
    )
    counts += np.asarray(extra)

    # Tweak so that no zeros get "1".
    while True:
        bad_zeros = [i for i in range(len(is_zero)) if is_zero[i] and counts[i] == 1]
        if not bad_zeros:
            break
        take_from = random.choice(bad_zeros)
        add_to = random.choice(
            [i for i in range(len(is_zero)) if counts[i] >= 1 and i != take_from]
        )
        counts[take_from] -= 1
        counts[add_to] += 1

    return counts


def _split_value_equally(delta, count):
    """Splits an integer or rational into roughly equal parts."""
    numer = sympy.numer(delta)
    denom = sympy.denom(delta)
    return [int(math.floor((numer + i) / count)) / denom for i in range(count)]


def integers_with_sum(value, count, entropy):
    """Returns list of integers with a given sum.

    Args:
      value: Target value.
      count: Integer >= 1; the number of integers to use.
      entropy: Entropy to use (in total).

    Returns:
      List of numbers summing to `value`.

    Raises:
      ValueError: If `value` is not an integer.
    """
    # Special cases.
    if count == 0:
        assert value == 0
        assert entropy == 0
        return []
    if count == 1:
        assert entropy == 0
        return [value]

    if not number.is_integer(value):
        raise ValueError(
            "value={} (type={}) is not an integer".format(value, type(value))
        )

    # Because e.g., (1, 1) and (2, 2) will both map to the same set of integers
    # when we normalize to have sum equal to `value`.
    entropy *= count / (count - 1)

    min_term_entropy = max(1, number.entropy_of_value(int(math.ceil(value / count))))
    term_entropies = entropy * np.random.dirichlet(np.ones(count))
    term_entropies = np.maximum(min_term_entropy, term_entropies)

    terms = [
        number.integer(term_entropy, signed=True) for term_entropy in term_entropies
    ]

    delta = value - sum(terms)
    deltas = _split_value_equally(delta, count)
    terms = [term + delta for term, delta in zip(terms, deltas)]
    random.shuffle(terms)
    return terms


def monomial(coefficient, variables, powers):
    """Makes a simple monomial term."""
    if not isinstance(variables, (list, tuple)):
        variables = [variables]
    if not isinstance(powers, (list, tuple, np.ndarray)):
        powers = [powers]

    terms = []

    for variable, power in zip(variables, powers):
        if power == 0:
            continue
        elif power == 1:
            terms.append(variable)
        else:
            terms.append(ops.Pow(variable, power))

    if not terms or isinstance(coefficient, sympy.Symbol) or abs(coefficient) != 1:
        if isinstance(coefficient, sympy.Symbol):
            terms.insert(0, coefficient)
        else:
            terms.insert(0, abs(coefficient))

    if len(terms) > 1:
        term = ops.Mul(*terms)
    else:
        term = terms[0]

    if not isinstance(coefficient, sympy.Symbol) and coefficient < 0:
        term = ops.Neg(term)

    return term


def sample_coefficients(degrees, entropy, min_non_zero=0, max_non_zero=None):
    """Generates grid of coefficients with shape `degrees + 1`.

    This corresponds to univariate if degrees has length 1, otherwise
    multivariate.

    Args:
      degrees: List of integers containing max degrees of variables.
      entropy: Float >= 0; entropy for generating entries.
      min_non_zero: Optional integer >= 1; the minimum number of non-zero coeffs.
      max_non_zero: Optional integer >= 1; the maximum number of non-zero coeffs.

    Returns:
      NumPy int array of shape `degrees + 1`.
    """
    if isinstance(degrees, int):
        degrees = [degrees]
    degrees = np.asarray(degrees)

    def random_index():
        return [random.randint(0, degrees[i]) for i in range(len(degrees))]

    indices = set()
    # Ensure a variable of degree `degrees[i]` occurs for every axis i.
    for i, degree in enumerate(degrees):
        if degree > 0:
            index = random_index()
            index[i] = degree
            indices.add(tuple(index))

    abs_max_non_zero = np.prod(degrees + 1)

    min_non_zero = max(min_non_zero, 1, len(indices))
    if max_non_zero is None:
        max_non_zero = min_non_zero + int(entropy / 2)

    min_non_zero = min(min_non_zero, abs_max_non_zero)
    max_non_zero = min(max_non_zero, abs_max_non_zero)
    max_non_zero = max(min_non_zero, max_non_zero)

    num_non_zero = random.randint(min_non_zero, max_non_zero)

    while len(indices) < num_non_zero:
        indices.add(tuple(random_index()))

    coeffs = np.zeros(degrees + 1, dtype=np.int64)
    entropies = entropy * np.random.dirichlet(np.ones(num_non_zero))

    for index, entry_entropy in zip(indices, entropies):
        value = number.integer(entry_entropy, signed=True, min_abs=1)
        coeffs.itemset(index, value)

    return coeffs


def expand_coefficients(coefficients, entropy, length=None):
    """Expands coefficients to multiple terms that sum to each coefficient.

    Args:
      coefficients: Array, such that `coefficients[i, j, ..., k]` is the
          coefficient of x**i * y**j * ... * z**k.
      entropy: Float >= 0; the entropy to use for generating extra randomness.
      length: Number of terms that appear, e.g., 2x + 3 has two terms. If `None`
          then a suitable length will be picked depending on the entropy
          requested.

    Returns:
      Numpy object array with the same shape as `coefficients`, containing lists.
    """
    coefficients = np.asarray(coefficients)
    shape = coefficients.shape

    expanded_coefficients = np.empty(shape, dtype=np.object)

    min_length = np.count_nonzero(coefficients) + 2
    if length is None:
        max_length = min_length + int(math.ceil(entropy) / 2)
        length = random.randint(min_length, max_length)
    if length < min_length:
        length = min_length

    is_zero_flat = np.reshape(coefficients, [-1]) == 0
    counts = expanded_coefficient_counts(length, is_zero=is_zero_flat)
    coeffs_entropy = entropy * np.random.dirichlet(np.maximum(1e-9, counts - 1))
    counts = np.reshape(counts, shape)
    coeffs_entropy = np.reshape(coeffs_entropy, shape)

    indices = list(zip(*np.indices(shape).reshape([len(shape), -1])))
    for power in indices:
        coeffs = integers_with_sum(
            value=coefficients.item(power),
            count=counts.item(power),
            entropy=coeffs_entropy.item(power),
        )
        expanded_coefficients.itemset(power, coeffs)

    return expanded_coefficients


def sample_expanded_coefficients(degrees, entropy, length=None):
    """Convenience function: samples and expands coeffs, entropy split equally."""
    coefficients = sample_coefficients(degrees, entropy / 2, max_non_zero=length)
    return expand_coefficients(coefficients, entropy / 2, length)


def coefficients_to_polynomial(coefficients, variables):
    """Converts array of lists of coefficients to a polynomial."""
    coefficients = np.asarray(coefficients)
    shape = coefficients.shape

    indices = list(zip(*np.indices(shape).reshape([len(shape), -1])))
    monomials = []
    for power in indices:
        coeffs = coefficients.item(power)
        if number.is_integer_or_rational(coeffs) or isinstance(coeffs, sympy.Symbol):
            coeffs = [coeffs]
        elif not isinstance(coeffs, list):
            raise ValueError(
                "Unrecognized coeffs={} type={}".format(coeffs, type(coeffs))
            )
        for coeff in coeffs:
            monomials.append(monomial(coeff, variables, power))
    random.shuffle(monomials)
    return ops.Add(*monomials)


def sample(variables, degrees, entropy, length=None):
    coefficients = sample_expanded_coefficients(degrees, entropy, length)
    return coefficients_to_polynomial(coefficients, variables)


def add_coefficients(coeffs1, coeffs2):
    """Adds together two sets of coefficients over same set of variables."""
    coeffs1 = np.asarray(coeffs1)
    coeffs2 = np.asarray(coeffs2)

    degrees1 = np.array(coeffs1.shape)
    degrees2 = np.array(coeffs2.shape)
    assert len(degrees1) == len(degrees2)

    extra1 = np.maximum(0, degrees2 - degrees1)
    extra2 = np.maximum(0, degrees1 - degrees2)

    pad1 = [(0, extra) for extra in extra1]
    pad2 = [(0, extra) for extra in extra2]

    coeffs1 = np.pad(coeffs1, pad1, "constant", constant_values=0)
    coeffs2 = np.pad(coeffs2, pad2, "constant", constant_values=0)

    return coeffs1 + coeffs2


def _random_factor(integer):
    factors = sympy.factorint(integer)
    result = 1
    for factor, power in six.iteritems(factors):
        result *= factor ** random.randint(0, power)
    return result


def coefficients_linear_split(coefficients, entropy):
    """Finds two sets of coefficients and multipliers summing to `coefficients`.

    Given `coefficients` (an integer vector), will sample integers `a, b`, and
    two sets of coefficients `coefficients_1, coefficients_2`, such that
    `a * coefficients_1 + b * coefficients_2 == coefficients`.

    Args:
      coefficients: Array of coefficients.
      entropy: Float >= 0; the amount of randomness used to sample.

    Returns:
      Tuple (a, b, coefficients_1, coefficients_2)`.
    """
    coefficients = np.asarray(coefficients)
    coefficients_shape = coefficients.shape
    coefficients = np.reshape(coefficients, [-1])

    entropy_a = max(1, random.uniform(0, entropy / 3))
    entropy_b = max(1, random.uniform(0, entropy / 3))
    entropy -= entropy_a + entropy_b
    entropy_coefficients = entropy * np.random.dirichlet(np.ones(len(coefficients)))

    # For each target coefficient z, we are required to solve the linear
    # Diophantine equation a*x + b*y = c. Bezout's theorem: this has a solution if
    # and only if gcd(a, b) divides c.
    # Thus to be solvable for all coefficients, a and b must be chosen such that
    # gcd(a, b) divides the gcd of the coefficients.
    coefficients_gcd = sympy.gcd([i for i in coefficients])
    coefficients_gcd = max(1, abs(coefficients_gcd))

    a = number.integer(entropy_a, signed=True, min_abs=1)
    b = number.integer(entropy_b, signed=True, min_abs=1, coprime_to=a)
    b *= _random_factor(coefficients_gcd)
    if random.choice([False, True]):
        a, b = b, a

    coefficients_1 = np.zeros(coefficients.shape, dtype=np.object)
    coefficients_2 = np.zeros(coefficients.shape, dtype=np.object)

    for index, coefficient in enumerate(coefficients):
        entropy_coeff = entropy_coefficients[index]
        t = number.integer(entropy_coeff, signed=True)
        x, y = diophantine_solve_linear_2d(c=coefficient, a=a, b=b, t=t)
        coefficients_1[index] = x
        coefficients_2[index] = y

    # Prevent all coefficients from being zero.
    while np.all(coefficients_1 == 0) or np.all(coefficients_2 == 0):
        index = random.randint(0, len(coefficients) - 1)
        scale = random.choice([-1, 1])
        coefficients_1[index] += scale * b
        coefficients_2[index] -= scale * a

    coefficients_1 = np.reshape(coefficients_1, coefficients_shape)
    coefficients_2 = np.reshape(coefficients_2, coefficients_shape)

    return a, b, coefficients_1, coefficients_2


def _degree_of_variable(polynomial, variable):
    polynomial = sympy.sympify(polynomial).expand()
    if polynomial.is_constant():
        return 0
    polynomial = sympy.poly(polynomial)
    if variable not in polynomial.free_symbols:
        return 0
    return polynomial.degree(variable)


def _sample_with_brackets(
    depth, variables, degrees, entropy, length, force_brackets=True
):
    """Internal recursive function for: constructs a polynomial with brackets."""
    # To generate arbitrary polynomial recursively, can do one of:
    # *   add two polynomials, with at least one having brackets.
    # *   multiply two polynomials.
    # *   call `sample` (i.e., polynomial without brackets).

    if force_brackets:
        length = max(2, length)

    if not force_brackets and (random.choice([False, True]) or length < 2):
        return sample(variables, degrees, entropy, length)

    length_left = random.randint(1, length - 1)
    length_right = length - length_left
    entropy_left, entropy_right = entropy * np.random.dirichlet(
        [length_left, length_right]
    )

    if random.choice([False, True]):
        # Add two. Force brackets on at least one of the polynomials, and sample
        # repeatedly until we don't get cancellation.
        while True:
            left = _sample_with_brackets(
                depth + 1, variables, degrees, entropy_left, length_left, True
            )
            right = _sample_with_brackets(
                depth + 1, variables, degrees, entropy_right, length_right, False
            )
            if random.choice([False, True]):
                left, right = right, left
            result = ops.Add(left, right)
            all_ok = True
            for variable, degree in zip(variables, degrees):
                if _degree_of_variable(result, variable) != degree:
                    all_ok = False
                    break
            if all_ok:
                return result
    else:
        # Multiply two.
        def sample_with_zero_check(degrees_, entropy_, length_):
            while True:
                result = _sample_with_brackets(
                    depth + 1, variables, degrees_, entropy_, length_, False
                )
                if degrees_.sum() > 0 or not result.sympy().is_zero:
                    return result

        degrees = np.asarray(degrees)

        def sample_degree(max_degree):
            """Select in range [0, max_degree], biased away from ends."""
            if max_degree <= 1 or random.choice([False, True]):
                return random.randint(0, max_degree)
            return random.randint(1, max_degree - 1)

        degrees_left = np.array([sample_degree(degree) for degree in degrees])
        degrees_right = degrees - degrees_left
        left = sample_with_zero_check(degrees_left, entropy_left, length_left)
        right = sample_with_zero_check(degrees_right, entropy_right, length_right)
        return ops.Mul(left, right)


def sample_with_brackets(variables, degrees, entropy, length=None):
    """Constructs a polynomial with brackets.

    Args:
      variables: List of variables to use.
      degrees: Max degrees of variables. This function guarantees that these will
          be obtained in the returned polynomial.
      entropy: Float >= 0; the randomness to use in generating the polynomial.
      length: Optional integer containing number of terms. If `None` then an
          appropriate one will be generated depending on the entropy.

    Returns:
      Instance of `ops.Op` containing the polynomial.
    """
    if isinstance(degrees, int):
        degrees = [degrees]
    if not isinstance(variables, (list, tuple)):
        variables = [variables]

    if length is None:
        length = 3 + random.randint(0, int(entropy / 2))

    # Add on some entropy to compensate for different expressions generating the
    # same apparent polynomial.
    entropy += combinatorics.log_number_binary_trees(length) / math.log(10)

    return _sample_with_brackets(0, variables, degrees, entropy, length, True)


def sample_with_small_evaluation(variable, degree, max_abs_input, entropy):
    """Generates a (canonically ordered) polynomial, with bounded evaluation.

    The coefficients are chosen to make use of the entropy, with the scaling
    adjusted so that all give roughly the same contribution to the output of the
    polynomial when the input is bounded in magnitude by `max_abs_input`.

    Args:
      variable: Variable to use in polynomial.
      degree: Degree of polynomial.
      max_abs_input: Number >= 1; max absolute value of input.
      entropy: Float; randomness for generating polynomial.

    Returns:
      Instance of `ops.Add`.
    """
    assert max_abs_input >= 1
    entropies = entropy * np.random.dirichlet(np.ones(degree + 1))
    coeffs = []

    for power in range(degree + 1):
        # This scaling guarantees that the terms give roughly equal contribution
        # to the typical magnitude of the polynomial when |input| <= max_abs_input.
        delta = 0.5 * (degree - 2 * power) * math.log10(max_abs_input)
        power_entropy = entropies[power] + delta
        min_abs = 1 if power == degree else 0
        coeff = number.integer(power_entropy, signed=True, min_abs=min_abs)
        coeffs.append(coeff)

    terms = [monomial(coeff, variable, power) for power, coeff in enumerate(coeffs)]
    return ops.Add(*terms)


def sample_messy_power(variable, entropy):
    """Returns unsimplified power expression like ((x**2)**3/x**4)**2/x**3."""
    if entropy <= 0:
        return variable

    which = random.choice([1, 2, 3])

    if which == 1:
        exponent_entropy = min(2, entropy)
        entropy -= exponent_entropy
        exponent = number.integer_or_rational(exponent_entropy, signed=True)
        left = sample_messy_power(variable, entropy)
        return ops.Pow(left, exponent)

    entropy_left = entropy / 2
    if entropy_left < 1:
        entropy_left = 0
    entropy_right = entropy - entropy_left
    if random.choice([False, True]):
        entropy_left, entropy_right = entropy_right, entropy_left

    left = sample_messy_power(variable, entropy_left)
    right = sample_messy_power(variable, entropy_right)
    if which == 2:
        return ops.Mul(left, right)
    else:
        return ops.Div(left, right)


def trim(coefficients):
    """Makes non-zero entry in the final slice along each axis."""
    coefficients = np.asarray(coefficients)
    non_zero = np.not_equal(coefficients, 0)
    ndim = coefficients.ndim

    for axis in range(ndim):
        length = coefficients.shape[axis]
        axis_complement = list(range(0, axis)) + list(range(axis + 1, ndim))
        non_zero_along_axis = np.any(non_zero, axis=tuple(axis_complement))

        slice_to = 0
        for index in range(length - 1, -1, -1):
            if non_zero_along_axis[index]:
                slice_to = index + 1
                break

        if slice_to < length:
            coefficients = coefficients.take(axis=axis, indices=list(range(slice_to)))

    return coefficients


def differentiate(coefficients, axis):
    """Differentiate coefficients (corresponding to polynomial) along axis."""
    coefficients = np.asarray(coefficients)
    indices = list(range(1, coefficients.shape[axis]))
    coefficients = coefficients.take(axis=axis, indices=indices)

    broadcast_shape = np.ones(coefficients.ndim, dtype=np.int32)
    broadcast_shape[axis] = len(indices)
    broadcast = np.asarray(indices).reshape(broadcast_shape)

    result = broadcast * coefficients
    return trim(result)


def integrate(coefficients, axis):
    """Integrate coefficients (corresponding to polynomial) along axis."""
    coefficients = np.asarray(coefficients)

    length = coefficients.shape[axis]
    broadcast_shape = np.ones(coefficients.ndim, dtype=np.int32)
    broadcast_shape[axis] = length
    powers = np.array([sympy.Integer(i) for i in range(1, length + 1)])
    powers = powers.reshape(broadcast_shape)

    result_unpadded = coefficients / powers

    pad = [(1 if i == axis else 0, 0) for i in range(coefficients.ndim)]
    return np.pad(result_unpadded, pad, "constant", constant_values=0)
