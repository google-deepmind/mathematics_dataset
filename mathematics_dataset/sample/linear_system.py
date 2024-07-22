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

"""Generate linear systems with given set of solutions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# Dependency imports
from mathematics_dataset.sample import number
from mathematics_dataset.sample import ops
from mathematics_dataset.sample import polynomials
import numpy as np
from six.moves import range
import sympy


def _make_equals_zero_split(monomials):
    """Returns an `ops.Eq` containing sum of monomials split on left and right."""
    left = []
    right = []
    for monomial in monomials:
        if random.choice([False, True]):
            left.append(monomial)
        else:
            right.append(ops.Neg(monomial))
    if not left:
        left = [0]
    if not right:
        right = [0]
    left = ops.Add(*left)
    right = ops.Add(*right)
    return ops.Eq(left, right)


def _is_trivial_in(matrix, variable):
    """Returns true if matrix_ij == 0 for some i and all j != variable."""
    matrix = np.asarray(matrix)
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    size = matrix.shape[0]
    if size == 1:
        return False
    for i in range(size):
        all_zero = True
        for j in range(size):
            if j != variable and matrix[i, j] != 0:
                all_zero = False
                break
        if all_zero:
            return True
    return False


def _invertible_matrix(degree, entropy, non_trivial_in):
    """Generates random invertible matrix."""
    matrix_entropies = entropy * np.random.dirichlet(np.ones(degree * degree))
    matrix_entropies = np.reshape(matrix_entropies, [degree, degree])
    matrix_entropies = np.maximum(1, matrix_entropies)

    while True:

        def gen(i, j):
            return number.integer(matrix_entropies[i, j], True)

        matrix = [
            [gen(i, j) for i in range(degree)] for j in range(degree)
        ]  # pylint: disable=g-complex-comprehension
        if non_trivial_in is not None and _is_trivial_in(matrix, non_trivial_in):
            continue
        if sympy.det(sympy.Matrix(matrix)) != 0:
            break

    matrix = np.asarray(matrix).astype(int)
    return matrix


def linear_system(variables, solutions, entropy, non_trivial_in=None, length=None):
    """Returns a linear system (set of equalities) with the given solutions.

    Args:
      variables: List of variables.
      solutions: List of solutions, of the same length as `variables`.
      entropy: Float >= 0; the entropy used.
      non_trivial_in: Optional integer corresponding to a variable for which the
          solution shouldn't be "trivial". E.g., "solve a + b = 3, a = -2 for a"
          is disallowed if `variables[non_trivial_in] == 'a'`.
      length: Total number of terms appearing; if `None` then selected wisely.

    Returns:
      List of `ops.Eq`.
    """
    degree = len(variables)
    assert degree == len(solutions)

    frac_entropy_matrix = random.uniform(1 / 3, 2 / 3)
    matrix = _invertible_matrix(degree, entropy * frac_entropy_matrix, non_trivial_in)
    solutions = np.asarray(solutions)
    constant = np.matmul(matrix, solutions.astype(int))
    flattened = np.concatenate([np.reshape(matrix, [degree * degree]), constant])
    is_zero = flattened == 0

    if length is None:
        min_length = np.count_nonzero(flattened) + 1
        max_length = max(min_length, 1 + int(degree * (1 + entropy / 2)))
        length = random.randint(min_length, max_length)

    counts = polynomials.expanded_coefficient_counts(length=length, is_zero=is_zero)

    entropies = (
        (1 - frac_entropy_matrix)
        * entropy
        * np.random.dirichlet(np.maximum(1e-9, counts - 1))
    )

    terms = []
    for i in range(len(flattened)):
        coeffs = polynomials.integers_with_sum(
            value=flattened[i], count=counts[i], entropy=entropies[i]
        )
        terms.append(coeffs)

    matrix = terms[: degree * degree]
    constant = terms[-degree:]
    equations = []
    for row_index in range(degree):
        monomials = []
        for col_index in range(degree):
            for term in matrix[row_index * degree + col_index]:
                monomials.append(polynomials.monomial(term, variables[col_index], 1))
        for term in constant[row_index]:
            monomials.append(polynomials.monomial(-term, None, 0))
        equations.append(_make_equals_zero_split(monomials))

    return equations
