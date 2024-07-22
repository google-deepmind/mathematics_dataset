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

"""Calculus related questions, e.g., "differentiate x**2"."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import random

# Dependency imports
from mathematics_dataset import example
from mathematics_dataset.sample import polynomials
from mathematics_dataset.util import composition
from mathematics_dataset.util import display
import numpy as np
from six.moves import range
import sympy


_ENTROPY_TRAIN = (3, 10)
_ENTROPY_INTERPOLATE = (8, 8)


def _make_modules(entropy):
    """Returns modules given "difficulty" parameters."""
    sample_args_pure = composition.PreSampleArgs(1, 1, *entropy)
    sample_args_composed = composition.PreSampleArgs(2, 4, *entropy)

    return {
        "differentiate_composed": functools.partial(
            differentiate_univariate, None, sample_args_composed
        ),
        "differentiate": functools.partial(differentiate, None, sample_args_pure),
    }


def train(entropy_fn):
    """Returns dict of training modules."""
    return _make_modules(entropy_fn(_ENTROPY_TRAIN))


def test():
    """Returns dict of testing modules."""
    return _make_modules(_ENTROPY_INTERPOLATE)


def test_extra():
    """Returns dict of extrapolation testing modules."""
    return {}


def _generate_polynomial(num_variables, entropy, derivative_order, derivative_axis):
    """Returns polynomial."""
    # Note: numpy randint has upper bound as ) not ], unlike python random.randint
    degrees = np.random.randint(1, 4, [num_variables])
    degrees[derivative_axis] = np.random.randint(0, 4)  # allow to be zero here.

    coefficients = polynomials.sample_coefficients(degrees, entropy)

    # We also generate coefficients that will disappear when differentiated.
    # Thus we don't account for the entropy used here.
    assert derivative_order > 0
    degrees[derivative_axis] = derivative_order - 1
    extra_coefficients = polynomials.sample_coefficients(degrees, entropy)

    return np.concatenate([extra_coefficients, coefficients], axis=derivative_axis)


def _template(module_count, derivative_order, num_variables):
    """Selects appropriate template."""
    templates = [
        "Найдите {nth}-ую производную от {eq} по переменной {var}.",
        "Какова {nth}-ая производная функции {eq} по переменной {var}?",
        "Определите {nth}-ую производную {eq} по {var}.",
        "Вычислите {nth}-ую производную выражения {eq} относительно {var}.",
    ]
    if derivative_order == 1:
        templates += [
            "Продифференцируйте {eq} по переменной {var}.",
            "Найдите производную {eq} по {var}.",
            "Какова производная {eq} по {var}?",
            "Вычислите дифференциал функции {eq} относительно {var}.",
        ]

    derivative_variable_is_unambiguous = num_variables == 1 and module_count == 1
    if derivative_variable_is_unambiguous:
        templates += [
            "Найдите {nth}-ую производную {eq}.",
            "Какова {nth}-ая производная {eq}?",
            "Определите {nth}-ую производную выражения {eq}.",
            "Вычислите {nth}-ую производную функции {eq}.",
        ]
        if derivative_order == 1:
            templates += [
                "Продифференцируйте {eq}.",
                "Найдите производную {eq}.",
                "Какова производная {eq}?",
                "Вычислите дифференциал функции {eq}.",
            ]

    return random.choice(templates)


def _sample_integrand(coefficients, derivative_order, derivative_axis, entropy):
    """Integrates `coefficients` and adds sampled "constant" terms."""
    coefficients = np.asarray(coefficients)

    # Integrate (with zero for constant terms).
    integrand = coefficients
    for _ in range(derivative_order):
        integrand = polynomials.integrate(integrand, derivative_axis)

    # Add on sampled constant terms.
    constant_degrees = np.array(integrand.shape) - 1
    constant_degrees[derivative_axis] = derivative_order - 1
    extra_coeffs = polynomials.sample_coefficients(constant_degrees, entropy)
    pad_amount = coefficients.shape[derivative_axis]
    pad = [
        (0, pad_amount if i == derivative_axis else 0) for i in range(coefficients.ndim)
    ]
    extra_coeffs = np.pad(extra_coeffs, pad, "constant", constant_values=0)
    return integrand + extra_coeffs


def _differentiate_polynomial(value, sample_args, context, num_variables):
    """Generates a question for differentiating a polynomial."""
    is_question = context is None
    if context is None:
        context = composition.Context()

    if value is not None:
        num_variables = value.coefficients.ndim

    entropy, sample_args = sample_args.peel()
    max_derivative_order = 3
    derivative_order = random.randint(1, max_derivative_order)
    entropy = max(0, entropy - math.log10(max_derivative_order))

    derivative_axis = random.randint(0, num_variables - 1)
    if value is None:
        coefficients = _generate_polynomial(
            num_variables, entropy, derivative_order, derivative_axis
        )
    else:
        coefficients = _sample_integrand(
            value.coefficients, derivative_order, derivative_axis, entropy
        )

    (entity,) = context.sample(sample_args, [composition.Polynomial(coefficients)])

    value = coefficients
    for _ in range(derivative_order):
        value = polynomials.differentiate(value, axis=derivative_axis)
    nth = display.StringOrdinal(derivative_order)

    if entity.has_expression():
        polynomial = entity.expression
        variables = entity.polynomial_variables
    else:
        variables = [sympy.Symbol(context.pop()) for _ in range(num_variables)]
        polynomial = entity.handle.apply(*variables)
    variable = variables[derivative_axis]

    if is_question:
        template = _template(context.module_count, derivative_order, len(variables))
        answer = polynomials.coefficients_to_polynomial(value, variables).sympy()
        return example.Problem(
            question=example.question(
                context, template, eq=polynomial, var=variable, nth=nth
            ),
            answer=answer,
        )
    else:
        fn_symbol = context.pop()
        variables_string = ", ".join(str(variable) for variable in variables)
        assert len(variables) == 1  # since below we don't specify var we diff wrt
        return composition.Entity(
            context=context,
            value=composition.Polynomial(value),
            description="Пусть {fn}({variables}) будет {nth}-ой производной от {eq}.",
            handle=composition.FunctionHandle(fn_symbol),
            fn=fn_symbol,
            variables=variables_string,
            nth=nth,
            eq=polynomial,
        )


def differentiate_univariate(value, sample_args, context=None):
    return _differentiate_polynomial(value, sample_args, context, 1)


@composition.module(composition.is_polynomial)
def differentiate(value, sample_args, context=None):
    num_variables = random.randint(1, 4)
    return _differentiate_polynomial(value, sample_args, context, num_variables)
