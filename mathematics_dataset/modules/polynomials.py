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

"""Polynomial manipulation (adding, composing, finding coefficients, etc)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import random

# Dependency imports
from mathematics_dataset import example
from mathematics_dataset.sample import number
from mathematics_dataset.sample import ops
from mathematics_dataset.sample import polynomials
from mathematics_dataset.util import composition
import numpy as np
from six.moves import range
import sympy


_ENTROPY_TRAIN = (3, 10)
_ENTROPY_INTERPOLATE = (8, 8)


def _make_modules(entropy):
    """Returns modules given "difficulty" parameters."""
    sample_args_pure = composition.PreSampleArgs(1, 1, *entropy)
    sample_args_composed = composition.PreSampleArgs(2, 4, *entropy)
    sample_args_mixed = composition.PreSampleArgs(1, 4, *entropy)

    return {
        "coefficient_named": functools.partial(
            coefficient_named, None, sample_args_pure
        ),
        "evaluate": functools.partial(evaluate, None, sample_args_pure),
        "evaluate_composed": functools.partial(evaluate, None, sample_args_composed),
        # TODO(b/124038948): consider doing pure sample args for 'add'?
        "add": functools.partial(add, None, sample_args_mixed),
        "expand": functools.partial(expand, None, sample_args_pure),
        "collect": functools.partial(collect, None, sample_args_pure),
        "compose": functools.partial(compose, None, sample_args_mixed),
        # Rearranging powers:
        "simplify_power": functools.partial(simplify_power, None, sample_args_pure),
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


def coefficient_named(value, sample_args, context=None):
    """E.g., "Express x^2 + 2x in the form h * x^2 + k * x + t and give h."."""
    del value  # not used
    if context is None:
        context = composition.Context()
    variable = sympy.Symbol(context.pop())

    entropy, sample_args = sample_args.peel()
    degree = random.randint(1, 4)
    if random.choice([False, True]):
        coefficients = polynomials.sample_coefficients(
            degree, entropy / 2, min_non_zero=random.randint(degree - 1, degree)
        )
        expanded = polynomials.expand_coefficients(coefficients, entropy / 2)
        expression = polynomials.coefficients_to_polynomial(expanded, variable)
    else:
        expression = polynomials.sample_with_brackets(variable, degree, entropy)
        coefficients = list(reversed(sympy.Poly(expression).all_coeffs()))

    named_coeffs = [sympy.Symbol(context.pop()) for _ in range(degree + 1)]
    canonical = polynomials.coefficients_to_polynomial(named_coeffs, variable)

    if random.random() < 0.2:  # only small probability of non-zero power
        power = random.randint(0, degree)
    else:
        non_zero_powers = [i for i in range(degree + 1) if coefficients[i] != 0]
        power = random.choice(non_zero_powers)

    value = coefficients[power]
    named_coeff = named_coeffs[power]

    template = random.choice(
        [
            "Express {expression} as {canonical} and give {target}.",
            "Rearrange {expression} to {canonical} and give {target}.",
            "Express {expression} in the form {canonical} and give {target}.",
            "Rearrange {expression} to the form {canonical} and give {target}.",
        ]
    )
    return example.Problem(
        question=example.question(
            context,
            template,
            expression=expression,
            canonical=canonical,
            target=named_coeff,
        ),
        answer=value,
    )


_TEMPLATES = [
    "What is {composed}?",
    "Calculate {composed}.",
    "Give {composed}.",
    "Determine {composed}.",
]


@composition.module(number.is_integer)
def evaluate(value, sample_args, context=None):
    """Entity for evaluating an integer-valued polynomial at a given point."""
    is_question = context is None
    if context is None:
        context = composition.Context()

    entropy, sample_args = sample_args.peel()

    if value is None:
        entropy_value = random.uniform(1, 1 + entropy / 3)
        entropy = max(0, entropy - entropy_value)
        value = number.integer(entropy_value, signed=True)

    entropy_input = random.uniform(1, 1 + entropy / 3)
    entropy = max(0, entropy - entropy_input)
    input_ = number.integer(entropy_input, signed=True)

    degree = random.randint(1, 3)

    entropies = entropy * np.random.dirichlet(list(range(1, degree + 1)))
    # Calculate coefficients in reverse order.
    target = value
    coeffs_reversed = []
    for i, coeff_entropy in enumerate(entropies):
        power = degree - i
        coeff = number.integer(coeff_entropy, signed=True)
        if input_ != 0:
            coeff += int(round(target / input_**power))
        if coeff == 0 and i == 0:
            # Don't allow zero in leading coefficient.
            coeff += random.choice([-1, 1])
        coeffs_reversed.append(coeff)
        target -= coeff * (input_**power)
    coeffs_reversed.append(target)

    coefficients = list(reversed(coeffs_reversed))

    (polynomial_entity, input_) = context.sample(
        sample_args, [composition.Polynomial(coefficients), input_]
    )
    composed = polynomial_entity.handle.apply(input_.handle)

    if is_question:
        template = random.choice(_TEMPLATES)
        return example.Problem(
            question=example.question(context, template, composed=composed),
            answer=value,
        )
    else:
        return composition.Entity(
            context=context,
            value=value,
            expression=composed,
            description="Let {self} be {composed}.",
            composed=composed,
        )


# TODO(b/124039290): merge with compose? both add and compose do similar things.
@composition.module(composition.is_integer_polynomial)
def add(value, sample_args, context=None):
    """E.g., "Let f(x)=2x+1, g(x)=3x+2. What is 5*f(x) - 7*g(x)?"."""
    is_question = context is None
    if context is None:
        context = composition.Context()

    entropy, sample_args = sample_args.peel()

    if value is None:
        max_degree = 3
        degree = random.randint(1, max_degree)
        entropy -= math.log10(max_degree)
        entropy_value = entropy / 2
        entropy -= entropy_value
        value = polynomials.sample_coefficients(
            degree, entropy=entropy_value, min_non_zero=random.randint(1, 3)
        )
        value = composition.Polynomial(value)

    c1, c2, coeffs1, coeffs2 = polynomials.coefficients_linear_split(
        value.coefficients, entropy
    )
    coeffs1 = polynomials.trim(coeffs1)
    coeffs2 = polynomials.trim(coeffs2)

    c1, c2, fn1, fn2 = context.sample(
        sample_args,
        [c1, c2, composition.Polynomial(coeffs1), composition.Polynomial(coeffs2)],
    )

    var = sympy.var(context.pop())

    expression = c1.handle * fn1.handle.apply(var) + c2.handle * fn2.handle.apply(var)

    if is_question:
        answer = polynomials.coefficients_to_polynomial(value.coefficients, var)
        answer = answer.sympy()
        template = random.choice(_TEMPLATES)
        return example.Problem(
            question=example.question(context, template, composed=expression),
            answer=answer,
        )
    else:
        intermediate_symbol = context.pop()
        intermediate = sympy.Function(intermediate_symbol)(var)
        return composition.Entity(
            context=context,
            value=value,
            description="Let {intermediate} = {composed}.",
            handle=composition.FunctionHandle(intermediate_symbol),
            intermediate=intermediate,
            composed=expression,
        )


def expand(value, sample_args, context=None):
    """E.g., "Expand (x**2 + 1)**2."."""
    del value  # not used
    if context is None:
        context = composition.Context()
    variable = sympy.Symbol(context.pop())
    entropy, sample_args = sample_args.peel()

    min_order = 1
    max_order = 5
    order = random.randint(min_order, max_order)
    entropy -= math.log10(max_order - min_order + 1)
    expression_ = polynomials.sample_with_brackets(variable, order, entropy)
    expanded = sympy.expand(expression_)
    template = random.choice(["Expand {expression}."])
    return example.Problem(
        question=example.question(context, template, expression=expression_),
        answer=expanded,
    )


@composition.module(composition.is_polynomial)
def collect(value, sample_args, context=None):
    """Collect terms in an unsimplified polynomial."""
    is_question = context is None
    if context is None:
        context = composition.Context()

    entropy, sample_args = sample_args.peel()
    if value is None:
        entropy_value, entropy = entropy * np.random.dirichlet([2, 3])
        degrees = [random.randint(1, 3)]
        value = composition.Polynomial(
            polynomials.sample_coefficients(degrees, entropy_value)
        )

    assert isinstance(value, composition.Polynomial)
    coefficients = value.coefficients

    all_coefficients_are_integer = True
    for coeff in coefficients.flat:
        if not number.is_integer(coeff):
            all_coefficients_are_integer = False
            break

    if all_coefficients_are_integer:
        coefficients = polynomials.expand_coefficients(coefficients, entropy)
    else:
        # put back the unused entropy
        sample_args = composition.SampleArgs(
            sample_args.num_modules, sample_args.entropy + entropy
        )

    num_variables = coefficients.ndim
    variables = [sympy.Symbol(context.pop()) for _ in range(num_variables)]
    unsimplified = polynomials.coefficients_to_polynomial(coefficients, variables)
    simplified = unsimplified.sympy().expand()

    # Bit of a hack: handle the very rare case where no number constants appearing
    if not ops.number_constants(unsimplified):
        unsimplified = ops.Add(unsimplified, ops.Constant(0))
    context.sample_by_replacing_constants(sample_args, unsimplified)

    if is_question:
        template = "Collect the terms in {unsimplified}."
        return example.Problem(
            question=example.question(context, template, unsimplified=unsimplified),
            answer=simplified,
        )
    else:
        function_symbol = context.pop()
        function = sympy.Function(function_symbol)(*variables)
        return composition.Entity(
            context=context,
            value=value,
            handle=composition.FunctionHandle(function_symbol),
            expression=unsimplified,
            polynomial_variables=variables,
            description="Let {function} = {unsimplified}.",
            function=function,
            unsimplified=unsimplified,
        )


def compose(value, sample_args, context=None):
    """E.g., "Let f(x)=2x+1, let g(x)=3x+10. What is f(g(x))?"."""
    del value  # unused
    if context is None:
        context = composition.Context()

    entropy, sample_args = sample_args.peel()
    entropy_f, entropy_g = entropy * np.random.dirichlet([1, 1])

    coeffs_f = polynomials.sample_coefficients([random.randint(1, 2)], entropy_f)
    coeffs_g = polynomials.sample_coefficients([random.randint(1, 2)], entropy_g)

    entity_f, entity_g = context.sample(
        sample_args,
        [composition.Polynomial(coeffs_f), composition.Polynomial(coeffs_g)],
    )

    variable = sympy.var(context.pop())

    poly_f = polynomials.coefficients_to_polynomial(coeffs_f, variable)
    poly_g = polynomials.coefficients_to_polynomial(coeffs_g, variable)

    poly_f_g = poly_f.sympy().subs(variable, poly_g.sympy()).expand()

    expression = composition.FunctionHandle(entity_f, entity_g).apply(variable)

    template = random.choice(_TEMPLATES)
    return example.Problem(
        question=example.question(context, template, composed=expression),
        answer=poly_f_g,
    )


def simplify_power(value, sample_args, context=None):
    """E.g., "Simplify ((x**2)**3/x**4)**2/x**3."."""
    del value  # unused
    if context is None:
        context = composition.Context()

    entropy, sample_args = sample_args.peel()

    variable = sympy.symbols(context.pop(), positive=True)
    unsimplified = polynomials.sample_messy_power(variable, entropy)
    answer = unsimplified.sympy()

    template = random.choice(
        [
            "Simplify {unsimplified} assuming {variable} is positive.",
        ]
    )
    return example.Problem(
        example.question(
            context, template, unsimplified=unsimplified, variable=variable
        ),
        answer,
    )
