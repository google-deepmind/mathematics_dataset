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

"""Number-related questions, e.g., "write seventy-two as a number"."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import random

# Dependency imports
from mathematics_dataset import example
from mathematics_dataset.sample import number
from mathematics_dataset.util import composition
from mathematics_dataset.util import display
import numpy as np
import six
from six.moves import range
import sympy


_ENTROPY_TRAIN = (3, 10)
_ENTROPY_INTERPOLATE = (8, 8)
_ENTROPY_EXTRAPOLATE = (12, 12)


# Number of module compositions appearing in train/test, and extrapolation data.
_NUM_MODULES_COMPOSED = [2, 4]


def _make_modules(entropy, num_modules_composed):
    """Returns modules given "difficulty" parameters."""
    fns = {
        "gcd": gcd,
        "lcm": lcm,
        "div_remainder": div_remainder,
        "is_prime": is_prime,
        "is_factor": is_factor,
        "round_number": round_number,
        "place_value": place_value,
        "list_prime_factors": list_prime_factors,
    }

    # These modules don't have both pure and composed.
    modules = {
        "base_conversion": functools.partial(base_conversion, *entropy),
    }

    sample_args_pure = composition.PreSampleArgs(1, 1, *entropy)
    sample_args_composed = composition.PreSampleArgs(
        num_modules_composed[0], num_modules_composed[1], *entropy
    )

    for name, module in six.iteritems(fns):
        modules[name] = functools.partial(module, None, sample_args_pure)
        modules[name + "_composed"] = functools.partial(
            module, None, sample_args_composed
        )

    return modules


def train(entropy_fn):
    """Returns dict of training modules."""
    return _make_modules(
        entropy=entropy_fn(_ENTROPY_TRAIN), num_modules_composed=_NUM_MODULES_COMPOSED
    )


def test():
    """Returns dict of testing modules."""
    return _make_modules(
        entropy=_ENTROPY_INTERPOLATE, num_modules_composed=_NUM_MODULES_COMPOSED
    )


def test_extra():
    """Returns dict of extrapolation testing modules."""
    sample_args_pure = composition.PreSampleArgs(1, 1, *_ENTROPY_EXTRAPOLATE)
    return {
        "round_number_big": functools.partial(round_number, None, sample_args_pure),
        "place_value_big": functools.partial(place_value, None, sample_args_pure),
    }


def place_value(value, sample_args, context=None):
    """E.g., "Q: What is the tens digit of 31859? A: 5."""
    del value  # unused for now
    if context is None:
        context = composition.Context()

    entropy, sample_args = sample_args.peel()
    integer = number.integer(entropy, signed=False, min_abs=1)
    (entity,) = context.sample(sample_args, [integer])

    integer_as_string = str(integer)
    num_digits = len(integer_as_string)

    firsts = ["", "ten ", "hundred "]
    seconds = [
        "thousands",
        "millions",
        "billions",
        "trillions",
        "quadrillions",
        "quintillions",
        "sextillions",
        "septillions",
        "octillions",
        "nonillions",
        "decillions",
    ]
    place_names = ["units", "tens", "hundreds"]
    for second in seconds:
        for first in firsts:
            place_names.append(first + second)

    place = random.randint(1, num_digits)  # 1 = units, 2 = tens, etc.
    place_name = place_names[place - 1]
    answer = sympy.Integer(integer_as_string[num_digits - place])

    return example.Problem(
        question=example.question(
            context,
            "What is the {place_name} digit of {integer}?",
            place_name=place_name,
            integer=entity.expression_else_handle,
        ),
        answer=answer,
    )


# TODO(b/124040078): add to composition system?
def round_number(value, sample_args, context=None):
    """Question for rounding integers and decimals."""
    del value  # unused for now
    if context is None:
        context = composition.Context()

    entropy, sample_args = sample_args.peel()

    # This is the power of 10 to round to. E.g., power == 0 corresponds to
    # rounding to the nearest integer; power == -2 corresponds to rounding to two
    # decimal places, and power == 3 corresponds to rounding to the nearest 1000.
    power = random.randint(-7, 6)

    answer_entropy = 1 + random.uniform(0, entropy / 2)
    entropy = max(1, entropy - answer_entropy)
    value_integer = number.integer(answer_entropy, signed=True)

    remainder_divisor = 10 ** int(math.ceil(entropy))
    remainder_range_lower = -remainder_divisor / 2
    remainder_range_upper = remainder_divisor / 2

    if value_integer <= 0:
        remainder_range_lower += 1
    if value_integer >= 0:
        remainder_range_upper -= 1

    remainder = random.randint(remainder_range_lower, remainder_range_upper)
    input_ = value_integer + sympy.Rational(remainder, remainder_divisor)
    scale = 10**power if power >= 0 else sympy.Rational(1, 10 ** (-power))
    input_ = input_ * scale
    value = value_integer * scale
    if not number.is_integer(input_):
        input_ = display.Decimal(input_)
    if not number.is_integer(value):
        value = display.Decimal(value)

    (input_,) = context.sample(sample_args, [input_])

    if power > 0:
        # Rounding to a power of ten.
        round_to = 10**power
        if random.choice([False, True]):
            # Write the rounding value as a word instead.
            round_to = display.StringNumber(
                round_to, join_number_words_with_hyphens=False
            )
        description = "the nearest {round_to}".format(round_to=round_to)
    elif power == 0 and random.choice([False, True]):
        # Round to nearest integer.
        description = "the nearest integer"
    else:
        # Round to decimal places.
        description = random.choice(["{dps} decimal place", "{dps} dp"])
        if power != -1:
            # Plural
            description += "s"
        dps = -power
        if random.choice([False, True]):
            dps = display.StringNumber(dps)
        description = description.format(dps=dps)

    template = random.choice(
        [
            "Round {input} to {description}.",
            "What is {input} rounded to {description}?",
        ]
    )

    return example.Problem(
        question=example.question(
            context, template, input=input_, description=description
        ),
        answer=value,
    )


def _semi_prime(entropy):
    """Generates a semi-prime with the given entropy."""
    # Add on extra entropy to account for the sparsity of the primes; we don't
    # actually use the integers sampled, but rather a random prime close to them;
    # thus some entropy is lost, which we must account for
    entropy += math.log10(max(1, entropy * math.log(10)))

    # We intentionally uniformy sample the "entropy" (i.e., approx number digits)
    # of the two factors.
    entropy_1, entropy_2 = entropy * np.random.dirichlet([1, 1])

    # Need >= 2 for randprime to always work (Betrand's postulate).
    approx_1 = number.integer(entropy_1, signed=False, min_abs=2)
    approx_2 = number.integer(entropy_2, signed=False, min_abs=2)

    factor_1 = sympy.ntheory.generate.randprime(approx_1 / 2, approx_1 * 2)
    factor_2 = sympy.ntheory.generate.randprime(approx_2 / 2, approx_2 * 2)

    return factor_1 * factor_2


def is_prime(value, sample_args, context=None):
    """Questions asking about primality."""
    del value  # unused for now
    if context is None:
        context = composition.Context()

    entropy, sample_args = sample_args.peel()
    composite = _semi_prime(entropy)

    if random.choice([False, True]):
        # Use the composite
        integer = composite
        is_prime_ = False
    else:
        # Take the next prime after the composite, to ensure the same distribution
        # as composites. Do "composite - 4" so we occasionally see "2" as a prime.
        integer = sympy.ntheory.generate.nextprime(composite - 4)
        is_prime_ = True

    (integer_entity,) = context.sample(sample_args, [integer])

    if random.choice([False, True]) and integer != 1:
        answer = not is_prime_
        attribute_name = random.choice(["composite", "a composite number"])
    else:
        answer = is_prime_
        attribute_name = random.choice(["prime", "a prime number"])

    return example.Problem(
        question=example.question(
            context,
            "Is {integer} {attribute}?",
            integer=integer_entity.expression_else_handle,
            attribute=attribute_name,
        ),
        answer=answer,
    )


def is_factor(value, sample_args, context=None):
    """E.g., "Is 5 a factor of 48?"."""
    del value  # unused
    if context is None:
        context = composition.Context()

    entropy, sample_args = sample_args.peel()

    entropy_factor = 1 + random.uniform(0, entropy / 3)
    entropy = max(0, entropy - entropy_factor)
    maybe_factor = number.integer(entropy_factor, False, min_abs=2)

    integer = maybe_factor * number.integer(entropy, False, min_abs=1)
    # Produce balanced classes.
    if random.choice([False, True]):
        # The following makes it not a factor.
        integer += random.randint(1, maybe_factor - 1)

    (entity,) = context.sample(sample_args, [integer])

    templates = [
        "Is {maybe_factor} a factor of {value}?",
        "Is {value} a multiple of {maybe_factor}?",
        "Does {maybe_factor} divide {value}?",
    ]
    if maybe_factor == 2:
        templates += [
            "Is {value} even?",
        ]
    template = random.choice(templates)

    answer = integer % maybe_factor == 0
    return example.Problem(
        question=example.question(
            context,
            template,
            maybe_factor=maybe_factor,
            value=entity.expression_else_handle,
        ),
        answer=answer,
    )


def list_prime_factors(value, sample_args, context=None):
    """E.g., "What are the prime factors of 36?"."""
    del value  # unused for now
    if context is None:
        context = composition.Context()

    entropy, sample_args = sample_args.peel()
    entropy = max(1, entropy)

    integer = number.integer(entropy, signed=False, min_abs=2)

    (entity,) = context.sample(sample_args, [integer])
    prime_factors = sorted(sympy.factorint(integer).keys())
    template = random.choice(
        [
            "What are the prime factors of {integer}?",
            "List the prime factors of {integer}.",
        ]
    )
    return example.Problem(
        question=example.question(
            context, template, integer=entity.expression_else_handle
        ),
        answer=display.NumberList(prime_factors),
    )


def _pair_with_large_hidden_factor(entropy):
    """Returns pair of numbers with possibly large common factor hidden."""
    entropy_p, entropy_q, _ = entropy * np.random.dirichlet([1, 1, 1])
    # Min entropy on p and q to minimize trivial solutions.
    entropy_p = max(1, entropy_p)
    entropy_q = max(1, entropy_q)
    entropy_mult = max(0, entropy - entropy_p - entropy_q)

    p = number.integer(entropy_p, False, min_abs=1)
    q = number.integer(entropy_q, False, min_abs=1)
    mult = number.integer(entropy_mult, False, min_abs=1)
    p *= mult
    q *= mult
    return p, q


def lcm(value, sample_args, context=None):
    """Question for least common multiple of p and q."""
    del value  # unused
    if context is None:
        context = composition.Context()

    entropy, sample_args = sample_args.peel()

    p, q = _pair_with_large_hidden_factor(entropy)
    answer = sympy.lcm(p, q)

    if random.choice([False, True]):
        p, q = context.sample(sample_args, [p, q])
        # Ask the question directly.
        adjective = random.choice(["least", "lowest", "smallest"])
        template = random.choice(
            [
                "Calculate the {adjective} common multiple of {p} and {q}.",
                "What is the {adjective} common multiple of {p} and {q}?",
            ]
        )
        return example.Problem(
            question=example.question(
                context,
                template,
                adjective=adjective,
                p=p.expression_else_handle,
                q=q.expression_else_handle,
            ),
            answer=answer,
        )
    else:
        # Phrase the question as finding the common denominator of two fractions.
        p = number.integer(2, signed=True, coprime_to=p) / p
        q = number.integer(2, signed=True, coprime_to=q) / q
        p, q = context.sample(sample_args, [p, q])

        template = random.choice(
            [
                "What is the common denominator of {p} and {q}?",
                "Find the common denominator of {p} and {q}.",
                "Calculate the common denominator of {p} and {q}.",
            ]
        )
        return example.Problem(
            question=example.question(
                context,
                template,
                p=p.expression_else_handle,
                q=q.expression_else_handle,
            ),
            answer=answer,
        )


def _random_coprime_pair(entropy):
    """Returns a pair of random coprime integers."""
    coprime_product = number.integer(entropy, False, min_abs=1)
    factors = sympy.factorint(coprime_product)

    def take():
        prime = random.choice(list(factors.keys()))
        power = factors[prime]
        del factors[prime]
        return prime**power

    if random.random() < 0.8 and len(factors) >= 2:
        # Disallow trivial factoring where possible.
        count_left = random.randint(1, len(factors) - 1)
        count_right = len(factors) - count_left
    else:
        count_left = random.randint(0, len(factors))
        count_right = len(factors) - count_left

    left = sympy.prod([take() for _ in range(count_left)])
    right = sympy.prod([take() for _ in range(count_right)])
    assert left * right == coprime_product
    return left, right


# @composition.module(number.is_positive_integer)
def gcd(value, sample_args, context=None):
    """Question for greatest common divisor of p and q."""
    is_question = context is None
    if context is None:
        context = composition.Context()

    entropy, sample_args = sample_args.peel()
    if value is None:
        value_entropy = 1 + random.uniform(0, entropy / 3)
        entropy = max(1, entropy - value_entropy)
        value = number.integer(value_entropy, False, min_abs=1)

    p_mult, q_mult = _random_coprime_pair(entropy)

    p = value * p_mult
    q = value * q_mult
    assert sympy.gcd(p, q) == value

    p, q = context.sample(sample_args, [p, q])

    adjective = (
        random.choice(["greatest", "highest"])
        + " common "
        + random.choice(["divisor", "factor"])
    )

    if is_question:
        template = random.choice(
            [
                "Calculate the {adjective} of {p} and {q}.",
                "What is the {adjective} of {p} and {q}?",
            ]
        )
        return example.Problem(
            question=example.question(context, template, adjective=adjective, p=p, q=q),
            answer=value,
        )
    else:
        return composition.Entity(
            context=context,
            value=value,
            description="Let {self} be the {adjective} of {p} and {q}.",
            adjective=adjective,
            p=p,
            q=q,
        )


# @composition.module(number.is_positive_integer)
def div_remainder(value, sample_args, context=None):
    """E.g., "What is the remainder when 27 is divided by 5?"."""
    is_question = context is None
    if context is None:
        context = composition.Context()

    entropy, sample_args = sample_args.peel()

    if value is None:
        entropy_value = 1 + random.uniform(0, entropy / 3)
        entropy = max(0, entropy - entropy_value)
        value = number.integer(entropy_value, signed=False)

    entropy_a, entropy_q = entropy * np.random.dirichlet([1, 1])
    a = number.integer(entropy_a, signed=False, min_abs=1)
    q = value + number.integer(entropy_q, signed=False, min_abs=1)

    p = a * q + value
    assert p % q == value
    p, q = context.sample(sample_args, [p, q])

    if is_question:
        template = random.choice(
            [
                "Calculate the remainder when {p} is divided by {q}.",
                "What is the remainder when {p} is divided by {q}?",
            ]
        )
        return example.Problem(
            question=example.question(
                context,
                template,
                p=p.expression_else_handle,
                q=q.expression_else_handle,
            ),
            answer=value,
        )
    else:
        return composition.Entity(
            context=context,
            value=value,
            description="Let {self} be the remainder when {p} is divided by {q}.",
            p=p,
            q=q,
        )


def base_conversion(min_entropy, max_entropy):
    """E.g., "What is 17 base 8 in base 10?"."""
    context = composition.Context()

    from_base = random.randint(2, 16)
    while True:
        to_base = random.randint(2, 16)
        if to_base != from_base:
            break

    # Entropy used up in selecting bases.
    entropy_used = math.log10(16 * 15)
    entropy = random.uniform(min_entropy - entropy_used, max_entropy - entropy_used)

    value = number.integer(entropy, signed=True)
    template = random.choice(
        [
            "{from_str} (base {from_base}) to base {to_base}",
            "Convert {from_str} (base {from_base}) to base {to_base}.",
            "What is {from_str} (base {from_base}) in base {to_base}?",
        ]
    )
    return example.Problem(
        question=example.question(
            context,
            template,
            from_str=display.NumberInBase(value, from_base),
            from_base=from_base,
            to_base=to_base,
        ),
        answer=display.NumberInBase(value, to_base),
    )
