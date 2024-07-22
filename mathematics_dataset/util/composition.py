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

"""For performing module composition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import string

# Dependency imports
from mathematics_dataset.sample import number
from mathematics_dataset.sample import ops
from mathematics_dataset.sample import polynomials
from mathematics_dataset.util import combinatorics
from mathematics_dataset.util import display
import numpy as np
import six
from six.moves import range
from six.moves import zip
import sympy


# Allowed symbols. Don't allow "e", as sympy can hang when printing it as a
# function symbol (and it's reserved for exponent).
_ALLOWED_SYMBOLS = set(string.ascii_lowercase).difference(set(["e"]))


class Polynomial(collections.namedtuple("Polynomial", ("coefficients"))):
    """Value wrapper for a polynomial function.

    Attributes:
      coefficients: Numpy array of coefficients; see `polynomials.py`.
    """

    def __new__(cls, coefficients):
        coefficients = np.asarray(coefficients)
        return super(Polynomial, cls).__new__(cls, coefficients)


def is_polynomial(value):
    return isinstance(value, Polynomial)


def is_integer_polynomial(value):
    if not is_polynomial(value):
        return False
    coefficients = np.reshape(value.coefficients, [-1])
    return all(number.is_integer(coeff) for coeff in coefficients)


# List of pairs of `(filter, sampler)`, where `filter` is a function returning
# True if the sampler is valid for the given value, and `sampler` returns an
# `Entity`.
_FILTERS_AND_SAMPLERS = []


def module(filter_):
    """Returns a Decorator for a module function.

    The returned decorator adds the function to the library of known modules.

    Args:
      filter_: Callable determining whether the module can handle a given value.

    Returns:
      Decorator that adds the module function to the library.
    """

    def decorator(module_fn):
        """Decorates a module function."""
        _FILTERS_AND_SAMPLERS.append((filter_, module_fn))
        return module_fn

    return decorator


class SampleArgs(collections.namedtuple("SampleArgs", ("num_modules", "entropy"))):
    """For sampling mathematical entities / questions."""

    def peel(self, frac=1):
        """Peels one (or `frac`) of a module's entropy.

        In addition to a portion of the entropy, this returns a new `SampleArgs`
        (since this object is immutable), which you should use when creating child
        modules.

        Args:
          frac: Float; proportion of module's entropy to take.

        Returns:
          Triple `(entropy, new_sample_args)`, where `new_sample_args` is a new
          `SampleArgs` with the entropy removed.
        """
        entropy = frac * self.entropy / self.num_modules
        new_sample_args = SampleArgs(
            num_modules=self.num_modules, entropy=self.entropy - entropy
        )
        return entropy, new_sample_args

    def split(self, count):
        """Splits the entropy and module counts up.

        Args:
          count: Integer >= 1; the split size.

        Returns:
          List of `SampleArgs` of length `count`, to be passed to create child
          entities.

        Raises:
          ValueError: If it was not possible to use up all the entropy, for example,
              all requested types were `WithValue`.
        """
        num_child_modules = self.num_modules - 1

        # Sample module counts at random
        module_counts = combinatorics.uniform_non_negative_integers_with_sum(
            count, num_child_modules
        )

        if num_child_modules == 0:
            if self.entropy > 0:
                raise ValueError("Unused entropy")
            entropies = np.zeros(count)
        else:
            entropies = self.entropy * np.random.dirichlet(
                np.maximum(1e-9, module_counts)
            )

        sample_args = []
        for i, num_modules in enumerate(module_counts):
            child_sample_args = SampleArgs(
                num_modules=num_modules, entropy=entropies[i]
            )
            sample_args.append(child_sample_args)

        return sample_args


class PreSampleArgs(
    collections.namedtuple(
        "PreSampleArgs", ("min_modules", "max_modules", "min_entropy", "max_entropy")
    )
):
    """Sample args before module count and entropy have been sampled."""

    def __call__(self):
        """Samples `SampleArgs`."""
        return SampleArgs(
            num_modules=random.randint(self.min_modules, self.max_modules),
            entropy=random.uniform(self.min_entropy, self.max_entropy),
        )

    def peel(self, *args, **kwargs):
        sample_args = self()
        return sample_args.peel(*args, **kwargs)

    def split(self, *args, **kwargs):
        sample_args = self()
        return sample_args.split(*args, **kwargs)


class FunctionHandle(object):
    """Special handle to allow function composition.

    For example, suppose fn1 = f o g
                         fn2 = h

    and we want to display fn1(fn2(x)) = f(g(h(x))). This function basically just
    stores the list of sympy functions.
    """

    def __init__(self, *function_entities):
        """Initialize a `FunctionHandle`.

        Args:
          *function_entities: List of function letters and `Entity`s representing
              functions, to be composed.
        """
        self._functions = []
        for fn in function_entities:
            if isinstance(fn, str):
                functions = [sympy.Function(fn)]
            else:
                assert isinstance(fn, Entity)
                assert isinstance(fn.handle, FunctionHandle)
                functions = fn.handle.functions
            self._functions += functions

    def apply(self, *input_):
        """Returns f(g(...(input)...)) where f, g, ... are the functions."""
        result = None
        for function in reversed(self._functions):
            if result is None:
                result = function(*input_)
            else:
                result = function(result)
        return result

    @property
    def functions(self):
        return self._functions

    def __str__(self):
        raise ValueError("This should not be directly converted to a string")


def _polynomial_entity(value, context):
    """Create a generic `Entity` describing a polynomial."""
    assert isinstance(value, Polynomial)
    coefficients = np.asarray(value.coefficients)
    num_variables = coefficients.ndim
    variables = [sympy.Symbol(context.pop()) for _ in range(num_variables)]
    function_symbol = context.pop()
    handle = FunctionHandle(function_symbol)
    handle_description = sympy.Function(function_symbol)(*variables)

    polynomial = polynomials.coefficients_to_polynomial(coefficients, variables)
    polynomial = polynomial.sympy()

    return Entity(
        context=context,
        value=value,
        expression=polynomial,
        polynomial_variables=variables,
        description="Let {function} = {polynomial}.",
        handle=handle,
        function=handle_description,
        polynomial=polynomial,
    )


class Context(object):
    """Keeps track of used symbols, and sampling of children.

    Each context is associated with an entity. Entities are constructed in a
    tree-like fashion.
    """

    def __init__(self, relation_symbols=None):
        """Initializes a `Context`.

        Args:
          relation_symbols: Set of symbols used "externally": this means (with
              reference to the tree in which this context sits) all symbols
              occurring that aren't in this node or sub-nodes.
        """
        if relation_symbols is None:
            relation_symbols = set()
        else:
            assert isinstance(relation_symbols, set)
        for symbol in relation_symbols:
            assert isinstance(symbol, str)

        self._relation_symbols = relation_symbols
        self._self_symbols = set()
        self._child_symbols = set()
        self._module_count = 1
        self._child_entities = []

    @property
    def relation_symbols(self):
        return self._relation_symbols.copy()

    @property
    def self_symbols(self):
        return self._self_symbols.copy()

    @property
    def child_symbols(self):
        return self._child_symbols.copy()

    @property
    def child_entities(self):
        return self._child_entities[:]

    def pop(self):
        """Returns an unused symbol (and keeps track of it being used)."""
        allowed = (
            _ALLOWED_SYMBOLS.difference(self._relation_symbols)
            .difference(self._self_symbols)
            .difference(self._child_symbols)
        )
        if not allowed:
            raise ValueError("Ran out of symbols")
        symbol = random.choice(list(allowed))
        self._self_symbols.add(symbol)
        return symbol

    def mark_used(self, symbol):
        """Marks a given symbol as used."""
        assert isinstance(symbol, str)
        if (
            symbol in self._relation_symbols
            or symbol in self._self_symbols
            or symbol in self._child_symbols
        ):
            raise ValueError("Symbol {} already used".format(symbol))
        self._self_symbols.add(symbol)

    @property
    def module_count(self):
        """Returns the number of modules sampled."""
        return self._module_count

    def _sampler(self, value, sample_args):
        """Returns a sampler appropriate for the given args.

        Args:
          value: Target value.
          sample_args: Instance of `SampleArgs` controlling entropy etc.

        Returns:
          A sampler for producing entities.

        Raises:
          ValueError: If no valid samplers were found.
        """
        valid = []
        for filter_, sampler in _FILTERS_AND_SAMPLERS:
            if filter_(value):
                valid.append(sampler)
        if not valid:
            raise ValueError(
                "No valid samplers found: value={} sample_args={}".format(
                    value, sample_args
                )
            )
        return random.choice(valid)

    def _value_entity(self, value, context):
        if isinstance(value, (sympy.Integer, sympy.Rational, display.Decimal)):
            return Entity(context=context, value=value, handle=value)
        if isinstance(value, Polynomial):
            return _polynomial_entity(value, context)
        raise ValueError(
            "Don't know how to handle value={} of type {}".format(value, type(value))
        )

    def sample(self, sample_args, values):
        """Sample multiple entities.

        Args:
          sample_args: Instance of `SampleArgs`. The min and max entropy
              and module count will be split up betwene the various entities
              sampled.
          values: List of values to sample.

        Returns:
          List of `Entity` of the same length as `types`.

        Raises:
          RuntimeError: If one of the modules generates a non-`Entity`.
        """
        # Can only sample children once.
        assert self._module_count == 1
        assert not self._child_symbols
        assert not self._child_entities

        if isinstance(sample_args, PreSampleArgs):
            sample_args = sample_args()
        sample_args_split = sample_args.split(len(values))

        def all_symbols():
            return self._relation_symbols.union(self._self_symbols).union(
                self._child_symbols
            )

        for value, child_sample_args in zip(values, sample_args_split):
            if number.is_integer(value):
                value = sympy.Integer(value)

            all_symbols_ = all_symbols()
            context = Context(all_symbols_)

            if child_sample_args.num_modules == 0:
                entity = self._value_entity(value, context)
            else:
                sampler = self._sampler(value, child_sample_args)
                entity = sampler(value, child_sample_args, context)
                if not isinstance(entity, Entity):
                    raise RuntimeError(
                        "Expected entity, but got {} instead".format(entity)
                    )
                if not number.is_integer_or_rational_or_decimal(
                    entity.value
                ) and not isinstance(entity.value, Polynomial):
                    raise RuntimeError(
                        "sampler {} returned invalid value of type {}".format(
                            sampler, type(entity.value)
                        )
                    )
                if (
                    number.is_integer_or_rational_or_decimal(value)
                    and entity.value != value
                ) or (
                    isinstance(value, Polynomial)
                    and not np.array_equal(
                        entity.value.coefficients, value.coefficients
                    )
                ):
                    raise RuntimeError(
                        "entity values differ, sampler={} wanted={} got={}".format(
                            sampler, value, entity.value
                        )
                    )
                if child_sample_args.num_modules != context.module_count:
                    raise RuntimeError(
                        "unused modules, value={} sample_args={} context.module_count={},"
                        " sampler={}".format(
                            value, child_sample_args, context.module_count, sampler
                        )
                    )
                self._module_count += context.module_count

            self._child_entities.append(entity)
            for symbol in context.self_symbols.union(context.child_symbols):
                assert symbol not in all_symbols_
                self._child_symbols.add(symbol)

        return self._child_entities

    def sample_by_replacing_constants(self, sample_args, expressions):
        """Replaces some of the constants with handles from other modules."""
        max_children = sample_args.num_modules - 1
        if max_children <= 0:
            return
        if isinstance(expressions, ops.Op):
            expressions = [expressions]
        constants = ops.number_constants(expressions)
        if not constants:
            raise ValueError(
                "No constants to replace in {}".format(
                    [str(expr) for expr in expressions]
                )
            )

        sample_count = random.randint(1, min(max_children, len(constants)))
        constants = random.sample(constants, sample_count)

        values = [constant.value for constant in constants]
        entities = self.sample(sample_args, values)
        for constant, entity in zip(constants, entities):
            constant.value = entity.handle


def expand_entities(context, **kwargs):
    """Returns prefix description and updated `**kwargs`.

    Args:
      context: Instance of `Context` containing all `Entity`s that occur.
      **kwargs: Dictionary of key/value pairs, some of which are `Entity`s.

    Returns:
      Pair `(child_description, new_kwargs)`. `child_description` is a description
      of the entities contained in `kwargs`, and `new_kwargs` contains handles.
    """
    kwargs = kwargs.copy()
    entities = set(context.child_entities)
    for key, maybe_entity in six.iteritems(kwargs):
        if isinstance(maybe_entity, Entity):
            entities.add(maybe_entity)
            kwargs[key] = maybe_entity.handle
    entities = list(entities)
    random.shuffle(entities)

    child_descriptions = []
    for entity in entities:
        child_descriptions.append(entity.child_description)
        if not entity.expression_used:
            child_descriptions.append(entity.description)

    child_description = " ".join([s for s in child_descriptions if s])
    return child_description, kwargs


class Entity(object):
    """An entity (e.g., representing an integer or function).

    Example usage:

    ```
    new_entity = Entity(
        context=context,
        value=17,
        description='Let {self} be the gcd of {q} and {q}.'
        p=p, q=q)
    ```

    Above, {self} will be replaced with a new symbol, to be used as the entity
    handle.
    """

    def __init__(
        self,
        context,
        value,
        description="",
        handle=None,
        expression=None,
        polynomial_variables=None,
        **description_kwargs
    ):
        """Initializes an entity.

        Args:
          context: Instance of `Context` keeping track of used symbols and child
              entities (that may not explicitly occur in `description_kwargs`).
          value: The value represented by this Entity.
          description: String, describing this Entity. This can contain '{self}' as
              a substring, in which case it will be substituted with a new handle
              (and `context` must be non-None).
          handle: Optional string/symbol that refers to this Entity. This must
              always be provided if '{self}' does not occur in `description`.
          expression: Optional string (or something that can be converted to a
              string, like ops.Op or sympy.expr) representing the value of this
              Entity, that can be used directly instead of using `handle`.
          polynomial_variables: If `expression` provided, then for polynomial
              entities, the variables used.
          **description_kwargs: Dict of substitutions (including entities) for use
              in `description`.

        Raises:
          ValueError: If handle specified and '{self}' in description; or neither.
        """
        self._value = value

        child_description, description_kwargs = expand_entities(
            context, **description_kwargs
        )

        if "{self}" in description:
            if handle is not None:
                raise ValueError("Cannot specify handle if {self} in description")
            handle = context.pop()
            description_kwargs["self"] = handle
            handle = sympy.var(handle)
        else:
            if handle is None:
                raise ValueError("Must specify handle if {self} not in description")
            if isinstance(handle, str):
                handle = sympy.var(handle)

        if (
            isinstance(value, Polynomial)
            and expression is not None
            and polynomial_variables is None
        ):
            raise ValueError("Must provided polynomial_variables")

        self._child_description = child_description
        self._description = description.format(**description_kwargs)
        self._handle = handle
        self._expression = expression
        self._polynomial_variables = polynomial_variables

        # For checking that we don't use both the handle and the expression form of
        # this entity.
        self._handle_used = False
        self._expression_used = False

    @property
    def value(self):
        """The actual value of the entity (e.g., sympy object)."""
        return self._value

    @property
    def child_description(self):
        """A string describing the entities that this Entity relies on."""
        return self._child_description

    @property
    def description(self):
        """A string describing the entity."""
        assert not self._expression_used
        self._handle_used = True
        return self._description

    def has_expression(self):
        """Returns whether there is an expression to use (instead of handle)."""
        return self._expression is not None

    @property
    def handle(self):
        """A "handle", e.g., SymPy symbol or `function.CompositionHandle`."""
        assert not self._expression_used
        self._handle_used = True
        return self._handle

    @property
    def expression(self):
        """An expression representing the entity; possibly None.

        If this is used, then `description` does not need to get used.

        Returns:
          String or None.
        """
        assert not self._handle_used
        self._expression_used = True
        return self._expression

    @property
    def polynomial_variables(self):
        """For when `expression` is not None, and this entity is a polynomial."""
        return self._polynomial_variables

    @property
    def expression_used(self):
        """Returns true if expression instead of handle was used."""
        return self._expression_used

    @property
    def expression_else_handle(self):
        """Returns an expression if present, else the handle."""
        if self.has_expression():
            return self.expression
        else:
            return self.handle

    def __str__(self):
        """Raises value error - should not attempt to convert to string directly."""
        raise ValueError("Should not convert Entity directly to string")
