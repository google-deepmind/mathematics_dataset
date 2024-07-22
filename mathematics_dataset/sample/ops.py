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

"""Mathematical operations used to build up expressions for printing.

We can't use sympy because sympy will automatically simplify many types of
expressions, even with `evaluate=False` passed in. For example:

*   Mul(-2, -3, evaluate=False) gives -(-6), not (-2) x (-3).
*   Add(2, 1, evaluate=False) gives 1 + 2, because the terms are sorted.

As such, it's easier just to work with our own op classes that display precisely
as we created them. This also allows us to use custom symbols for the
expressions, such as the multiplication symbol.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

# Dependency imports
from absl import logging
from mathematics_dataset.sample import number
from mathematics_dataset.util import display
import numpy as np
import six
from six.moves import zip
import sympy


MUL_SYMBOL = "*"
DIV_SYMBOL = "/"
POW_SYMBOL = "**"
GT_SYMBOL = ">"
LT_SYMBOL = "<"
GE_SYMBOL = ">="
LE_SYMBOL = "<="
EQ_SYMBOL = "="
NE_SYMBOL = "!="


# Operator precedence levels. Used to insert brackets if necessary.
_EQ_PRECEDENCE = 0
_CONSTANT_PRECEDENCE = 1
_POW_PRECEDENCE = 2
_SQRT_PRECEDENCE = 3
_MUL_PRECEDENCE = 4
_ADD_PRECEDENCE = 5


def bracketed(child, parent, bracket_if_same_precedence):
    """Returns string representation of `child`, possibly bracketed.

    Args:
      child: Instance of `Op` or a valid value for `ConstantOp`.
      parent: Instance of `Op`. Used to determine whether `child` needs to be
          bracketed first before appearing in the parent op's expression.
      bracket_if_same_precedence: Whether to bracket if the child has the same
          operator precedence as the parent.

    Returns:
      String representation of `child`.
    """
    if not isinstance(child, Op):
        child = Constant(child)

    child_precedence = child.precedence
    parent_precedence = parent.precedence
    if parent_precedence > child_precedence or (
        parent_precedence == child_precedence and not bracket_if_same_precedence
    ):
        return str(child)
    else:
        return "({})".format(child)


def _flatten(iterable):
    """Returns list."""
    if isinstance(iterable, (list, tuple)):
        result = list(iterable)
    else:
        assert isinstance(iterable, dict)
        keys = sorted(six.iterkeys(iterable))
        result = [iterable[key] for key in keys]
    # Check we don't have any hierarchy in the structure (otherwise would need
    # to use something recursive like tf.contrib.framework.nest.flatten).
    for item in result:
        assert not isinstance(item, (list, tuple, dict))
    return result


def _pack_sequence_as(example, flat):
    if isinstance(example, list) or isinstance(example, tuple):
        return flat
    else:
        assert isinstance(example, dict)
        keys = sorted(six.iterkeys(example))
        return {key: value for key, value in zip(keys, flat)}


@six.add_metaclass(abc.ABCMeta)
class Op(object):
    """An operation.

    This needs to support being transformed into sympy (and possibly in the future
    other types such as an appropriately formatted string), when given the op
    arguments.
    """

    def __init__(self, children):
        """Initialize this `Op` base class.

        Args:
          children: Iterable structure containing child ops.
        """
        assert isinstance(children, (list, dict, tuple))
        flat_children = _flatten(children)
        flat_children = [
            child if isinstance(child, Op) else Constant(child)
            for child in flat_children
        ]
        children = _pack_sequence_as(children, flat_children)
        self._children = children

    @property
    def children(self):
        """Returns iterable or dict over immediate children."""
        return self._children

    def descendants(self):
        """Returns list of all descendants (self, children, grandchildren, etc)."""
        descendants = [self]
        flat_children = _flatten(self._children)
        for child in flat_children:
            descendants += child.descendants()
        return descendants

    @abc.abstractmethod
    def __str__(self):
        """Returns a string format of this op."""

    @abc.abstractmethod
    def sympy(self):
        """Returns the sympifcation of this op."""

    def _sympy_(self):
        """Convenience method to automatically sympify this object."""
        try:
            return self.sympy()
        except AttributeError as e:
            # Note: we print this error here, before raising it again, because sympy
            # will think `AttributeError` refers to this object not having a `_sympy_`
            # method, rather than having it, which leads to otherwise confusing error
            # messages.
            logging.error("Encountered attribute error while trying to sympify: %s", e)
            raise e

    @abc.abstractproperty
    def precedence(self):
        """Returns the precedence (integer) of this op."""


class Constant(Op):
    """Returns a constant value; a nullary op."""

    def __init__(self, value):
        super(Constant, self).__init__([])
        if isinstance(value, six.integer_types):
            value = sympy.Integer(value)
        self._value = value

    def __str__(self):
        return str(self._value)

    def sympy(self):
        return self._value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def _is_simple(self):
        """Returns whether it's a simple number, rather than a division or neg."""
        if isinstance(self._value, sympy.Symbol):
            return True
        elif (
            isinstance(self._value, int)
            or isinstance(self._value, sympy.Integer)
            or isinstance(self._value, display.Decimal)
            or isinstance(self._value, np.int64)
            or isinstance(self._value, np.int32)
        ):
            return self._value >= 0
        elif isinstance(self._value, sympy.Rational):
            return False
        elif isinstance(self._value, sympy.Function):
            return True
        else:
            raise ValueError("Unknown type {}".format(type(self._value)))

    @property
    def precedence(self):
        if self._is_simple():
            return _CONSTANT_PRECEDENCE
        else:
            return _MUL_PRECEDENCE


class _SumLikeOp(Op):
    """Abstract op for sum-like terms which may contain negative entries."""

    @abc.abstractmethod
    def expanded_signs_and_terms(self):
        """Returns a list of arguments, plus any sub-arguments from sub-adds.

        E.g., if this op is `Add(Add(2, Neg(3)), Mul(4, 5), 1)`, then will return
        `[(True, 2), (False, 3), (True, Mul(4, 5)), (True, 1)]` (the arguments of
        the inner add have been extracted).
        """

    def __str__(self):
        signs_and_terms = self.expanded_signs_and_terms()
        if not signs_and_terms:
            return "0"
        for i, (sign, term) in enumerate(signs_and_terms):
            if i == 0:
                if sign:
                    expression = bracketed(term, self, True)
                else:
                    expression = "-" + bracketed(term, self, True)
            else:
                if sign:
                    expression += " + " + bracketed(term, self, True)
                else:
                    expression += " - " + bracketed(term, self, True)
        return expression


class Identity(_SumLikeOp):
    """The identity op (a unitary op)."""

    def __init__(self, input_):
        super(Identity, self).__init__({"input": input_})

    def expanded_signs_and_terms(self):
        if isinstance(self.children["input"], _SumLikeOp):
            return self.children["input"].expanded_signs_and_terms()
        else:
            return [(True, self.children["input"])]

    def __str__(self):
        return str(self.children["input"])

    def sympy(self):
        return self.children["input"].sympy()

    @property
    def precedence(self):
        return self.children["input"].precedence


class Neg(_SumLikeOp):
    """Negation, a unary op. Also has special display when appearing in a sum."""

    def __init__(self, arg):
        super(Neg, self).__init__({"input": arg})

    def expanded_signs_and_terms(self):
        if isinstance(self.children["input"], _SumLikeOp):
            inner_signs_and_terms = self.children["input"].expanded_signs_and_terms()
            return [(not sign, term) for (sign, term) in inner_signs_and_terms]
        else:
            return [(False, self.children["input"])]

    def sympy(self):
        return -sympy.sympify(self.children["input"])

    def inner(self):
        return self.children["input"]

    @property
    def precedence(self):
        return _ADD_PRECEDENCE


class Add(_SumLikeOp):
    """Addition."""

    def __init__(self, *args):
        super(Add, self).__init__(args)

    def expanded_signs_and_terms(self):
        """Returns a list of arguments, plus any sub-arguments from sub-adds.

        E.g., if this op is `Add(Add(2, 3), Mul(4, 5), 1)`, then will return
        `[2, 3, Mul(4, 5), 1]` (the arguments of the inner add have been extracted).
        """
        expanded = []
        for arg in self.children:
            if isinstance(arg, _SumLikeOp):
                expanded += arg.expanded_signs_and_terms()
            else:
                expanded.append((True, arg))
        return expanded

    def sympy(self):
        return sympy.Add(*[sympy.sympify(arg) for arg in self.children])

    @property
    def precedence(self):
        return _ADD_PRECEDENCE


class Sub(Op):
    """Subtraction."""

    def __init__(self, left, right):
        super(Sub, self).__init__({"left": left, "right": right})

    def __str__(self):
        return (
            bracketed(self.children["left"], self, False)
            + " - "
            + bracketed(self.children["right"], self, True)
        )

    def sympy(self):
        return sympy.Add(self.children["left"], sympy.Mul(-1, self.children["right"]))

    @property
    def precedence(self):
        return _ADD_PRECEDENCE


class Mul(Op):
    """Multiplication."""

    def __init__(self, *args):
        super(Mul, self).__init__(args)

    def __str__(self):
        if not self.children:
            return "1"
        else:
            args = [bracketed(arg, self, False) for arg in self.children]
            return MUL_SYMBOL.join(args)

    def sympy(self):
        return sympy.Mul(*[sympy.sympify(arg) for arg in self.children])

    @property
    def precedence(self):
        return _MUL_PRECEDENCE


class Div(Op):
    """Division."""

    def __init__(self, numer, denom):
        super(Div, self).__init__({"numer": numer, "denom": denom})

    def __str__(self):
        return "{}{}{}".format(
            bracketed(self.children["numer"], self, True),
            DIV_SYMBOL,
            bracketed(self.children["denom"], self, True),
        )

    def sympy(self):
        return sympy.Mul(self.children["numer"], sympy.Pow(self.children["denom"], -1))

    @property
    def precedence(self):
        return _MUL_PRECEDENCE


class Pow(Op):
    """Power a to the power b."""

    def __init__(self, a, b):
        super(Pow, self).__init__({"a": a, "b": b})

    def __str__(self):
        return "{}{}{}".format(
            bracketed(self.children["a"], self, True),
            POW_SYMBOL,
            bracketed(self.children["b"], self, True),
        )

    def sympy(self):
        return sympy.Pow(
            sympy.sympify(self.children["a"]), sympy.sympify(self.children["b"])
        )

    @property
    def precedence(self):
        return _POW_PRECEDENCE


class Sqrt(Op):
    """Square root of a value."""

    def __init__(self, a):
        super(Sqrt, self).__init__({"a": a})

    def __str__(self):
        return "sqrt({})".format(self.children["a"])

    def sympy(self):
        return sympy.sqrt(self.children["a"])

    @property
    def precedence(self):
        return _POW_PRECEDENCE


class Eq(Op):
    """Equality."""

    def __init__(self, left, right):
        super(Eq, self).__init__({"left": left, "right": right})

    def __str__(self):
        return "{} = {}".format(self.children["left"], self.children["right"])

    def sympy(self):
        return sympy.Eq(self.children["left"], self.children["right"])

    @property
    def precedence(self):
        return _EQ_PRECEDENCE


def number_constants(expressions):
    """Returns list of integer, rational, decimal constants in the expressions."""
    if isinstance(expressions, Op):
        expressions = [expressions]
    descendants = []
    for expression in expressions:
        descendants += expression.descendants()
    candidate_constants = [op for op in descendants if isinstance(op, Constant)]
    return [
        constant
        for constant in candidate_constants
        if number.is_integer_or_rational_or_decimal(constant.value)
    ]
