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

"""Containers for "[example] problems" (i.e., question/answer) pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from mathematics_dataset.util import composition


def question(context, template, **kwargs):
    """Makes a question, using the given context and template.

    The format is similar to that for python's `format` function, for example:

    ```
    question(context, 'What is {} plus {p} over {q}?', 2, p=3, q=4)
    ```

    The main difference between this and the standard python formatting is that
    this understands `Entity`s in the arguments, and will do appropriate expansion
    of text and prefixing of their descriptions.

    Arguments:
      context: Instance of `composition.Context`, for extracting entities needed
          for describing the problem.
      template: A string, like "Calculate the value of {exp}.".
      **kwargs: A dictionary mapping arguments to values, e.g.,
          `{'exp': sympy.Add(2, 3, evaluate=False)}`.

    Returns:
      String.
    """
    assert isinstance(context, composition.Context)
    assert isinstance(template, str)
    prefix, kwargs = composition.expand_entities(context, **kwargs)
    if prefix:
        prefix += " "
    return prefix + template.format(**kwargs)


Problem = collections.namedtuple("Problem", ("question", "answer"))
