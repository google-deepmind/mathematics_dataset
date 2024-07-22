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

"""Prints to stdout different curriculum questions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import textwrap

# Dependency imports
from absl import app
from absl import flags
from absl import logging
from mathematics_dataset import generate_settings
from mathematics_dataset.modules import modules
import six
from six.moves import range


FLAGS = flags.FLAGS

flags.DEFINE_string("filter", "", "restrict to matching module names")
flags.DEFINE_integer("per_train_module", 10, "Num of examples per train module")
flags.DEFINE_integer("per_test_module", 10, "Num of examples per test module")
flags.DEFINE_bool("show_dropped", False, "Whether to print dropped questions")


filtered_modules = collections.OrderedDict([])
counts = {}


def _make_entropy_fn(level, num_levels):
    """This returns a function that returns a subrange of entropy.

    E.g., if level=1 (medium) and num_levels=3, then the returned function will
    map the range [x, x + y] to [x + y/3, x + 2y/3].

    Args:
      level: Integer in range [0, num_levels - 1].
      num_levels: Number of difficulty levels.

    Returns:
      Function to restrict entropy range.
    """
    lower = level / num_levels
    upper = (level + 1) / num_levels

    def modify_entropy(range_):
        assert len(range_) == 2
        length = range_[1] - range_[0]
        return (range_[0] + lower * length, range_[0] + upper * length)

    return modify_entropy


def _filter_and_flatten(modules_):
    """Returns flattened dict, filtered according to FLAGS."""
    flat = collections.OrderedDict()

    def add(submodules, prefix=None):
        for key, module_or_function in six.iteritems(submodules):
            full_name = prefix + "__" + key if prefix is not None else key
            if isinstance(module_or_function, dict):
                add(module_or_function, full_name)
            else:
                if FLAGS.filter not in full_name:
                    continue
                flat[full_name] = module_or_function

    add(modules_)

    # Make sure list of modules are in deterministic order. This is important when
    # generating across multiple machines.
    flat = collections.OrderedDict(
        [(key, flat[key]) for key in sorted(six.iterkeys(flat))]
    )

    return flat


def init_modules(train_split=False):
    """Inits the dicts containing functions for generating modules."""
    if filtered_modules:
        return  # already initialized

    all_modules = collections.OrderedDict([])
    if train_split:
        all_modules["train-easy"] = modules.train(_make_entropy_fn(0, 3))
        all_modules["train-medium"] = modules.train(_make_entropy_fn(1, 3))
        all_modules["train-hard"] = modules.train(_make_entropy_fn(2, 3))
    else:
        all_modules["train"] = modules.train(_make_entropy_fn(0, 1))

    all_modules["interpolate"] = modules.test()
    all_modules["extrapolate"] = modules.test_extra()

    counts["train"] = FLAGS.per_train_module
    counts["train-easy"] = FLAGS.per_train_module // 3
    counts["train-medium"] = FLAGS.per_train_module // 3
    counts["train-hard"] = FLAGS.per_train_module // 3
    counts["interpolate"] = FLAGS.per_test_module
    counts["extrapolate"] = FLAGS.per_test_module

    for regime_, modules_ in six.iteritems(all_modules):
        filtered_modules[regime_] = _filter_and_flatten(modules_)


def sample_from_module(module):
    """Samples a problem, ignoring samples with overly long questions / answers.

    Args:
      module: Callable returning a `Problem`.

    Returns:
      Pair `(problem, num_dropped)`, where `problem` is an instance of `Problem`
      and `num_dropped` is an integer >= 0 indicating the number of samples that
      were dropped.
    """
    num_dropped = 0
    while True:
        problem = module()
        question = str(problem.question)
        if len(question) > generate_settings.MAX_QUESTION_LENGTH:
            num_dropped += 1
            if FLAGS.show_dropped:
                logging.warning("Dropping question: %s", question)
            continue
        answer = str(problem.answer)
        if len(answer) > generate_settings.MAX_ANSWER_LENGTH:
            num_dropped += 1
            if FLAGS.show_dropped:
                logging.warning("Dropping question with answer: %s", answer)
            continue
        return problem, num_dropped


def main(unused_argv):
    """Prints Q&As from modules according to FLAGS.filter."""
    init_modules()

    text_wrapper = textwrap.TextWrapper(
        width=80, initial_indent=" ", subsequent_indent="  "
    )

    for regime, flat_modules in six.iteritems(filtered_modules):
        per_module = counts[regime]
        for module_name, module in six.iteritems(flat_modules):
            # These magic print constants make the header bold.
            print("\033[1m{}/{}\033[0m".format(regime, module_name))
            num_dropped = 0
            for _ in range(per_module):
                problem, extra_dropped = sample_from_module(module)
                num_dropped += extra_dropped
                text = text_wrapper.fill(
                    "{}  \033[92m{}\033[0m".format(problem.question, problem.answer)
                )
                print(text)
            if num_dropped > 0:
                logging.warning("Dropped %d examples", num_dropped)


if __name__ == "__main__":
    app.run(main)
