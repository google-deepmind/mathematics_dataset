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

"""Probability questions (sampling, independence, expectations, ...)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import random
import string

# Dependency imports
from mathematics_dataset import example
from mathematics_dataset.modules import train_test_split
from mathematics_dataset.util import combinatorics
from mathematics_dataset.util import composition
from mathematics_dataset.util import display
from mathematics_dataset.util import probability
import numpy as np
from six.moves import range
from six.moves import zip


_LETTERS = string.ascii_lowercase

_MAX_FRAC_TRIVIAL_PROB = 0.1

# Maximum number of colours and objects in a bag.
_MAX_DISTINCT_LETTERS = 6
_MAX_TOTAL_LETTERS = 20
_MAX_LETTER_REPEAT = 10

_SWR_SAMPLE_COUNT = [2, 4]
_SWR_SAMPLE_COUNT_EXTRAPOLATE = [5, 5]

_GERUNDS = {
    "pick": "picking",
}


def _make_modules(is_train):
    """Returns modules, with split based on the boolean `is_train`."""
    return {
        "swr_p_sequence": functools.partial(
            swr_prob_sequence, is_train=is_train, sample_range=_SWR_SAMPLE_COUNT
        ),
        "swr_p_level_set": functools.partial(
            swr_prob_level_set, is_train=is_train, sample_range=_SWR_SAMPLE_COUNT
        ),
    }


def train(entropy_fn):
    """Returns dict of training modules."""
    del entropy_fn  # unused
    return _make_modules(is_train=True)


def test():
    """Returns dict of testing modules."""
    return _make_modules(is_train=False)


def test_extra():
    """Returns dict of extrapolation testing modules."""
    return {
        "swr_p_sequence_more_samples": functools.partial(
            swr_prob_sequence, is_train=None, sample_range=_SWR_SAMPLE_COUNT_EXTRAPOLATE
        ),
        "swr_p_level_set_more_samples": functools.partial(
            swr_prob_level_set,
            is_train=None,
            sample_range=_SWR_SAMPLE_COUNT_EXTRAPOLATE,
        ),
    }


def _sequence_event(values, length, verb):
    """Returns sequence (finite product) event.

    Args:
      values: List of values to sample from.
      length: Length of the sequence to generate.
      verb: Verb in infinitive form.

    Returns:
      Instance of `probability.FiniteProductEvent`, together with a text
      description.
    """
    del verb  # unused
    samples = [random.choice(values) for _ in range(length)]
    events = [probability.DiscreteEvent([sample]) for sample in samples]
    event = probability.FiniteProductEvent(events)
    sequence = "".join(str(sample) for sample in samples)
    event_description = "sequence {sequence}".format(sequence=sequence)
    return event, event_description


def _word_series(words, conjunction="and"):
    """Combines the words using commas and the final conjunction."""
    len_words = len(words)
    if len_words == 0:
        return ""
    if len_words == 1:
        return words[0]
    return "{} {} {}".format(", ".join(words[:-1]), conjunction, words[-1])


def _level_set_event(values, length, verb):
    """Generates `LevelSetEvent`; see _generate_sequence_event."""
    counts = combinatorics.uniform_non_negative_integers_with_sum(len(values), length)
    counts_dict = dict(list(zip(values, counts)))
    event = probability.CountLevelSetEvent(counts_dict)

    shuffled_values = list(values)
    random.shuffle(shuffled_values)

    counts_and_values = [
        "{} {}".format(counts_dict[value], value)
        for value in shuffled_values
        if counts_dict[value] > 0
    ]
    counts_and_values = _word_series(counts_and_values)
    template = random.choice(
        [
            "{verbing} {counts_and_values}",
        ]
    )
    verbing = _GERUNDS[verb]
    event_description = template.format(
        counts_and_values=counts_and_values, verbing=verbing
    )
    return event, event_description


LetterBag = collections.namedtuple(
    "LetterBag", ("weights", "random_variable", "letters_distinct", "bag_contents")
)


def _sample_letter_bag(is_train, min_total):
    """Samples a "container of letters" and returns info on it."""
    while True:
        num_distinct_letters = random.randint(1, _MAX_DISTINCT_LETTERS)
        num_letters_total = random.randint(
            max(num_distinct_letters, min_total),
            min(_MAX_TOTAL_LETTERS, num_distinct_letters * _MAX_LETTER_REPEAT),
        )
        letter_counts = combinatorics.uniform_positive_integers_with_sum(
            num_distinct_letters, num_letters_total
        )

        # Test/train split.
        if (
            is_train is None
            or train_test_split.is_train(sorted(letter_counts)) == is_train
        ):
            break

    letters_distinct = random.sample(_LETTERS, num_distinct_letters)
    weights = {i: 1 for i in range(num_letters_total)}

    letters_with_repetition = []
    for letter, count in zip(letters_distinct, letter_counts):
        letters_with_repetition += [letter] * count
    random.shuffle(letters_with_repetition)

    random_variable = probability.DiscreteRandomVariable(
        {i: letter for i, letter in enumerate(letters_with_repetition)}
    )

    if random.choice([False, True]):
        bag_contents = "".join(letters_with_repetition)
    else:
        letters_and_counts = [
            "{}: {}".format(letter, count)
            for letter, count in zip(letters_distinct, letter_counts)
        ]
        bag_contents = "{" + ", ".join(letters_and_counts) + "}"

    return LetterBag(
        weights=weights,
        random_variable=random_variable,
        letters_distinct=letters_distinct,
        bag_contents=bag_contents,
    )


def _swr_space(is_train, sample_range):
    """Returns probability space for sampling without replacement."""
    num_sampled = random.randint(*sample_range)
    sample = _sample_letter_bag(is_train=is_train, min_total=num_sampled)

    space = probability.SampleWithoutReplacementSpace(sample.weights, num_sampled)

    random_variable = probability.FiniteProductRandomVariable(
        [sample.random_variable] * num_sampled
    )

    random_variable.description = (
        str(display.StringNumber(num_sampled))
        + " letters picked without replacement from "
        + sample.bag_contents
    )

    return sample.letters_distinct, space, random_variable


def _sample_without_replacement_probability_question(is_train, event_fn, sample_range):
    """Question for prob of some event when sampling without replacement."""

    def too_big(event_in_space):
        if isinstance(event_in_space, probability.SequenceEvent):
            size = len(event_in_space.all_sequences())
        else:
            assert isinstance(event_in_space, probability.FiniteProductEvent)
            size = np.prod([len(event.values) for event in event_in_space.events])
        return size > int(2e5)

    allow_trivial_prob = random.random() < _MAX_FRAC_TRIVIAL_PROB

    while True:
        distinct_letters, space, random_variable = _swr_space(is_train, sample_range)

        event, event_description = event_fn(
            values=distinct_letters, length=space.n_samples, verb="pick"
        )
        event_in_space = random_variable.inverse(event)
        if too_big(event_in_space):
            continue
        answer = space.probability(event_in_space)
        if answer not in [0, 1] or allow_trivial_prob:
            break

    context = composition.Context()

    template = random.choice(
        [
            "{random_variable_capitalize}. What is prob of {event}?",
            "{random_variable_capitalize}. Give prob of {event}.",
            "What is prob of {event} when {random_variable}?",
            "Calculate prob of {event} when {random_variable}.",
        ]
    )
    question = example.question(
        context,
        template,
        random_variable=random_variable.description,
        random_variable_capitalize=(str(random_variable.description).capitalize()),
        event=event_description,
    )
    return example.Problem(question, answer)


def swr_prob_sequence(is_train, sample_range):
    """Probability of given sequence when sampling without replacement."""
    return _sample_without_replacement_probability_question(
        is_train=is_train, event_fn=_sequence_event, sample_range=sample_range
    )


def swr_prob_level_set(is_train, sample_range):
    """Probability of given level set when sampling without replacement."""
    return _sample_without_replacement_probability_question(
        is_train=is_train, event_fn=_level_set_event, sample_range=sample_range
    )
