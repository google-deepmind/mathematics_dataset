# Mathematics Dataset

This dataset code generates mathematical question and answer pairs, from a range
of question types at roughly school-level difficulty. This is designed to test
the mathematical learning and algebraic reasoning skills of learning models.

Original paper: [Analysing Mathematical
Reasoning Abilities of Neural Models](https://openreview.net/pdf?id=H1gR5iR5FX)
(Saxton, Grefenstette, Hill, Kohli).

## Example questions

```
Question: Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r.
Answer: 4

Question: Calculate -841880142.544 + 411127.
Answer: -841469015.544

Question: Let x(g) = 9*g + 1. Let q(c) = 2*c + 1. Let f(i) = 3*i - 39. Let w(j) = q(x(j)). Calculate f(w(a)).
Answer: 54*a - 30

Question: Let e(l) = l - 6. Is 2 a factor of both e(9) and 2?
Answer: False

Question: Let u(n) = -n**3 - n**2. Let e(c) = -2*c**3 + c. Let l(j) = -118*e(j) + 54*u(j). What is the derivative of l(a)?
Answer: 546*a**2 - 108*a - 118

Question: Three letters picked without replacement from qqqkkklkqkkk. Give prob of sequence qql.
Answer: 1/110
```

## Pre-generated data

[Pre-generated files](https://console.cloud.google.com/storage/browser/mathematics-dataset)

### Version 1.0

This is the version released with the original paper. It contains 2 million
(question, answer) pairs per module, with questions limited to 160 characters in
length, and answers to 30 characters in length. Note the training data for each
question type is split into "train-easy", "train-medium", and "train-hard". This
allows training models via a curriculum. The data can also be mixed together
uniformly from these training datasets to obtain the results reported in the
paper. Categories:

* **algebra** (linear equations, polynomial roots, sequences)
* **arithmetic** (pairwise operations and mixed expressions, surds)
* **calculus** (differentiation)
* **comparison** (closest numbers, pairwise comparisons, sorting)
* **measurement** (conversion, working with time)
* **numbers** (base conversion, remainders, common divisors and multiples,
  primality, place value, rounding numbers)
* **polynomials** (addition, simplification, composition, evaluating, expansion)
* **probability** (sampling without replacement)

## Getting the source

### PyPI

The easiest way to get the source is to use pip:

```shell
$ pip install mathematics_dataset
```

### From GitHub

Alternately you can get the source by cloning the mathematics_dataset
repository:

```shell
$ git clone https://github.com/deepmind/mathematics_dataset
$ pip install --upgrade mathematics_dataset/
```

## Generating examples

Generated examples can be printed to stdout via the `generate` script. For
example:

```shell
python -m mathematics_dataset.generate --filter=linear_1d
```

will generate example (question, answer) pairs for solving linear equations in
one variable.

We've also included `generate_to_file.py` as an example of how to write the
generated examples to text files. You can use this directly, or adapt it for
your generation and training needs.

