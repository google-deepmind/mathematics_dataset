# Mathematics Dataset

This dataset code generates mathematical question and answer pairs, from a range
of question types (such as in arithmetic, algebra, probability, etc), at roughly
school-level difficulty. This is designed to test the mathematical learning and
reasoning skills of learning models. Original paper: [Analysing Mathematical
Reasoning Abilities of Neural Models](https://openreview.net/pdf?id=H1gR5iR5FX)
(Saxton, Grefenstette, Hill, Kohli).

In addition to the generation code, **pre-generated** data files are available.

* [Version 1.0]
  (https://console.cloud.google.com/storage/browser/mathematics-dataset)
  (original paper), containing 2 million (question, answer) pairs per module,
  with questions limited to 160 characters in length, and answers to 30
  characters in length. Note the training data for each question type is split
  into "train-easy", "train-medium", and "train-hard". This allows training of
  neural networks (etc) via a curriculum. The data can also be mixed together
  uniformly from these training datasets to obtain the results reported in the
  paper.

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
