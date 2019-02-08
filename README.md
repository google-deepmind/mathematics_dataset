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

## Open-source version

### Getting the source

Clone the mathematics_dataset source code:

```shell
git clone https://github.com/deepmind/mathematics_dataset
```

### Required dependencies

You will need to install some python packages. In particular this package
depends on absl, numpy, six, and sympy.

For example, these packages can be installed via pip:

```shell
pip install absl-py numpy six sympy
```

You will also need a recent version of bazel. If not, follow
[these directions](https://bazel.build/versions/master/docs/install.html).

### Generating examples

For debugging, generated examples can be printed to stdout via the `generate`
script. For example:

```shell
cd mathematics_dataset
bazel build -c opt mathematics_dataset:generate
./bazel-bin/mathematics_dataset/generate --filter=linear_1d
```

will generate example (question, answer) pairs for solving linear equations in
one variable.

We've also included `generate_to_file.py` as an example of how to write the
generated examples to text files. You can use this directly, or adapt it for
your generation and training needs.
