# Copyright 2019 DeepMind Technologies Limited.
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

"""Module setuptools script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup

description = """A synthetic dataset of school-level mathematics questions.

This dataset code generates mathematical question and answer pairs, from a range
of question types (such as in arithmetic, algebra, probability, etc), at roughly
school-level difficulty. This is designed to test the mathematical learning and
reasoning skills of learning models.

Original paper: Analysing Mathematical Reasoning Abilities of Neural Models
(Saxton, Grefenstette, Hill, Kohli) (https://openreview.net/pdf?id=H1gR5iR5FX).
"""

setup(
    name="mathematics_dataset",
    version="1.0.1",
    description="A synthetic dataset of school-level mathematics questions",
    long_description=description,
    author="DeepMind",
    author_email="saxton@google.com",
    license="Apache License, Version 2.0",
    keywords="mathematics dataset",
    url="https://github.com/deepmind/mathematics_dataset",
    packages=find_packages(),
    install_requires=[
        "absl-py>=0.1.0,<0.10",
        "numpy>=1.10,<1.19",
        "six>=1.0,<2.0",
        "sympy>=1.2,<1.6",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
