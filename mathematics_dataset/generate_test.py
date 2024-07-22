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

"""Tests for mathematics_dataset.generate."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import absltest
from absl.testing import parameterized
from mathematics_dataset import generate
import six
from six.moves import range


class GenerateTest(parameterized.TestCase):

    def testMakeEntropyFn(self):
        entropy_full = generate._make_entropy_fn(0, 1)
        self.assertEqual(entropy_full((2, 3)), (2, 3))
        entropy_third = generate._make_entropy_fn(2, 3)
        self.assertEqual(entropy_third((3, 6)), (5, 6))

    @parameterized.parameters("train", "interpolate", "extrapolate")
    def testGenerate(self, regime):
        generate.init_modules()
        for module in six.itervalues(generate.filtered_modules[regime]):
            for _ in range(3):
                question = module()
                str(question)


if __name__ == "__main__":
    absltest.main()
