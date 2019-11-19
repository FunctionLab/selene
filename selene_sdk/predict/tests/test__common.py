"""
Test methods in the _common methods module
"""
import numpy as np
import unittest

from selene_sdk.predict._common import get_reverse_complement_encoding
from selene_sdk.sequences import Genome


class TestReverseComplement(unittest.TestCase):

    def setUp(self):
        self.example_encoding = np.array(
            [[0., 0., 0., 1.],
             [1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0.25, 0.25, 0.25, 0.25]]
        )

    def test_rc_encoding_default(self):
        observed = get_reverse_complement_encoding(
            self.example_encoding,
            Genome.BASES_ARR,
            Genome.COMPLEMENTARY_BASE_DICT)
        expected = [
            [0.25, 0.25, 0.25, 0.25],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
        ]
        self.assertEqual(observed.tolist(), expected)

    def test_rc_encoding_nonstandard_ordering(self):
        observed = get_reverse_complement_encoding(
            self.example_encoding,
            ['A', 'T', 'G', 'C'],
            Genome.COMPLEMENTARY_BASE_DICT)
        expected = [
            [0.25, 0.25, 0.25, 0.25],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
        ]
        self.assertEqual(observed.tolist(), expected)


if __name__ == "__main__":
    unittest.main()
