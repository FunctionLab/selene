import unittest

import numpy as np

from selene.model_predict import in_silico_mutagenesis_sequences, \
        in_silico_mutagenesis_predict


class TestModelPredict(unittest.TestCase):

    def setUp(self):
        self.bases_arr = ['A', 'C', 'G', 'T']
        self.bases_encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.input_sequence = "ATCCG"

    def test_in_silico_muta_sequences_single(self):
        observed = in_silico_mutagenesis_sequences("ATCCG")
        expected = [
            (0, 'C'), (0, 'G'), (0, 'T'),
            (1, 'A'), (1, 'C'), (1, 'G'),
            (2, 'A'), (2, 'G'), (2, 'T'),
            (3, 'A'), (3, 'G'), (3, 'T'),
            (4, 'A'), (4, 'C'), (4, 'T')]

        expected_lists = [[e] for e in expected]
        self.assertListEqual(observed, expected_lists)

    def test_in_silico_muta_sequences_double(self):
        observed = in_silico_mutagenesis_sequences(
            "ATC", mutate_n_bases=2)
        expected = [
            [(0, 'C'), (1, 'A')], [(0, 'G'), (1, 'A')], [(0, 'T'), (1, 'A')],
            [(0, 'C'), (1, 'C')], [(0, 'G'), (1, 'C')], [(0, 'T'), (1, 'C')],
            [(0, 'C'), (1, 'G')], [(0, 'G'), (1, 'G')], [(0, 'T'), (1, 'G')],

            [(0, 'C'), (2, 'A')], [(0, 'G'), (2, 'A')], [(0, 'T'), (2, 'A')],
            [(0, 'C'), (2, 'G')], [(0, 'G'), (2, 'G')], [(0, 'T'), (2, 'G')],
            [(0, 'C'), (2, 'T')], [(0, 'G'), (2, 'T')], [(0, 'T'), (2, 'T')],

            [(1, 'A'), (2, 'A')], [(1, 'C'), (2, 'A')], [(1, 'G'), (2, 'A')],
            [(1, 'A'), (2, 'G')], [(1, 'C'), (2, 'G')], [(1, 'G'), (2, 'G')],
            [(1, 'A'), (2, 'T')], [(1, 'C'), (2, 'T')], [(1, 'G'), (2, 'T')],
        ]
        self.assertCountEqual(observed, expected)

if __name__ == "__main__":
    unittest.main()
