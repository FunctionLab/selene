import unittest

import numpy as np

from selene_sdk.sequences.proteome import Proteome
from selene_sdk.sequences.proteome import _get_sequence_from_coords


class TestProteome(unittest.TestCase):
    def setUp(self):
        self.proteome = Proteome("selene_sdk/sequences/tests/files/small.faa")
        self.fill_value = np.divide(1, len(Proteome.BASES_ARR), dtype=np.float32)

    def test_sequence_to_encoding(self):
        sequence = "ARNDCEQGHILKMFPSTWYVARNDCEQGHILKMFPSTWYV"
        observed = Proteome.sequence_to_encoding(sequence)
        expected = np.vstack([np.identity(20), np.identity(20)])
        self.assertSequenceEqual(observed.tolist(), expected.tolist())

    def test_sequence_to_encoding_unknown_bases(self):
        sequence = "ARNDCEQGHILKMFPSTWYVXARNDCEQGHILKMFPSTWYV"
        observed = Proteome.sequence_to_encoding(sequence)
        expected = np.vstack([np.identity(20), np.full(20, self.fill_value), np.identity(20)])
        self.assertSequenceEqual(observed.tolist(), expected.tolist())

    def test_encoding_to_sequence(self):
        encoding = np.identity(20)
        observed = Proteome.encoding_to_sequence(encoding)
        expected = "ARNDCEQGHILKMFPSTWYV"
        self.assertEqual(observed, expected)

    def test_encoding_to_sequence_unknown_bases(self):
        encoding = np.vstack([np.full(20, self.fill_value), np.identity(20)])
        observed = Proteome.encoding_to_sequence(encoding)
        expected = "XARNDCEQGHILKMFPSTWYV"
        self.assertEqual(observed, expected)

    def test_get_prots(self):
        self.assertSequenceEqual(self.proteome.get_prots(), ["prot0", "prot1"])

    def test_sequence_in_bounds_is_out_of_bounds(self):
        self.assertEqual(self.proteome.coords_in_bounds("prot0", 10000, 100001),
                         False)

    def test_get_encoding_from_coords(self):
        encoding = self.proteome.get_encoding_from_coords("prot0", 0, 42)
        expected = np.vstack([np.identity(20), np.full(20, self.fill_value), np.identity(20)])
        self.assertSequenceEqual(encoding.tolist(), expected.tolist())

    def test_get_sequence_from_coords(self):
        sequence = self.proteome.get_sequence_from_coords("prot0", 22, 42)
        expected = "RNDCEQGHILKMFPSTWYV"
        self.assertEqual(sequence, expected)

    def test_get_sequence_from_coords_out_of_bounds(self):
        sequence = self.proteome.get_sequence_from_coords("prot0", 55, 59)
        self.assertEqual(sequence, "")

if __name__ == "__main__":
    unittest.main()
