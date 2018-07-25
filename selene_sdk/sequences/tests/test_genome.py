import unittest

import numpy as np

from selene_sdk.sequences.genome import _get_sequence_from_coords
from selene_sdk.sequences.sequence import sequence_to_encoding, \
    encoding_to_sequence


class TestGenome(unittest.TestCase):

    def setUp(self):
        self.bases_arr = np.array(['A', 'C', 'G', 'T'])
        self.bases_encoding = {
            'A': 0, 'C': 1, 'G': 2, 'T': 3,
            'a': 0, 'c': 1, 'g': 2, 't': 3,
        }

        self.len_chrs = {
            "chr1": 104,
            "chr2": 16,
            "chr3": 64,
            "chr4": 8
        }

    def _genome_sequence(self, chrom, start, end, strand):
        base_sequence_pos = "AGCTTCCA"
        base_sequence_neg = "TCGAAGGT"

        repeat_base_seq = int(
            self.len_chrs[chrom] / len(base_sequence_pos))

        sequence = None
        if strand == '+':
            sequence = base_sequence_pos * repeat_base_seq
        else:
            sequence = base_sequence_neg * repeat_base_seq
        return sequence[start:end]

    def test_sequence_to_encoding(self):
        sequence = "ctgCGCAA"
        observed = sequence_to_encoding(sequence, self.bases_encoding, self.bases_arr)
        expected = np.array([
            [0., 1., 0., 0.], [0., 0., 0., 1.],  # ct
            [0., 0., 1., 0.], [0., 1., 0., 0.],  # gC
            [0., 0., 1., 0.], [0., 1., 0., 0.],  # CG
            [1., 0., 0., 0.], [1., 0., 0., 0.]   # AA
        ])
        self.assertSequenceEqual(observed.tolist(), expected.tolist())

    def test_sequence_to_encoding_unknown_bases(self):
        sequence = "AnnUAtCa"
        observed = sequence_to_encoding(sequence, self.bases_encoding, self.bases_arr)
        expected = np.array([
            [1., 0., 0., 0.], [.25, .25, .25, .25],      # An
            [.25, .25, .25, .25], [.25, .25, .25, .25],  # nU
            [1., 0., 0., 0.], [0., 0., 0., 1.],          # At
            [0., 1., 0., 0.], [1., 0., 0., 0.]           # Ca
        ])
        self.assertSequenceEqual(observed.tolist(), expected.tolist())

    def test_encoding_to_sequence(self):
        encoding = np.array([
            [1., 0., 0., 0.], [1., 0., 0., 0.],
            [0., 0., 0., 1.], [0., 0., 1., 0.],
            [0., 1., 0., 0.], [0., 0., 0., 1]])
        observed = encoding_to_sequence(encoding, self.bases_arr, "N")
        expected = "AATGCT"
        self.assertEqual(observed, expected)

    def test_encoding_to_sequence_unknown_bases(self):
        encoding = np.array([
            [0., 0., 1., 0.], [0.25, 0.25, 0.25, 0.25],
            [1., 0., 0., 0.], [0., 0., 0., 1.],
            [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])
        observed = encoding_to_sequence(encoding, self.bases_arr, "N")
        expected = "GNATNN"
        self.assertEqual(observed, expected)

    def test__get_sequence_from_coords_pos_strand(self):
        observed = _get_sequence_from_coords(
            self.len_chrs, self._genome_sequence, "chr1", 0, 14, '+')
        expected = "AGCTTCCAAGCTTC"
        self.assertEqual(observed, expected)

    def test__get_sequence_from_coords_neg_strand(self):
        observed = _get_sequence_from_coords(
            self.len_chrs, self._genome_sequence, "chr3", 59, 64, '-')
        expected = "AAGGT"
        self.assertEqual(observed, expected)

    def test__get_sequence_from_coords_ValueError(self):
        with self.assertRaises(ValueError):
            _get_sequence_from_coords(
                self.len_chrs, self._genome_sequence, "chr2", 1, 10, '=')

    def test__get_sequence_from_coords_out_of_bounds(self):
        observed1 = _get_sequence_from_coords(
            self.len_chrs, self._genome_sequence, "chr2", 17, 20, '+')

        observed2 = _get_sequence_from_coords(
            self.len_chrs, self._genome_sequence,
            "chr4", 2, 10, '-')

        self.assertEqual(observed1, "")
        self.assertEqual(observed2, "")


if __name__ == "__main__":
    unittest.main()
