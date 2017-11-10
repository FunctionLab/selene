import unittest

import numpy as np

from proteus import Genome


class TestGenome(unittest.TestCase):

    def setUp(self):
        self.genome = Genome("data/test_files/fasta/small.fasta")

    def test_chrs_attribute(self):
        """Test that the chromosomes specified in a FASTA file are all stored
        in `chrs`.
        """
        observed = self.genome.chrs
        expected = ["chr1", "chr2", "chr3", "chr4"]
        msg = "Genome `chrs` expected to be {0} but was {1}".format(
            expected, observed)
        self.assertEqual(observed, expected, msg)

    def test_len_chrs_attribute(self):
        """Test that the chromosome sequence lengths are correct.
        """
        observed = self.genome.len_chrs
        expected = {"chr1": 100, "chr2": 100, "chr3": 10, "chr4": 50}
        for chrom, length in observed.items():
            msg = "{0} len expected to be {1} but was {2}".format(
                chrom, expected[chrom], length)
            self.assertEqual(length, expected[chrom], msg)

    def test_get_sequence_unknown_base(self):
        """Test that unknown bases in genomic coordinates are returned
        correctly for both the forward and reverse strand directions.
        """
        chrom = "chr1"
        start, end = (0, 50)

        observed_fwd = self.genome.get_sequence_from_coords(
            chrom, start, end, '+')
        observed_rev = self.genome.get_sequence_from_coords(
            chrom, start, end, '-')

        expected = 'N' * 50

        msg_fwd = "Expected '+' sequence {0} but was {1}".format(
            expected, observed_fwd)
        msg_rev = "Expected '-' sequence {0} but was {1}".format(
            expected, observed_rev)

        self.assertEqual(observed_fwd, expected, msg_fwd)
        self.assertEqual(observed_rev, expected, msg_rev)

    def test_get_sequence_bounds(self):
        """Test that out-of-bounds inputs returns an empty string.
        """
        oob_cases = {
            "negative start": (-2, 4),
            "start past chr len": (10, 11),
            "end past chr len": (9, 11)
        }

        chrom = "chr3"
        strand = '-'

        for case, (start, end) in oob_cases.items():
            observed = self.genome.get_sequence_from_coords(
                chrom, start, end, strand)
            msg = ("Expected empty string for case {0} input [{1}, {2}) "
                   "but was {3}").format(case, start, end, observed)
            self.assertEqual(observed, "", msg)

    def test_get_sequence(self):
        """Test that `get_sequence_from_coords` is correct for both the
        forward and reverse strand directions.
        """
        chrom = "chr2"
        start, end = (27, 35)

        observed_fwd = self.genome.get_sequence_from_coords(
            chrom, start, end, '+')
        observed_rev = self.genome.get_sequence_from_coords(
            chrom, start, end, '-')

        expected_fwd = "ctgCGCAA"
        expected_rev = "TTGCGcag"

        msg_fwd = "Expected '+' sequence {0} but was {1}".format(
            expected_fwd, observed_fwd)
        msg_rev = "Expected '-' sequence {0} but was {1}".format(
            expected_rev, observed_rev)

        self.assertEqual(observed_fwd, expected_fwd, msg_fwd)
        self.assertEqual(observed_rev, expected_rev, msg_rev)

    def test_invalid_strand_side(self):
        """Test that an input of an invalid strand side throws a ValueError.
        """
        chrom = "chr2"
        start, end = (27, 35)

        with self.assertRaises(ValueError):
            self.genome.get_sequence_from_coords(
                chrom, start, end, '=')

    def test_sequence_to_encoding(self):
        """Test that we return the correct one-hot encoding for a standard
        sequence.
        """
        sequence = "ctgCGCAA"
        observed = self.genome.sequence_to_encoding(sequence)
        expected = np.array([
            [0., 1., 0., 0.], [0., 0., 0., 1.],  # ct
            [0., 0., 1., 0.], [0., 1., 0., 0.],  # gC
            [0., 0., 1., 0.], [0., 1., 0., 0.],  # CG
            [1., 0., 0., 0.], [1., 0., 0., 0.]   # AA
        ])
        msg = "Incorrect encoding returned for sequence {0}".format(sequence)
        self.assertSequenceEqual(observed.tolist(), expected.tolist(), msg)

    def test_sequence_to_encoding_unknown_bases(self):
        """Test that we return the correct one-hot encoding for a sequence
        that contains unknown bases.
        """
        sequence = "NnnUAtCa"
        observed = self.genome.sequence_to_encoding(sequence)
        expected = np.array([
            [.25, .25, .25, .25], [.25, .25, .25, .25],  # Nn
            [.25, .25, .25, .25], [.25, .25, .25, .25],  # nU
            [1., 0., 0., 0.], [0., 0., 0., 1.],          # At
            [0., 1., 0., 0.], [1., 0., 0., 0.]           # Ca
        ])
        msg = "Incorrect encoding returned for sequence {0}".format(sequence)
        self.assertSequenceEqual(observed.tolist(), expected.tolist(), msg)

    def test_encoding_to_sequence(self):
        """Test that we return the correct sequence from a one-hot encoding.
        """
        encoding = np.array([
            [.25, .25, .25, .25], [.25, .25, .25, .25],
            [1., 0., 0., 0.], [0., 0., 0., 1.],
            [0., 1., 0., 0.], [1., 0., 0., 0.],
            [0., 0., 1., 0.], [0., 0., 1., 0.]
        ])
        observed = self.genome.encoding_to_sequence(encoding)
        expected = "NNATCAGG"
        msg = "Expected sequence {0} from encoding but got {1}".format(
            expected, observed)
        self.assertEqual(observed, expected, msg)

    def test_get_coords_sequence_encoding_fwd(self):
        """Test that we return the correct one-hot encoding for a given
        genomic coordinates input.
        """
        # chr1, 40, 52
        example1_expected = np.ones((12, 4)) * .25
        example1_expected[10, :] = [0., 0., 0., 1.]
        example1_expected[11, :] = [1., 0., 0., 0.]

        # chr4, 0, 49
        example2_expected = np.ones((49, 4)) * 0.25

        # chr3, 4, 10
        example3_expected = np.zeros((6, 4))
        example3_expected[0:3, :] = 0.25
        example3_expected[[3, 4, 5], [0, 3, 1]] = 1.

        # chr2, 0, 5
        example4_expected = np.zeros((5, 4))
        example4_expected[[0, 1, 2, 3, 4], [3, 3, 2, 1, 3]] = 1.

        examples = {
            ("chr1", 40, 52): example1_expected,
            ("chr4", 0, 49): example2_expected,
            ("chr3", 4, 10): example3_expected,
            ("chr2", 0, 5): example4_expected
        }
        for case, expected in examples.items():
            observed = self.genome.get_encoding_from_coords(*case, strand='+')
            msg = "Incorrect one-hot encoding returned for case {0}".format(
                case)
            self.assertSequenceEqual(
                observed.tolist(), expected.tolist(), msg)


if __name__ == "__main__":
    unittest.main()
