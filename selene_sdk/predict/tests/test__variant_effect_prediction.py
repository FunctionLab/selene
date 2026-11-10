"""
Test the sequences built for variants in the _variant_effect_prediction module
"""
import random
import unittest

from selene_sdk.predict._variant_effect_prediction import _process_alt
from selene_sdk.sequences import Genome


class _Reference(object):
    """In-memory stand-in for a Genome: one chromosome, 1000 random bases."""

    def __init__(self, sequence):
        self.sequence = sequence

    def get_sequence_from_coords(self, chrom, start, end, pad=False):
        return self.sequence[start:end]

    def sequence_to_encoding(self, sequence):
        return Genome.sequence_to_encoding(sequence)


class TestProcessAlt(unittest.TestCase):

    def setUp(self):
        rng = random.Random(0)
        self.genome = ''.join(rng.choice('ACGT') for _ in range(1000))
        self.reference = _Reference(self.genome)
        self.radius = 50

    def _alt_window(self, pos, ref, alt):
        """What variant_effect_prediction passes in and what _process_alt returns, decoded."""
        center = pos + len(ref) // 2
        start, end = center - self.radius, center + self.radius
        wt = Genome.sequence_to_encoding(self.genome[start:end])
        encoding = _process_alt('chr1', pos, ref, alt, start, end, wt, self.reference)
        return start, end, Genome.encoding_to_sequence(encoding)

    def _alt_genome(self, pos, ref, alt):
        # pos is the 1-based VCF position of the first REF base
        self.assertEqual(self.genome[pos - 1:pos - 1 + len(ref)], ref)
        return self.genome[:pos - 1] + alt + self.genome[pos - 1 + len(ref):]

    def test_substitution(self):
        pos = 500
        ref = self.genome[pos - 1]
        alt = 'A' if ref != 'A' else 'C'
        start, end, seq = self._alt_window(pos, ref, alt)
        self.assertEqual(seq, self._alt_genome(pos, ref, alt)[start:end])

    def test_insertion(self):
        for n in (1, 2, 3, 10):
            pos = 500
            ref = self.genome[pos - 1]
            alt = ref + 'T' * n
            start, end, seq = self._alt_window(pos, ref, alt)
            self.assertEqual(len(seq), end - start)
            self.assertIn(seq, self._alt_genome(pos, ref, alt))

    def test_deletion(self):
        # VCF-style deletions (REF = anchor base + deleted bases, ALT = anchor base): the window must be a stretch of
        # the alternative genome, with the deleted bases replaced exactly where they are
        for pos in (400, 500, 501):
            for n in (1, 2, 3, 10, 25):
                ref = self.genome[pos - 1:pos + n]
                alt = ref[0]
                start, end, seq = self._alt_window(pos, ref, alt)
                lhs_start = start - len(ref) // 2 + len(alt) // 2
                expected = self._alt_genome(pos, ref, alt)[lhs_start:lhs_start + end - start]
                self.assertEqual(seq, expected, msg='%d-bp deletion at %d' % (n, pos))

    def test_deletion_dash_alt(self):
        pos, n = 500, 4
        ref = self.genome[pos - 1:pos - 1 + n]
        start, end, seq = self._alt_window(pos, ref, '-')
        self.assertEqual(len(seq), end - start)
        self.assertIn(seq, self._alt_genome(pos, ref, ''))


if __name__ == '__main__':
    unittest.main()
