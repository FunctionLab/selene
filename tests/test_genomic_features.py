import os
import unittest

from proteus import GenomicFeatures


class TestGenomicFeatures(unittest.TestCase):

    def setUp(self):
        GENOMIC_FEATURES_DIR = "data/test_files/ChIP_CTCF_6feats"

        features_file = os.path.join(
            GENOMIC_FEATURES_DIR,
            "distinct_features.txt")
        features_list = None
        with open(features_file, 'r') as features_fh:
            features_list = [f.strip() for f in features_fh.readlines()]
        features_fh.close()

        self.query_features = GenomicFeatures(
            os.path.join(GENOMIC_FEATURES_DIR, "sorted_aggregate.bed.gz"),
            features_list)

    def test_feature_index_map_attribute(self):
        """Test that `feature_index_map` contains the expected values.
            str (feature) -> int (index) dict
        """
        observed = self.query_features.feature_index_map
        expected = {"CTCF": 0,
                    "eGFP-FOS": 1,
                    "GABP": 2,
                    "Pbx3": 3,
                    "Pol2": 4,
                    "TBP": 5}
        msg = ("GenomicFeatures `feature_index_map` expected to be {0} "
               "but was {1}").format(expected, observed)
        self.assertEqual(observed, expected, msg)

    def test_index_feature_map_attribute(self):
        """Test that `index_feature_map` contains the expected values.
            int (index) -> str (feature) dict
        """
        observed = self.query_features.index_feature_map
        expected = {0: "CTCF",
                    1: "eGFP-FOS",
                    2: "GABP",
                    3: "Pbx3",
                    4: "Pol2",
                    5: "TBP"}
        msg = ("GenomicFeatures `index_feature_map` expected to be {0} "
               "but was {1}").format(expected, observed)
        self.assertEqual(observed, expected, msg)

    def test_is_positive(self):
        """Test that `is_positive` correctly evaluates whether coordinates
        are a 'positive' or 'negative' bin (a bin that has or does not have
        at least one genomic feature, respectively)
        """
        chrom = "chr1"
        thresholds = [0.40, 0.50, 0.80]
        coordinates_expected_labels = {
            (0, 201): [False, False, False],
            (29200, 29570): [True, True, False],
            (29200, 30000): [False, False, False],
            (29200, 29600): [True, True, False],
            (29500, 29700): [True, False, False],
            (91215, 91416): [True, True, True],
            (91440, 91641): [True, True, False],
            (91530, 91731): [False, False, False]
        }
        for (start, end), labels in coordinates_expected_labels.items():
            for index, threshold in enumerate(thresholds):
                observed = self.query_features.is_positive(
                    chrom, start, end, threshold)
                expected = labels[index]
                msg = ("Expected ({0}, {1}, {2}) under threshold {3} "
                       "to return `is_positive`: {4}.").format(
                            chrom, start, end, threshold, expected)
                self.assertEqual(observed, expected, msg)

    def test_get_feature_data(self):
        """Test that `get_feature_data` retrieves the proper one-hot encoding
        of the features for the queried coordinates
        """
        thresholds = [0.35, 0.50, 0.80]
        coordinates_expected_labels = {
            # chr1	713950	714246
            ("chr1", 713950, 714544): [
                [1, 0, 1, 0, 1, 1],
                [1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 1, 0]
            ],
            # chr10	63348553	63348837
            ("chr10", 63348553, 63349171): [
                [0, 1, 1, 0, 1, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ],
            # chr10	63348553	63348837
            ("chr10", 63348553, 63348953): [
                [0, 1, 1, 0, 1, 0],
                [0, 1, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0]
            ]
        }
        for (chrom, start, end), OHE_features in \
                coordinates_expected_labels.items():
            for index, threshold in enumerate(thresholds):
                observed = self.query_features.get_feature_data(
                    chrom, start, end, strand='+', threshold=threshold)
                expected = OHE_features[index]
                msg = ("Expected ({0}, {1}, {2}) under threshold {3} "
                       "to return: {4}.").format(
                            chrom, start, end, threshold, expected)
                self.assertSequenceEqual(observed.tolist(), expected, msg)


if __name__ == "__main__":
    unittest.main()
