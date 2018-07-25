import os
import unittest

import numpy as np

from selene_sdk.targets import GenomicFeatures
from selene_sdk.targets.genomic_features import _any_positive_rows, \
    _is_positive_row, _get_feature_data


class TestGenomicFeatures(unittest.TestCase):

    def setUp(self):
        self.features = [
            "CTCF", "eGFP-FOS", "GABP", "Pbx3", "Pol2", "TBP"
        ]
        self.feature_index_map = {
            "CTCF": 0, "eGFP-FOS": 1, "GABP": 2, "Pbx3": 3, "Pol2": 4, "TBP": 5
        }

        self.n_features = len(self.features)

        # CTCF only, between 16110 and 16239
        self.rows_example1 =  \
            [["1", "16110", "16190", "CTCF"],  # len 70
             ["1", "16128", "16158", "CTCF"],  # len 30
             ["1", "16149", "16239", "CTCF"]]  # len 90

        # CTCF only, between 91128 and 91358
        self.rows_example2 =  \
            [["2", "91128", "91358", "CTCF"],  # len 200
             ["2", "91130", "91239", "CTCF"],  # len 109
             ["2", "91156", "91310", "CTCF"]]  # len 154

        # multiple features, between 8533 and 9049
        self.rows_example3 = \
            [["chr3", "8533", "8817", "eGFP-FOS"],  # len 284
             ["chr3", "8541", "8651", "GABP"],      # len 110
             ["chr3", "8574", "8629", "Pol2"],      # len 145
             ["chr3", "8619", "9049", "CTCF"],       # len 430
             ["chr3", "8620", "8680", "TBP"],       # len 60
             ["chr3", "8645", "8720", "TBP"]]       # len 75

    def get_feature_rows(self, chrom, start, end):
        """This function disregards (`start`, `end`) input
        """
        if chrom is None:
            return None

        if chrom == "1":
            return self.rows_example1
        elif chrom == "2":
            return self.rows_example2
        elif chrom == "3":
            return self.rows_example3
        else:
            return []

    ############################################
    # Correctness tests for `_is_positive_row`
    ############################################

    def test__is_positive_row_false(self):
        query_start, query_end = (16150, 16351)  # len 201
        feat_start, feat_end = (16110, 16190)    # len 80
        threshold = 0.50
        self.assertFalse(
            _is_positive_row(
                query_start, query_end, feat_start, feat_end, threshold))

    def test__is_positive_row_true_eq_threshold(self):
        query_start, query_end = (16110, 16309)  # len 199
        feat_start, feat_end = (16110, 16190)    # len 80
        threshold = 0.40
        self.assertTrue(
            _is_positive_row(
                query_start, query_end, feat_start, feat_end, threshold))

    def test__is_positive_row_true_gt_threshold(self):
        query_start, query_end = (16110, 16311)  # len 201
        feat_start, feat_end = (16110, 16290)    # len 170
        threshold = 0.80
        self.assertTrue(
            _is_positive_row(
                query_start, query_end, feat_start, feat_end, threshold))

    ############################################
    # Correctness tests for `_any_positive_rows`
    ############################################

    def test__any_positive_rows_none_rows(self):
        rows = None
        query_start, query_end = (10, 100)
        threshold = {k: 0.50 for k in self.features}
        self.assertFalse(
            _any_positive_rows(rows, query_start, query_end, threshold))

    def test__any_positive_rows_empty_rows(self):
        rows = []
        query_start, query_end = (10, 100)
        threshold = {k: 0.50 for k in self.features}
        self.assertFalse(
            _any_positive_rows(rows, query_start, query_end, threshold))

    def test__any_positive_rows_false(self):
        rows = self.rows_example1
        query_start, query_end = (16150, 16351)
        threshold = {k: 0.50 for k in self.features}
        self.assertFalse(
            _any_positive_rows(rows, query_start, query_end, threshold))

    def test__any_positive_rows_true(self):
        rows = self.rows_example1
        query_start, query_end = (16150, 16351)
        threshold = {k: 0.40 for k in self.features}
        self.assertTrue(
            _any_positive_rows(rows, query_start, query_end, threshold))

    ############################################
    # Correctness tests for `_get_feature_data`
    ############################################

    def test__get_feature_data_none_rows(self):
        query_chrom, query_start, query_end = (None, 10, 211)
        threshold = np.array([0.50] * self.n_features).astype(np.float32)

        expected_encoding = [0, 0, 0, 0, 0, 0]
        observed_encoding = _get_feature_data(
            query_chrom, query_start, query_end, threshold,
            self.feature_index_map, self.get_feature_rows)

        self.assertSequenceEqual(
            observed_encoding.tolist(), expected_encoding)

    def test__get_feature_data_empty_rows(self):
        query_chrom, query_start, query_end = ("7", 10, 211)
        threshold = np.array([0.50] * self.n_features).astype(np.float32)

        expected_encoding = [0, 0, 0, 0, 0, 0]
        observed_encoding = _get_feature_data(
            query_chrom, query_start, query_end, threshold,
            self.feature_index_map, self.get_feature_rows)

        self.assertSequenceEqual(
            observed_encoding.tolist(), expected_encoding)

    def test__get_feature_data_single_feat_positive(self):
        query_chrom, query_start, query_end = ("1", 16100, 16350)
        threshold = np.array([0.50] * self.n_features).astype(np.float32)

        expected_encoding = [1, 0, 0, 0, 0, 0]
        observed_encoding = _get_feature_data(
            query_chrom, query_start, query_end, threshold,
            self.feature_index_map, self.get_feature_rows)

        self.assertSequenceEqual(
            observed_encoding.tolist(), expected_encoding)

    def test__get_feature_data_no_feat_positive(self):
        query_chrom, query_start, query_end = ("2", 91027, 91228)
        threshold = np.array([0.51] * self.n_features).astype(np.float32)

        expected_encoding = [0, 0, 0, 0, 0, 0]
        observed_encoding = _get_feature_data(
            query_chrom, query_start, query_end, threshold,
            self.feature_index_map, self.get_feature_rows)

        self.assertSequenceEqual(
            observed_encoding.tolist(), expected_encoding)

    def test__get_feature_data_multiple_feats_positive(self):
        query_chrom, query_start, query_end = ("3", 8619, 8719)
        threshold = np.array([0.50] * self.n_features).astype(np.float32)

        expected_encoding = [1, 1, 0, 0, 0, 1]
        observed_encoding = _get_feature_data(
            query_chrom, query_start, query_end, threshold,
            self.feature_index_map, self.get_feature_rows)

        self.assertSequenceEqual(
            observed_encoding.tolist(), expected_encoding)

    def test__get_feature_data_different_thresholds(self):
        query_chrom, query_start, query_end = ("3", 8619, 8719)
        threshold = np.array([0.50, 0.0, 0.0, 0.0, 0.0, 1.0]).astype(np.float32)

        expected_encoding = [1, 1, 1, 0, 1, 0]
        observed_encoding = _get_feature_data(
            query_chrom, query_start, query_end, threshold,
            self.feature_index_map, self.get_feature_rows)

        self.assertSequenceEqual(
            observed_encoding.tolist(), expected_encoding)

    ############################################
    # GenomicFeatures integration tests
    ############################################

    def test_GenomicFeatures_single_threshold(self):
        data_path = os.path.join(
            "selene_sdk", "targets", "tests",
            "files", "sorted_aggregate.bed.gz")
        query_features = GenomicFeatures(
            data_path, self.features, 0.50)
        self.assertDictEqual(
            query_features.feature_thresholds,
            {k: 0.50 for k in self.features})
        self.assertSequenceEqual(
            query_features._feature_thresholds_vec.tolist(),
            [0.50] * self.n_features)

    def test_GenomicFeatures_diff_thresholds(self):
        data_path = os.path.join(
            "selene_sdk", "targets", "tests",
            "files", "sorted_aggregate.bed.gz")
        query_features = GenomicFeatures(
            data_path, self.features,
            {"default": 0.50, "CTCF": 0.0, "Pol2": 0.15})
        self.assertEqual(
            query_features.feature_thresholds,
            {"CTCF": 0.0, "eGFP-FOS": 0.50,
             "GABP": 0.50, "Pbx3": 0.50,
             "Pol2": 0.15, "TBP": 0.50})
        np.testing.assert_almost_equal(
            query_features._feature_thresholds_vec.tolist(),
            [0.0, 0.50, 0.50, 0.50, 0.15, 0.50])

    def test_GenomicFeatures_lambda_thresholds(self):
        def _feature_thresholds(f):
            if f == "Pbx3":
                return 0.30
            elif f == "CTCF":
                return 0.40
            else:
                return 0.50

        data_path = os.path.join(
            "selene_sdk", "targets", "tests",
            "files", "sorted_aggregate.bed.gz")
        query_features = GenomicFeatures(
            data_path, self.features, _feature_thresholds)
        self.assertEqual(
            query_features.feature_thresholds,
            {"CTCF": 0.40, "eGFP-FOS": 0.50,
             "GABP": 0.50, "Pbx3": 0.30,
             "Pol2": 0.50, "TBP": 0.50})
        np.testing.assert_almost_equal(
            query_features._feature_thresholds_vec.tolist(),
            [0.40, 0.50, 0.50, 0.30, 0.50, 0.50])

if __name__ == "__main__":
    unittest.main()
