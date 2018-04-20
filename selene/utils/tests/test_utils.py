import os
import unittest
from copy import deepcopy
import numpy as np

from selene.utils import  get_matrix_from_in_silico_mutagenesis_results


class TestInSilicoMutagenesisResultUtils(unittest.TestCase):
    """
    Tests the various utils for handling results from in silico mutagenesis experiments.
    """
    def test__get_matrix_from_in_silico_mutagenesis_results_no_ref_pred(self):
        mut_preds = np.array([1, 2, 3, 4, 5, 6])
        mut_encs = []
        ref_enc = np.array([[1., 1.],
                            [0., 0.],
                            [0., 0.],
                            [0., 0.]])
        for pos in range(ref_enc.shape[1]):
            for base in range(1, ref_enc.shape[0]):
                x = deepcopy(ref_enc)
                x[:, pos] = 0.
                x[base, pos] = 1.
                mut_encs.append(x)
        mut_encs = np.stack(mut_encs)
        expected = np.array([[0., 0.],
                             [1., 4.],
                             [2., 5.],
                             [3., 6.]])
        res = get_matrix_from_in_silico_mutagenesis_results(mut_encs, mut_preds, ref_enc)
        self.assertTrue(np.array_equal(expected, res))

    def test__get_matrix_from_in_silico_mutagenesis_results_with_ref_pred(self):
        mut_preds = np.array([1, 2, 3, 4, 5, 6])
        mut_encs = []
        ref_pred = 7.
        ref_enc = np.array([[1., 1.],
                            [0., 0.],
                            [0., 0.],
                            [0., 0.]])
        for pos in range(ref_enc.shape[1]):
            for base in range(1, ref_enc.shape[0]):
                x = deepcopy(ref_enc)
                x[:, pos] = 0.
                x[base, pos] = 1.
                mut_encs.append(x)
        mut_encs = np.stack(mut_encs)
        expected = np.array([[7., 7.],
                             [1., 4.],
                             [2., 5.],
                             [3., 6.]])
        res = get_matrix_from_in_silico_mutagenesis_results(mut_encs, mut_preds, ref_enc, ref_pred)
        self.assertTrue(np.array_equal(expected, res))


if __name__ == "__main__":
    unittest.main()
