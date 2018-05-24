"""This module provides the `ISMResult` class, which stores results for
an *in silico* mutagenesis experiment.

"""
import numpy as np
import pandas as pd

from selene.sequences import Genome


class ISMResult(object):
    """An object storing the results of an *in silico* mutagenesis
     experiment.

    """
    def __init__(self, data_frame, sequence_type=Genome):
        """Constructs a new `ISMResult` object.

        Parameters
        ----------
        data_frame : pandas.DataFrame
            The data frame with the results from the *in silico*
            mutagenesis experiments.

        sequence_type : class, optional
            Default is `selene.sequences.Genome`. The type of sequence
            that the *in silico* mutagenesis results are associated
            with. This should generally be a subclass of
            `selene.sequences.Sequence`

        """
        # Construct the reference sequence.
        alpha = set(sequence_type.BASES_ARR)
        ref_seq = [""] * (int(data_frame["pos"].max()) + 1)
        seen = set()
        for row_idx, row in data_frame.iterrows():
            if row_idx != 0:  # Skip the reference value
                cur_ref = row["ref"]
                if cur_ref not in alpha:
                    raise ValueError(
                        "Found character \'{0}\' from outside current alphabet"
                        " on row {1}.".format(cur_ref, row_idx))
                i = int(row["pos"])
                seen.add(i)
                if ref_seq[i] != "":
                    if ref_seq[i] != cur_ref:
                        raise Exception(
                            "Found 2 different letters for reference \'{0}\'"
                            " and \'{1}\' on row {2}.".format(ref_seq[i],
                                                              cur_ref,
                                                              row_idx))
                else:
                    ref_seq[i] = cur_ref
        if len(seen) != len(ref_seq):
            raise Exception(
                "Expected characters for {0} positions, but only found {1} of "
                "them.".format(len(ref_seq), len(seen)))
        ref_seq = "".join(ref_seq)
        self._reference_sequence = ref_seq
        self._data_frame = data_frame
        self._sequence_type = sequence_type

    @property
    def reference_sequence(self):
        """The reference sequence that the *in silico* mutagenesis
        experiment was performed on.

        Returns
        -------
        str
            The reference sequence (i.e. non-mutated input) as a
            string of characters.

        """
        return self._reference_sequence

    @property
    def sequence_type(self):
        """The type of underlying sequence.

        Returns
        -------
        class
            The type of sequence that the *in silico* mutagenesis was
            performed on.

        """
        return self._sequence_type

    def get_score_matrix_for(self, feature, reference_mask=None,
                             dtype=np.float32):
        """Extracts a feature from the *in silico* mutagenesis results
        as a matrix, where the reference base positions hold the value
        for the reference prediction, and alternative positions hold the
        results for making a one-base change from the reference base to
        the specified alternative base.

        Parameters
        ----------
        feature : str
            The name of the feature to extract as a matrix.

        reference_mask : float, optional
            Default is `None`. A value to mask the reference entries
            with. If left as `None`, then no masking will be performed
            on the reference positions.

        dtype : `numpy.dtype`, optional
            Default is `numpy.float32`. The data type to use for the returned
            matrix.

        Returns
        -------
        numpy.ndarray, dtype=`dtype`
            A LxN shaped array (where L is the sequence length, and N
            is the size of the alphabet) that holds the results from the
            *in silico* mutagenesis experiment for the specified feature.

        """
        ret = self._sequence_type.sequence_to_encoding(
            self._reference_sequence).astype(dtype=dtype)
        alpha = set(self._sequence_type.BASES_ARR)
        for row_idx, row in self._data_frame.iterrows():
            if row_idx == 0:  # Extract reference value in first row.
                if reference_mask is None:
                    ret *= row[feature]
                else:
                    ret *= reference_mask
            base = row["alt"]
            i = int(row["pos"])
            if base not in alpha:
                raise ValueError(
                    "Found character \'{0}\' from outside current alphabet"
                    " on row {1}.".format(base, row_idx))
            ret[i, self._sequence_type.BASE_TO_INDEX[base]] = row[feature]
        return ret

    @staticmethod
    def from_file(input_path, sequence_type=Genome):
        """Loads an `ISMResult` from a `pandas.DataFrame` stored in a
        file.

        Parameters
        ----------
        input_path : str
            A path to the input file.

        sequence_type : class, optional
            Default is `selene.sequences.Genome`. The type of sequence
            that the *in silico* mutagenesis results are associated
            with.

        Returns
        -------
        selene.interpret.ISMResult
            The results that were stored in the file.

        """
        return ISMResult(pd.read_csv(input_path, sep="\t", header=0),
                         sequence_type=sequence_type)
