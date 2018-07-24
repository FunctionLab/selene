"""
This module provides the `ISMResult` class, which stores results for
an *in silico* mutagenesis experiment.

"""
import numpy as np
import pandas as pd

from ..sequences import Genome


class ISMResult(object):
    """
    An object storing the results of an *in silico* mutagenesis
    experiment.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        The data frame with the results from the *in silico*
        mutagenesis experiments.

    sequence_type : class, optional
        Default is `selene_sdk.sequences.Genome`. The type of sequence
        that the *in silico* mutagenesis results are associated
        with. This should generally be a subclass of
        `selene_sdk.sequences.Sequence`

    Raises
    ------
    ValueError
        If the input data frame contains a base not included in the
        alphabet of `sequence_type`.

    Exception
        If multiple reference positions are specified in the input
        data frame.

    Exception
        If the input data does not contain scores for every mutation
        at every position.

    """
    def __init__(self, data_frame, sequence_type=Genome):
        """
        Constructs a new `selene_sdk.interpret.ISMResult` object.

        """
        # Construct the reference sequence.
        alpha = set(sequence_type.BASES_ARR)
        ref_seq = [""] * (int(data_frame["pos"].iloc[-1]) + 1)
        seen = set()
        for row_idx, row in data_frame.iterrows():
            # Skip the reference value
            if not (row_idx == 0 and row["alt"] == "NA" and
                    row["ref"] == "NA"):
                cur_ref = row["ref"]
                if cur_ref not in alpha and cur_ref != sequence_type.UNK_BASE:
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
        """
        The reference sequence that the *in silico* mutagenesis
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
        """
        The type of underlying sequence. This should generally be a
        subclass of `selene_sdk.sequences.Sequence`.

        Returns
        -------
        class
            The type of sequence that the *in silico* mutagenesis was
            performed on.

        """
        return self._sequence_type

    def get_score_matrix_for(self, feature, reference_mask=None,
                             dtype=np.float64):
        """
        Extracts a feature from the *in silico* mutagenesis results
        as a matrix, where the reference base positions hold the value
        for the reference prediction, and alternative positions hold the
        results for making a one-base change from the reference base to
        the specified alternative base.

        Parameters
        ----------
        feature : str
            The name of the feature to extract as a matrix.

        reference_mask : float or None, optional
            Default is `None`. A value to mask the reference entries
            with. If left as `None`, then no masking will be performed
            on the reference positions.

        dtype : numpy.dtype, optional
            Default is `numpy.float64`. The data type to use for the
            returned matrix.

        Returns
        -------
        numpy.ndarray
            A :math:`L \\times N` shaped array (where :math:`L` is the
            sequence length, and :math:`N` is the size of the alphabet
            of `sequence_type`) that holds the results from the
            *in silico* mutagenesis experiment for the specified
            feature. The elements will be of type `dtype`.

        Raises
        ------
        ValueError
            If the input data frame contains a base not included in the
            alphabet of `sequence_type`.
        """
        if reference_mask is not None:
            reference_mask = dtype(reference_mask)
        ret = self._sequence_type.sequence_to_encoding(
            self._reference_sequence).astype(dtype=dtype)
        ret[ret < 0] = 0.  # Set N's to zero to avoid spurious masking.
        alpha = set(self._sequence_type.BASES_ARR)
        for row_idx, row in self._data_frame.iterrows():
            # Extract reference value in first row.
            if row_idx == 0:
                if row["alt"] == "NA" and row["ref"] == "NA":
                    if reference_mask is None:
                        reference_mask = dtype(row[feature])
                    ret *= reference_mask
                    continue
                else:
                    if reference_mask is None:
                        reference_mask = 0.
                    ret *= reference_mask
            base = row["alt"]
            i = int(row["pos"])
            if base not in alpha:
                if base != self._sequence_type.UNK_BASE:
                    raise ValueError(
                        "Found character \'{0}\' from outside current alphabet"
                        " on row {1}.".format(base, row_idx))
            else:
                ret[i, self._sequence_type.BASE_TO_INDEX[base]] = dtype(
                    row[feature])
        return ret

    @staticmethod
    def from_file(input_path, sequence_type=Genome):
        """
        Loads a `selene_sdk.interpret.ISMResult` from a `pandas.DataFrame`
        stored in a file of comma separated values (CSV).

        Parameters
        ----------
        input_path : str
            A path to the file of comma separated input values.

        sequence_type : class, optional
            Default is `selene_sdk.sequences.Genome`. The type of sequence
            that the *in silico* mutagenesis results are associated
            with. This should generally be a subclass of
            `selene_sdk.sequences.Sequence`.

        Returns
        -------
        selene_sdk.interpret.ISMResult
            The *in silico* mutagenesis results that were stored in the
            specified input file.

        """
        return ISMResult(pd.read_csv(input_path, sep="\t", header=0,
                                     dtype=str, na_values=None,
                                     keep_default_na=False),
                         sequence_type=sequence_type)
