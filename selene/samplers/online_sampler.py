from abc import ABCMeta

from .sampler import Sampler
from ..sequences import Genome
from ..targets import GenomicFeatures


class OnlineSampler(Sampler, metaclass=ABCMeta):

    STRAND_SIDES = ('+', '-')

    def __init__(self,
                 genome,
                 query_feature_data,
                 distinct_features,
                 random_seed=436,
                 validation_holdout=['6', '7'],
                 test_holdout=['8', '9'],
                 sequence_length=1001,
                 center_bin_to_predict=201,
                 feature_thresholds=0.5,
                 mode="train"):
        super(OnlineSampler, self).__init__(
            random_seed=random_seed
        )

        if (sequence_length + center_bin_to_predict) % 2 != 0:
            raise ValueError(
                "Sequence length of {0} with a center bin length of {1} "
                "is invalid. These 2 inputs should both be odd or both be "
                "even.".format(
                    sequence_length, center_bin_to_predict))

        surrounding_sequence_length = \
            sequence_length - center_bin_to_predict
        if surrounding_sequence_length < 0:
            raise ValueError(
                "Sequence length of {0} is less than the center bin "
                "length of {1}.".format(
                    sequence_length, center_bin_to_predict))

        # specifying a test holdout partition is optional
        if test_holdout:
            self.modes.append("test")
            if isinstance(validation_holdout, (list,)) and \
                    isinstance(test_holdout, (list,)):
                self.validation_holdout = [
                    str(c) for c in validation_holdout]
                self.test_holdout = [str(c) for c in test_holdout]
                self._holdout_type = "chromosome"
            elif isinstance(validation_holdout, float) and \
                    isinstance(test_holdout, float):
                self.validation_holdout = validation_holdout
                self.test_holdout = test_holdout
                self._holdout_type = "proportion"
            else:
                raise ValueError(
                    "Validation holdout and test holdout must have the "
                    "same type (list or float) but validation was "
                    "type {0} and test was type {1}".format(
                        type(validation_holdout), type(test_holdout)))
        else:
            self.test_holdout = None
            if isinstance(validation_holdout, (list,)):
                self.validation_holdout = [
                    str(c) for c in validation_holdout]
            else:
                self.validation_holdout = validation_holdout

        if mode not in self.modes:
            raise ValueError(
                "Mode must be one of {0}. Input was '{1}'.".format(
                    self.modes, mode))
        self.mode = mode

        self.surrounding_sequence_radius = int(
            surrounding_sequence_length / 2)
        self.sequence_length = sequence_length
        self.bin_radius = int(center_bin_to_predict / 2)
        self._start_radius = self.bin_radius
        if center_bin_to_predict % 2 == 0:
            self._end_radius = self.bin_radius
        else:
            self._end_radius = self.bin_radius + 1

        self.genome = Genome(genome)

        self._features = []
        with open(distinct_features, 'r') as file_handle:
            for line in file_handle:
                self._features.append(line.strip())
        self.n_features = len(self._features)

        self.query_feature_data = GenomicFeatures(
            query_feature_data, self._features,
            feature_thresholds=feature_thresholds)

    def get_feature_from_index(self, feature_index):
        """Returns the feature corresponding to an index in the feature
        vector.

        Parameters
        ----------
        feature_index : int

        Returns
        -------
        str
        """
        return self.query_feature_data.index_feature_map[feature_index]

    def get_sequence_from_encoding(self, encoding):
        return self.genome.encoding_to_sequence(encoding)
