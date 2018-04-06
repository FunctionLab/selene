from .base_sampler import BaseSampler
from ..sequences import Genome
from ..targets import GenomicFeatures


class OnlineSampler(BaseSampler):

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
        # @TODO: this could be more flexible. Sequence len and center bin
        # len do not necessarily need to be odd numbers...
        if sequence_length % 2 == 0 or center_bin_to_predict % 2 == 0:
            raise ValueError(
                "Both the sequence length and the center bin length "
                "should be odd numbers. Sequence length was {0} and "
                "bin length was {1}.".format(
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
            # @TODO: make sure that isinstance works in this
            # situation
            if isinstance(validation_holdout, (list,)) and \
                    isinstance(test_holdout, (list,)):
            #if type(validation_holdout) == type(list()) and \
            #        type(test_holdout) == type(list()):
                print("both are type list")
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
            #if type(validation_holdout) == type(list()):
                print("validation holdout is type list")
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
        self.bin_radius = int((center_bin_to_predict - 1) / 2)

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

    def sample(self, batch_size):
        raise NotImplementedError

    def get_data_and_targets(self, mode, batch_size, n_samples):
        raise NotImplementedError

    def get_validation_set(self, batch_size, n_samples=None):
        raise NotImplementedError
