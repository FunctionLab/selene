import h5py
import numpy as np
import scipy.io

from .sampler import Sampler

class MatFileSampler(Sampler):

    def __init__(self,
                 training_data,
                 validation_data,
                 test_data=None,
                 random_seed=436,
                 mode="train"):
        super(MatFileSampler, self).__init__(
            random_seed=random_seed
        )
        if test_data:
            self.modes.append("test")

        self.sample_from_mode = {}
        self.randcache = {}
        for mode in self.modes:
            self.sample_from_mode[mode] = {
                "data": None,
                "indices": [],
                "sample_next": 0
            }

        train_data_mats, n_train = self._load_mat_file(training_data)
        self.sample_from_mode["train"]["data"] = train_data_mats
        self.sample_from_mode["train"]["indices"] = np.arange(n_train).tolist()

        valid_data_mats, n_validate = self._load_mat_file(validation_data)
        self.sample_from_mode["validate"]["data"] = valid_data_mats
        self.sample_from_mode["validate"]["indices"] = \
            np.arange(n_validate).tolist()

        if test_data:
            test_data_mats, n_test = self._load_mat_file(test_data)
            self.sample_from_mode["test"]["data"] = test_data_mats
            self.sample_from_mode["test"]["indices"] = \
                np.arange(n_test).tolist()

        self.sequence_length = train_data_mats[0].shape[1]
        self.n_features = train_data_mats[1].shape[1]

        for mode, mode_info in self.sample_from_mode.items():
            np.random.shuffle(mode_info["indices"])

    def _load_mat_file(self, mode, mat_file,
                       data_key_suffix="xdata",
                       target_key_suffix="data"):
        """Assumes the keys are "<mode>xdata" and "<mode>data",
        e.g. "trainxdata" being the matrix of one-hot encoded input sequences
        and "traindata" being the matrix of corresponding genomic features.
        """
        seqs_key = "{0}{1}".format(mode, data_key_suffix)
        tgts_key = "{0}{1}".format(mode, target_key_suffix)
        try:  # see if it will load using scipy first
            mat = scipy.io.loadmat(mat_file)
            n_samples, _ = mat[tgts_key].shape
            return (mat[seqs_key], mat[tgts_key]), n_samples
        except NotImplementedError:
            mat_fh = h5py.File(mat_file, 'r')
            sequence_data = mat[seqs_key][()].T
            tgts = mat[tgts_key][()].T
            n_samples, _ = tgts.shape
            mat_fh.close()
            return (sequence_data, tgts, n_samples)

    def sample(self, batch_size=1):
        mode = self.mode
        indices = self.sample_from_mode[mode]["indices"]
        input_data, target_data = self.sample_from_mode[mode]["data"]

        sample_up_to = self.sample_from_mode["sample_next"] + batch_size
        use_indices = None
        if sample_up_to > len(indices):
            np.random.shuffle(self.sample_from_mode[mode]["indices"])
            self.sample_from_mode["sample_next"] = 0
            use_indices = indices[:batch_size]
        else:
            sample_next = self.sample_from_mode["sample_next"] + batch_size
            use_indices = indices[sample_next:sample_next + batch_size]
        self.sample_from_mode["sample_next"] += batch_size
        sequences = np.transpose(
            input_data[use_indices, :, :], (0, 2, 1)).astype(float)
        targets = target_data[use_indices, :].astype(float)
        return (sequences, targets)

    def get_data_and_targets(self, mode, batch_size, n_samples):
        indices = self.sample_from_mode[mode]["indices"]
        input_data, target_data = self.sample_from_mode[mode]["data"]

        sequences_and_targets = []
        targets_mat = []
        for _ in range(0, n_samples, self.batch_size):
            sample_next = self.sample_from_mode["sample_next"] + batch_size
            use_indices = None
            if sample_next > len(indices):
                np.random.shuffle(self.sample_from_mode[mode]["indices"])
                self.sample_from_mode["sample_next"] = 0
                use_indices = indices[:batch_size]
            else:
                use_indices = indices[sample_next:sample_next + batch_size]
            self.sample_from_mode["sample_next"] += batch_size
            sequences = np.transpose(
                input_data[use_indices, :, :], (0, 2, 1)).astype(float)
            targets = target_data[use_indices, :].astype(float)

            sequences_and_targets.append((sequences, targets))
            targets_mat.append(targets)
        targets_mat = np.vstack(targets_mat)
        return sequences_and_targets, targets_mat

    def get_validation_set(self, batch_size, n_samples=None):
        if not n_samples:
            n_samples = len(self.sample_from_mode["validate"].indices)
        return self.get_data_and_targets("validate", batch_size, n_samples)
