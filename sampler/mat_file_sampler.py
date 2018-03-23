from base_sampler import BaseSampler

class MatFileSampler(BaseSampler):

    def __init__(self, train_mat, valid_mat, test_mat=None):
        self.train_mat = train_mat

    def sample(self, batch_size=1):
        raise NotImplementedError

    def get_data_and_targets(self, mode, batch_size, n_samples):
        raise NotImplementedError

    def get_validation_set(self, batch_size, n_samples=None):
        raise NotImplementedError
