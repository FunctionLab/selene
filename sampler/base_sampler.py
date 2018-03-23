"""Sampler interface"""

class BaseSampler(object):

    def __init__(self):
        print("base class attributes here")

    def sample(self, batch_size=1):
        raise NotImplementedError

    def get_data_and_targets(self, mode, batch_size, n_samples):
        raise NotImplementedError

    def get_validation_set(self, batch_size, n_samples=None):
        raise NotImplementedError
