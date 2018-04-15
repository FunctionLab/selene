"""
Provides the base class and methods for target feature classes.
"""
from abc import ABCMeta
from abc import abstractmethod


class Target(metaclass=ABCMeta):
    """
    Base class for target features.
    """
    @abstractmethod
    def get_feature_data(self, *args, **kwargs):
        """
        Retrieve the feature data for some coordinate.
        """
        raise NotImplementedError
