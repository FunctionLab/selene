"""
This module provides the `Target` base class, which defines the
interface for target feature classes. Target feature classes are classes
which define a way to access a "target feature" such as a label or
annotation on an input sequence.

"""
from abc import ABCMeta
from abc import abstractmethod


class Target(metaclass=ABCMeta):
    """
    The abstract base class for all target feature classes.
    Target features classes are classes which define a way to access a
    "target feature" such as a label or annotation on an input sequence.

    """
    @abstractmethod
    def get_feature_data(self, *args, **kwargs):
        """
        Retrieve the feature data for some coordinate.

        """
        raise NotImplementedError()
