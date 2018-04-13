"""
This class is the abstract base class for biological sequences to load training examples from.
"""
from abc import ABCMeta
from abc import abstractmethod


class Sequence(metaclass=ABCMeta):
    """
    The base class for biological sequence classes.
    """
    @abstractmethod
    def sequence_in_bounds(self, *args, **kwargs):
        """
        Checks if given coordinates are in the sequence.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sequence_from_coords(self, *args, **kwargs):
        """
        Extracts a string of sequence at the given coordinates.
        """
        raise NotImplementedError

    @abstractmethod
    def get_encoding_from_coords(self, *args, **kwargs):
        """
        Extracts the numerical encoding for a sequence occuring at the given coordinates.
        """
        raise NotImplementedError

    @abstractmethod
    def sequence_to_encoding(self, sequence):
        """
        Transforms a biological sequence into a numerical representation.
        """
        raise NotImplementedError

    @abstractmethod
    def encoding_to_sequence(self, encoding):
        """
        Transforms the input numerical representation of a biological sequence into a string representation.
        """
        raise NotImplementedError
