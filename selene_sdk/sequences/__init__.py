"""
This module provides the types for representing biological sequences.
"""
from .sequence import Sequence
from .sequence import sequence_to_encoding
from .sequence import encoding_to_sequence
from .sequence import get_reverse_encoding
from .genome import Genome
from .proteome import Proteome

__all__ = ["Sequence", "Genome", "Proteome", "sequence_to_encoding",
           "encoding_to_sequence", "get_reverse_encoding"]
