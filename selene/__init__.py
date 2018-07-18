"""
This is the main module for Selene.
"""
__all__ = ["sequences", "targets", "samplers", "utils",
           "predict", "interpret", "__version__"]
name = "selene"

from .version import __version__
from .evaluate_model import EvaluateModel
from .train_model import TrainModel
from .non_strand_specific_module import NonStrandSpecific
