"""
This module provides the classes and methods for prediction handlers,
which generally are used for logging and saving outputs from models.
"""
from .handler import PredictionsHandler
from .handler import write_to_file
from .absolute_diff_score_handler import AbsDiffScoreHandler
from .diff_score_handler import DiffScoreHandler
from .logit_score_handler import LogitScoreHandler
from .write_predictions_handler import WritePredictionsHandler
from .write_ref_alt_handler import WriteRefAltHandler

__all__ = ["PredictionsHandler", "write_to_file",
           "AbsDiffScoreHandler", "DiffScoreHandler",
           "LogitScoreHandler", "WritePredictionsHandler",
           "WriteRefAltHandler"]
