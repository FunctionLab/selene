"""
This module provides the classes and methods for prediction handlers,
which generally are used for logging and saving outputs from models.
"""
from .handler import PredictionsHandler
from .handler import write_to_file
from .diff_score_handler import DiffScoreHandler
from .logit_score_handler import LogitScoreHandler
from .write_predictions_handler import WritePredictionsHandler
from .write_ref_alt_handler import WriteRefAltHandler

__all__ = ["PredictionsHandler", "write_to_file", "DiffScoreHandler",
           "LogitScoreHandler", "WritePredictionsHandler",
           "WriteRefAltHandler"]
