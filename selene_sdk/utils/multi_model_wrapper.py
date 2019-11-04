"""
This module specifies the MultiModelWrapper class, currently intended for
use through Selene's API (as opposed to the CLI).

Loads multiple models and concatenates their outputs.
"""
import torch
import torch.nn as nn


class MultiModelWrapper(nn.Module):

    def __init__(self, sub_models, concat_dim=1):
        """
        The multi-model wrapper class can be used to concatenate the
        outputs of multiple models along a pre-specified axis. The wrapper
        can be used to load and run multiple trained models during prediction
        functions. This class should not be used for training. We also have
        not yet provided support for using this class through the CLI.

        This class can be used to initialize
        `selene_sdk.predict.AnalyzeSequences` with a corresponding list of
        `trained_model_path`s. Please ensure the ordering of the two lists
        (`sub_models` here and `trained_model_path` in AnalyzeSequences)
        match.

        Parameters
        ----------
        sub_models : list(torch.nn.Module)
            The 'sub-models' that are used in this multi-model wrapper class.
        concat_dim : int, optional
            Default is 1. The dimension along which to concatenate the models'
            predictions.
        """
        super(MultiModelWrapper, self).__init__()
        self.sub_models = sub_models
        self._concat_dim = concat_dim

    def cuda(self):
        for sm in self.sub_models:
            sm.cuda()

    def eval(self):
        for sm in self.sub_models:
            sm.eval()

    def forward(self, x):
        return torch.cat(
            [sm(x) for sm in self.sub_models], self._concat_dim)

