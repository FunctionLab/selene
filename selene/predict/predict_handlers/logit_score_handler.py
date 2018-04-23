import numpy as np
from scipy.special import logit

from .handler import _write_to_file, PredictionsHandler


class LogitScoreHandler(PredictionsHandler):

    def __init__(self,
                 baseline_prediction,
                 features_list,
                 nonfeature_columns,
                 out_filename):
        self.logit_baseline = logit(baseline_prediction)
        self.column_names = nonfeature_columns + features_list
        self.results = []
        self.samples = []
        self.out_filename = out_filename

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids):
        absolute_logits = np.abs(self.logit_baseline - logit(batch_predictions))
        self.results.append(absolute_logits)
        self.samples.append(batch_ids)
        return absolute_logits

    def write_to_file(self):
        self.results = np.vstack(self.results)
        self.samples = np.vstack(self.samples)
        _write_to_file(self.results,
                       self.samples,
                       self.column_names,
                       self.out_filename)
