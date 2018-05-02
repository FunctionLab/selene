import numpy as np
from scipy.special import logit

from .handler import write_to_file, PredictionsHandler


class LogitScoreHandler(PredictionsHandler):

    def __init__(self,
                 features_list,
                 nonfeature_columns,
                 out_filename):
        self.needs_base_pred = True
        self.column_names = nonfeature_columns + features_list
        self.results = []
        self.samples = []
        self.out_filename = out_filename

    def handle_NA(self, batch_ids):
        super().handle_NA(batch_ids)

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids,
                                 baseline_predictions):
        absolute_logits = np.abs(
            logit(baseline_predictions) - logit(batch_predictions))
        self.results.append(absolute_logits)
        self.samples.append(batch_ids)

    def write_to_file(self):
        self.results = np.vstack(self.results)
        self.samples = np.vstack(self.samples)
        write_to_file(self.results,
                       self.samples,
                       self.column_names,
                       self.out_filename)
