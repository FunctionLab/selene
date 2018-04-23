import numpy as np

from .handler import _write_to_file, PredictionsHandler


class DiffScoreHandler(PredictionsHandler):

    def __init__(self,
                 baseline_prediction,
                 features_list,
                 nonfeature_columns,
                 out_filename):
        self.baseline_prediction = baseline_prediction
        self.column_names = nonfeature_columns + features_list
        self.results = []
        self.samples = []
        self.out_filename = out_filename

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids):
        absolute_diffs = np.abs(self.baseline_prediction - batch_predictions)
        self.results.append(absolute_diffs)
        self.samples.append(batch_ids)
        return absolute_diffs

    def write_to_file(self):
        self.results = np.vstack(self.results)
        self.samples = np.vstack(self.samples)
        _write_to_file(self.results,
                       self.samples,
                       self.column_names,
                       self.out_filename)
