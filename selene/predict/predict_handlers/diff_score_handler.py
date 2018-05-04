import numpy as np

from .handler import write_to_file, PredictionsHandler


class DiffScoreHandler(PredictionsHandler):

    def __init__(self,
                 features_list,
                 nonfeature_columns,
                 out_filename):
        super(DiffScoreHandler, self).__init__()

        self.needs_base_pred = True
        self.column_names = nonfeature_columns + features_list
        self.results = []
        self.samples = []
        self.NA_samples = []
        self.out_filename = out_filename

    def handle_NA(self, batch_ids):
        super().handle_NA(batch_ids)

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids,
                                 baseline_predictions):
        absolute_diffs = np.abs(baseline_predictions - batch_predictions)
        self.results.append(absolute_diffs)
        self.samples.append(batch_ids)

    def write_to_file(self):
        self.results = np.vstack(self.results)
        self.samples = np.vstack(self.samples)
        write_to_file(self.results,
                       self.samples,
                       self.column_names,
                       self.out_filename)
