import numpy as np

from .handler import write_to_file, PredictionsHandler

class WritePredictionsHandler(PredictionsHandler):

    def __init__(self, features_list, nonfeature_columns, out_filename):
        self.needs_base_pred = False
        self.column_names = nonfeature_columns + features_list
        self.results = []
        self.samples = []
        self.out_filename = out_filename

    def handle_NA(self, batch_ids):
        super().handle_NA(batch_ids)

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids):
        self.results.append(batch_predictions)
        self.samples.append(batch_ids)

    def write_to_file(self):
        self.results = np.vstack(self.results)
        self.samples = np.vstack(self.samples)
        write_to_file(self.results,
                       self.samples,
                       self.column_names,
                       self.out_filename)
