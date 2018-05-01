from .handler import PredictionsHandler
from .write_predictions_handler import WritePredictionsHandler

class WriteRefAltHandler(PredictionsHandler):

    def __init__(self, features_list, nonfeature_columns, out_filename):
        self.needs_base_pred = True
        self.ref_writer = WritePredictionsHandler(
            features_list, nonfeature_columns, f"{out_filename}.ref")
        self.alt_writer = WritePredictionsHandler(
            features_list, nonfeature_columns, f"{out_filename}.alt")

    def handle_NA(self, batch_ids):
        self.ref_writer.handle_NA(batch_ids)
        self.alt_writer.handle_NA(batch_ids)

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids,
                                 base_predictions):
        self.ref_writer.handle_batch_predictions(
            base_predictions, batch_ids)
        self.alt_writer.handle_batch_predictions(
            batch_predictions, batch_ids)

    def write_to_file(self):
        self.ref_writer.write_to_file()
        self.alt_writer.write_to_file()
