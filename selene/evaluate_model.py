import logging
import os

import numpy as np
import torch
from torch.autograd import Variable

from .utils import initialize_logger
from .utils import load_features_list
from .utils import PerformanceMetrics


logger = logging.getLogger("selene")


class EvaluateModel(object):
    """
    Evaluate model on a test set of sequences with known targets.
    """

    def __init__(self,
                 model,
                 criterion,
                 data_sampler,
                 features_file,
                 trained_model_file,
                 batch_size,
                 output_dir,
                 n_test_samples=None,
                 report_gt_feature_n_positives=10,
                 use_cuda=False,
                 known_targets=True):
        self.model = model
        self.criterion = criterion

        trained_model = torch.load(trained_model_file)
        self.model.load_state_dict(trained_model["state_dict"])
        self.model.eval()

        self.sampler = data_sampler

        self.features = load_features_list(features_file)
        self.index_to_feature = {i: f for i, f in enumerate(self.features)}

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        initialize_logger(
            os.path.join(self.output_dir, f"{__name__}.log"),
            verbosity=2)

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

        self.batch_size = batch_size

        self.known_targets = known_targets
        if self.known_targets:
            self._metrics = PerformanceMetrics(
                self.get_feature_from_index,
                report_gt_feature_n_positives=report_gt_feature_n_positives)

            self._test_data, self._all_test_targets = \
                self.sampler.get_data_and_targets(self.batch_size, n_test_samples)
        else:
            self._test_data, = \
                self.sampler.get_data_and_targets(self.batch_size, n_test_samples)

    def get_feature_from_index(self, feature):
        return self.index_to_feature[feature]

    def evaluate(self):
        batch_losses = []
        all_predictions = []
        for (inputs, targets) in self._test_data:

            inputs = torch.Tensor(inputs)
            targets = torch.Tensor(targets)

            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            inputs = Variable(inputs, volatile=True)
            targets = Variable(targets, volatile=True)

            predictions = self.model(inputs.transpose(1, 2))
            loss = self.criterion(predictions, targets)

            all_predictions.append(predictions.data.cpu().numpy())
            batch_losses.append(loss.data[0])
        all_predictions = np.vstack(all_predictions)

        average_scores = self._metrics.update(
            all_predictions, self._all_test_targets)

        np.savez_compressed(
            os.path.join(self.output_dir, "test_predictions.npz"),
            data=all_predictions)

        np.savez_compressed(
            os.path.join(self.output_dir, "test_targets.npz"),
            data=self._all_test_targets)

        loss = np.average(batch_losses)
        logger.debug(f"[STATS] average loss: {loss}")
        for name, score in average_scores.items():
            logger.debug(f"[STATS] average {name}: {score}")

        test_performance = os.path.join(
            self.output_dir, "test_performance.txt")
        feature_scores_dict = self._metrics.write_feature_scores_to_file(
            test_performance)

        return feature_scores_dict
