"""This module provides the `PerformanceMetrics` class and supporting
functionality for tracking and computing model performance.

"""
from collections import defaultdict, namedtuple
import logging

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


logger = logging.getLogger("selene")
Metric = namedtuple("Metric", ["fn", "data"])


def compute_score(targets, predictions,
                  compute_score_fn,
                  report_gt_feature_n_positives=10):
    """
    # TODO(DOCUMENTATION): Finish.

    Parameters
    ----------
    targets
    predictions
    compute_score_fn
    report_gt_feature_n_positives

    Returns
    -------

    """
    feature_scores = np.ones(targets.shape[1]) * -1
    for index, feature_preds in enumerate(predictions.T):
        feature_targets = targets[:, index]
        if len(np.unique(feature_targets)) > 1 and \
                np.sum(feature_targets) > report_gt_feature_n_positives:
            feature_scores[index] = compute_score_fn(
                feature_targets, feature_preds)
        else:
            print("Did not compute metrics for a feature.")
    valid_feature_scores = [s for s in feature_scores if s >= 0]
    if not valid_feature_scores:
        return None, feature_scores
    average_score = np.average(valid_feature_scores)
    return average_score, feature_scores


def get_feature_specific_scores(data, get_feature_from_ix_fn):
    """
    # TODO(DOCUMENTATION): Finish.

    Parameters
    ----------
    data
    get_feature_from_ix_fn

    Returns
    -------

    """
    feature_score_dict = {}
    for index, score in enumerate(data):
        feature = get_feature_from_ix_fn(index)
        if score >= 0:
            feature_score_dict[feature] = score
        else:
            feature_score_dict[feature] = None
    return feature_score_dict


class PerformanceMetrics(object):
    """Report metrics in addition to loss

    Attributes
    ----------
    # TODO(DOCUMENTATION): Finish.
    """

    def __init__(self,
                get_feature_from_ix_fn,
                report_gt_feature_n_positives=10):
        """
        # TODO(DOCUMENTATION): Finish.

        Parameters
        ----------
        get_feature_from_ix_fn
        report_gt_feature_n_positives
        """
        self.skip_threshold = report_gt_feature_n_positives
        self.feature_from_ix = get_feature_from_ix_fn
        self.metrics = {
            "roc_auc": Metric(fn=roc_auc_score, data=[]),
            "average_precision": Metric(fn=average_precision_score, data=[])
        }

    def add_metric(self, name, metric_fn):
        """
        # TODO(DOCUMENTATION): Finish.

        Parameters
        ----------
        name
        metric_fn

        Returns
        -------

        """
        self.metrics[name] = Metric(fn=metric_fn, data=[])

    def remove_metric(self, name):
        """
        # TODO(DOCUMENTATION): Finish.

        Parameters
        ----------
        name

        Returns
        -------

        """
        data = self.metrics[name].data
        del self.metrics[name]
        return data

    def update(self, targets, predictions):
        """
        # TODO(DOCUMENTATION): Finish.

        Parameters
        ----------
        targets
        predictions

        Returns
        -------

        """
        metric_scores = {}
        for name, metric in self.metrics.items():
            avg_score, feature_scores = compute_score(
                targets, predictions, metric.fn,
                report_gt_feature_n_positives=self.skip_threshold)
            metric.data.append(feature_scores)
            metric_scores[name] = avg_score
        return metric_scores

    def write_feature_scores_to_file(self, output_file):
        """
        # TODO(DOCUMENTATION): Finish.

        Parameters
        ----------
        output_file

        Returns
        -------

        """
        feature_scores = defaultdict(dict)
        for name, metric in self.metrics.items():
            feature_score_dict = get_feature_specific_scores(
                metric.data[-1], self.feature_from_ix)
            for feature, score in feature_score_dict.items():
                if score is None:
                    feature_scores[feature] = None
                else:
                    feature_scores[feature][name] = score

        metric_cols = [m for m in self.metrics.keys()]
        cols = '\t'.join(["features"] + metric_cols)
        with open(output_file, 'w+') as file_handle:
            file_handle.write("{0}\n".format(cols))
            for feature, metric_scores in sorted(feature_scores.items()):
                if not metric_scores:
                    file_handle.write("{0}\tNA\tNA\n".format(feature))
                else:
                    metric_score_cols = '\t'.join(
                        ["{0:.4f}".format(s) for s in metric_scores.values()])
                    file_handle.write("{0}\t{1}\n".format(feature,
                                                          metric_score_cols))
        return feature_scores
