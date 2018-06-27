"""
This module provides the `PerformanceMetrics` class and supporting
functionality for tracking and computing model performance.

"""
from collections import defaultdict, namedtuple
import types
import logging

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


logger = logging.getLogger("selene")


Metric = namedtuple("Metric", ["fn", "data"])
"""
A tuple containing a metric function and the results from applying that
metric to some values.

Parameters
----------
fn : types.FunctionTYpe
    A metric.
data : list(float)
    A list holding the results from applying the metric.

Attributes
----------
fn : types.FunctionType
    A metric.
data : list(float)
    A list holding the results from applying the metric.

"""


def compute_score(prediction, target, metric_fn,
                  report_gt_feature_n_positives=10):
    """
    Using a user-specified metric, computes the distance between
    two tensors.

    Parameters
    ----------
    prediction : numpy.ndarray
        Value predicted by user model.
    target : numpy.ndarray
        True value that the user model was trying to predict.
    metric_fn : types.FunctionType
        A metric that can measure the distance between the prediction
        and target variables.
    report_gt_feature_n_positives : int, optional
        Default is 10. The minimum number of positive examples for a
        feature in order to compute the score for it.

    Returns
    -------
    average_score, feature_scores : tuple(float, numpy.ndarray)
        A tuple containing the average of all feature scores, and a
        vector containing the scores for each feature. If there were
        no features meeting our filtering thresholds, will return
        `(None, [])`.
    """
    feature_scores = np.ones(target.shape[1]) * -1
    for index, feature_preds in enumerate(prediction.T):
        feature_targets = target[:, index]
        if len(np.unique(feature_targets)) > 1 and \
                np.sum(feature_targets) > report_gt_feature_n_positives:
            feature_scores[index] = metric_fn(
                feature_targets, feature_preds)
    valid_feature_scores = [s for s in feature_scores if s >= 0]
    if not valid_feature_scores:
        return None, feature_scores
    average_score = np.average(valid_feature_scores)
    return average_score, feature_scores


def get_feature_specific_scores(data, get_feature_from_index_fn):
    """
    Generates a dictionary mapping feature names to feature scores from
    an intermediate representation.

    Parameters
    ----------
    data : list(tuple(int, float))
        A list of tuples, where each tuple contains a feature's index
        and the score for that feature.
    get_feature_from_index_fn : types.FunctionType
        A function that takes an index (`int`) and returns a feature
        name (`str`).

    Returns
    -------
    dict
        A dictionary mapping feature names (`str`) to scores (`float`).
        If there was no score for a feature, its score will be set to
        `None`.

    """
    feature_score_dict = {}
    for index, score in enumerate(data):
        feature = get_feature_from_index_fn(index)
        if score >= 0:
            feature_score_dict[feature] = score
        else:
            feature_score_dict[feature] = None
    return feature_score_dict


class PerformanceMetrics(object):
    """
    Tracks and calculates metrics to evaluate how closely a model's
    predictions match the true values it was designed to predict.

    Parameters
    ----------
    get_feature_from_index_fn : types.FunctionType
        A function that takes an index (`int`) and returns a feature
        name (`str`).
    report_gt_feature_n_positives : int, optional
        Default is 10. The minimum number of positive examples for a
        feature in order to compute the score for it.

    Attributes
    ----------
    skip_threshold : int
        The minimum number of positive examples of a feature that must
        be included in an update for a metric score to be
        calculated for it.
    get_feature_from_index : types.FunctionType
        A function that takes an index (`int`) and returns a feature
        name (`str`).
    metrics : dict
        A dictionary that maps metric names (`str`) to metric objects
        (`Metric`). By default, this contains `"roc_auc"` and
        `"average_precision"`.

    """

    def __init__(self,
                 get_feature_from_index_fn,
                 report_gt_feature_n_positives=10):
        """
        Creates a new object of the `PerformanceMetrics` class.
        """
        self.skip_threshold = report_gt_feature_n_positives
        self.get_feature_from_index = get_feature_from_index_fn
        self.metrics = {
            "roc_auc": Metric(fn=roc_auc_score, data=[]),
            "average_precision": Metric(fn=average_precision_score, data=[])
        }

    def add_metric(self, name, metric_fn):
        """
        Begins tracking of the specified metric.

        Parameters
        ----------
        name : str
            The name of the metric.
        metric_fn : types.FunctionType
            A metric function.

        """
        self.metrics[name] = Metric(fn=metric_fn, data=[])

    def remove_metric(self, name):
        """
        Ends the tracking of the specified metric, and returns the
        previous scores associated with that metric.

        Parameters
        ----------
        name : str
            The name of the metric.

        Returns
        -------
        list(float)
            The list of feature-specific scores obtained by previous
            uses of the specified metric.

        """
        data = self.metrics[name].data
        del self.metrics[name]
        return data

    def update(self, prediction, target):
        """
        Evaluates the tracked metrics on a model prediction and its
        target value, and adds this to the metric histories.

        Parameters
        ----------
        prediction : numpy.ndarray
            Value predicted by user model.
        target : numpy.ndarray
            True value that the user model was trying to predict.

        Returns
        -------
        dict
            A dictionary mapping each metric names (`str`) to the
            average score of that metric across all features
            (`float`).

        """
        metric_scores = {}
        for name, metric in self.metrics.items():
            avg_score, feature_scores = compute_score(
                prediction, target, metric.fn,
                report_gt_feature_n_positives=self.skip_threshold)
            metric.data.append(feature_scores)
            metric_scores[name] = avg_score
        return metric_scores

    def write_feature_scores_to_file(self, output_path):
        """
        Writes each metric's score for each feature to a specified
        file.

        Parameters
        ----------
        output_path : str
            The path to the output file where performance metrics will
            be written.

        Returns
        -------
        dict
            A dictionary mapping feature names (`str`) to
            sub-dictionaries (`dict`). Each sub-dictionary then maps
            metric names (`str`) to the score for that metric on the
            given feature. If a metric was not evaluated on a given
            feature, the score will be `None`.

        """
        feature_scores = defaultdict(dict)
        for name, metric in self.metrics.items():
            feature_score_dict = get_feature_specific_scores(
                metric.data[-1], self.get_feature_from_index)
            for feature, score in feature_score_dict.items():
                if score is None:
                    feature_scores[feature] = None
                else:
                    feature_scores[feature][name] = score

        metric_cols = [m for m in self.metrics.keys()]
        cols = '\t'.join(["features"] + metric_cols)
        with open(output_path, 'w+') as file_handle:
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
