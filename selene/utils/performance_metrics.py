from collections import defaultdict, namedtuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


Metric = namedtuple("Metric", ["fn", "data"])


def compute_score(targets, predictions,
                  compute_score_fn,
                  report_gt_feature_n_positives=10):
    feature_scores = np.ones(targets.shape[1]) * -1
    for index, feature_preds in enumerate(predictions.T):
        feature_targets = targets[:, index]
        if len(np.unique(feature_targets)) > 1 and \
                np.sum(feature_targets) < report_gt_feature_n_positives:
            feature_scores[index] = compute_score_fn(
                feature_targets, feature_preds)

    valid_feature_scores = [s for s in feature_scores if s >= 0]
    average_score = np.average(valid_feature_scores)
    return average_score, feature_scores

def get_feature_specific_scores(data, get_feature_from_ix_fn):
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
    """

    def __init__(self,
                get_feature_from_ix_fn,
                report_gt_feature_n_positives=10):
        self.skip_threshold = report_gt_feature_n_positives
        self.feature_from_ix = get_feature_from_ix_fn
        self.metrics = {
            "roc_auc": Metric(fn=roc_auc_score, data=[]),
            "average_precision": Metric(fn=average_precision_score, data=[])
        }

    def add_metric(self, name, metric_fn):
        self.metrics[name] = Metric(fn=metric_fn, data=[])

    def remove_metric(self, name):
        data = self.metrics[name].data
        del self.metrics[name]
        return data

    def update(self, targets, predictions):
        metric_scores = {}
        for name, metric in self.metrics.items():
            avg_score, feature_scores = compute_score(
                targets, predictions, metric.fn,
                report_gt_feature_n_positives=self.skip_threshold)
            metric.data.append(feature_scores)
            metric_scores[name] = avg_score
        return metric_scores

    def write_feature_scores_to_file(self, output_file):
        feature_scores = defaultdict(dict)
        for name, metric in self.metrics.items():
            feature_score_dict = get_feature_specific_scores(
                metric.data[-1], self.feature_from_ix)
            for feature, score in feature_score_dict.items():
                feature_scores[feature][name] = score

        metric_cols = [m for m in self.metrics.keys()]
        cols = '\t'.join(["features"] + metric_cols)
        print(output_file)
        print(cols, len(feature_scores))
        with open(output_file, 'w+') as file_handle:
            file_handle.write(f"{cols}\n")
            for feature, metric_scores in sorted(feature_scores.items()):
                metric_score_cols = '\t'.join(
                    [f"{s:.4f}" for s in metric_scores.values()])
                file_handle.write(f"{feature}\t{metric_score_cols}\n")

        return feature_scores

