"""
Description:
    Compare the diff scores for the nominally significant and
    nonsignificant variants.

Output:
    Saves all information and the figure generated to files

Usage:
    variant_groups_comparison.py <features-path> <nonsig-scores-npz>
                                   <sig-scores-npz> <output-dir>
    variant_groups_comparison.py -h | --help

Options:
    -h --help                 Show this screen.

    <features-path>           Path to the features file.
    <nonsig-scores-npz>       Path to the matrix of difference scores for
                              nonsignificant GWAS SNPs.
    <sig-scores-npz>          Path to the matrix of difference scores for
                              nominally significant GWAS SNPs.
    <output-dir>              Path to the output directory.
"""
import os

from docopt import docopt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.preprocessing import quantile_transform
import statsmodels.stats.api as sms
from statsmodels.sandbox.stats.multicomp import multipletests

from selene_sdk.utils import load_features_list


plt.style.use("seaborn-colorblind")


def compute_feature_mannwhitneyu(features,
                                 nonsig_scores_matrix,
                                 sig_scores_matrix):
    """
    For each feature, compute the 1-sided Wilcoxon rank sum test for
    the 2 groups of variants.
    """
    feature_pvalues = {}
    for i, feature in enumerate(features):
        nonsig_scores = nonsig_scores_matrix[:, i]
        sig_scores = sig_scores_matrix[:, i]
        _, pval = scipy.stats.mannwhitneyu(
            sig_scores, nonsig_scores, alternative="greater")
        feature_pvalues[feature] = pval
    return feature_pvalues


def visualize_group_means(feature,
                          qvalue,
                          sig_mean,
                          nonsig_mean,
                          sig_lower,
                          sig_upper,
                          nonsig_lower,
                          nonsig_upper,
                          output_dir='.'):
    """
    Generate the figure visualizing the mean/CIs for the 2 groups of variants.
    """
    plt.rcParams["font.size"] = 10
    title = "Feature {0} (q-value={1:.2e})".format(feature, qvalue)
    fig = plt.figure()
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(6, 4))
    ax.errorbar([1, 2],
                [sig_mean, nonsig_mean],
                yerr=[sig_upper - sig_lower, nonsig_upper - nonsig_lower],
                fmt='o')
    plt.xticks([.25, 1, 2, 2.75],
               ["",
                "GWAS nominally significant",
                "GWAS nonsignificant",
                ""])
    ax.set_title(title)
    ax.set_ylabel("Gaussian transformed predicted effect scores")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "{0}_group_means.png".format(feature)),
        dpi=600)


if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="1.0")

    features = load_features_list(
        arguments["<features-path>"])
    nonsig_scores_matrix = np.load(
        arguments["<nonsig-scores-npz>"])["data"]
    sig_scores_matrix = np.load(
        arguments["<sig-scores-npz>"])["data"]
    output_dir = arguments["<output-dir>"]

    os.makedirs(output_dir, exist_ok=True)

    feature_pvalues = compute_feature_mannwhitneyu(
        features, nonsig_scores_matrix, sig_scores_matrix)

    with open(os.path.join(output_dir, "feature_pvalues.txt"),
              'w+') as file_handle:
        for (f, pval) in feature_pvalues.items():
            file_handle.write("{0}\t{1}\n".format(f, pval))
    print("Finished computing p-values for 1-sided Wilcoxon rank sum test")

    features_ordered, pvalues = zip(*feature_pvalues.items())
    below_alpha, qvalues, _, _ = multipletests(
        pvalues, alpha=0.05, method="fdr_bh", is_sorted=False)
    below_alpha_indices = [ix for ix, a in enumerate(below_alpha) if a]
    print("FDR adjustment: {0} features below alpha = 0.05".format(
        len(below_alpha_indices)))

    # quantile normalization for visualization
    combined_mat = np.vstack([sig_scores_matrix, nonsig_scores_matrix])
    combined_mat_qt = quantile_transform(
        combined_mat,
        axis=0,
        output_distribution="normal",
        random_state=10)

    np.savez_compressed(os.path.join(output_dir, "combined_scores_qt.npz"),
                        data=combined_mat_qt)
    qt_sig_scores_matrix = combined_mat_qt[:sig_scores_matrix.shape[0], :]
    qt_nonsig_scores_matrix = combined_mat_qt[sig_scores_matrix.shape[0]:, :]

    info = list(zip(features_ordered, pvalues, qvalues))
    # sort by q-value
    sorted_info = sorted(info, key=lambda tup: tup[2])

    visualize_data = None
    with open(os.path.join(output_dir, "feature_qvalues_and_qt_means.txt"),
              'w+') as file_handle:
        column_names = ["feature", "pvalue", "qvalue",
                        "signif_mean", "nonsignif_mean",
                        "signif_lo_interval", "signif_up_interval",
                        "nonsignif_lo_interval", "nonsignif_up_interval"]
        file_handle.write("{0}\n".format("\t".join(column_names)))
        for index, (feat, pval, qval) in enumerate(sorted_info):
            sig_scores = qt_sig_scores_matrix[:, index]
            nonsig_scores = qt_nonsig_scores_matrix[:, index]

            sig_group_descr = sms.DescrStatsW(sig_scores)
            nonsig_group_descr = sms.DescrStatsW(nonsig_scores)

            sig_lower, sig_upper = sig_group_descr.tconfint_mean()
            nonsig_lower, nonsig_upper = nonsig_group_descr.tconfint_mean()

            sig_mean = sig_group_descr.mean
            nonsig_mean = nonsig_group_descr.mean

            values = [feat, pval, qval,
                      sig_mean, nonsig_mean,
                      sig_lower, sig_upper,
                      nonsig_lower, nonsig_upper]
            if visualize_data is None:
                # visualize the feature with the smallest q-value
                visualize_data = values[:1] + values[2:]
            file_handle.write("{0}\n".format(
                '\t'.join([str(s) for s in values])))

    visualize_group_means(*visualize_data, output_dir)
