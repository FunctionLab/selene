"""
This module provides the methods for visualizing different ouputs
from selene analysis methods.

"""
from collections import defaultdict
from copy import deepcopy
import os
import re
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.patheffects
from matplotlib.text import TextPath
import pkg_resources
from plotly.offline import download_plotlyjs, plot
import plotly.graph_objs as go
import seaborn as sns
import tabix

from ..sequences import Genome


_SVG_PATHS = {'T': "M 0,100 l 100, 0 l 0,-25 l -37.5, 0 l 0,-75 l -25, 0 " +
                   "l 0,75 l -37.5,0 l 0,25 z",
              'C': ("M 100,12.5 l 0,25 c 0,0 -25,-15 -50,-12.5 " +
                    "c 0,0 -25,0 -25,25 c 0,0 0,25 25,25 c 0,0 25,2.5 50,-15" +
                    " l 0, 25 C 100,87.5 75,100 50,100 C 50,100 0,100 0,50 " +
                    "C 0,50 0,0 50,0 C 50,0 75,0 100,12.5 z"),
              'G': ("M 100,12.5 l 0,25 c 0,0 -25,-15 -50,-12.5 " +
                    "c 0,0 -25,0 -25,25 c 0,0 0,25 25,25 c 0,0 25,2.5 50,-15" +
                    " l 0, 25 C 100,87.5 75,100 50,100 C 50,100 0,100 0,50 " +
                    "C 0,50 0,0 50,0 C 50,0 75,0 100,12.5 M 100,37.5 " +
                    "l 0,17.5 l -50,0 l 0,-17 l 25,0 l 0,-25 l 25,0 z"),
              'A': ("M 0,0 l 37.5,100 l 25,0 l 37.5,-100 l -25,0 l -9.375,25" +
                    " l -31.25,0 l -9.375,-25 l -25,0 z 0,0 M 43.75, 50 " +
                    "l 12.5,0 l -5.859375,15.625 l -5.859375,-15.625 z"),
              'U': ("M 0,100 l 25,0 l 0,-50 C 25,50 25,25 50,25" +
                    " C 50,25 75,25 75,50 l 0,50 l 25,0 L 100,50 " +
                    "C 100,50 100,0, 50,0 C 50,0 0,0 0,50 l 0,50 z")}


def _svg_parse(path_string):
    """Functionality for parsing a string from source vector graphics.

    Source is from `matplotlib.org/2.1.1/gallery/showcase/firefox.html`
    with minor modifications.

    Parameters
    ----------
    path_string : str
        String containing the path code from an SVG file.

    Returns
    -------
    list(numpy.uint8), numpy.ndarray, dtype=np.float32
        A 2-tuple containing code types and coordinates for a matplotlib
        path.

    """
    commands = {'M': (Path.MOVETO,),
                'L': (Path.LINETO,),
                'Q': (Path.CURVE3,)*2,
                'C': (Path.CURVE4,)*3,
                'Z': (Path.CLOSEPOLY,)}
    path_re = re.compile(r'([MLHVCSQTAZ])([^MLHVCSQTAZ]+)', re.IGNORECASE)
    float_re = re.compile(r'(?:[\s,]*)([+-]?\d+(?:\.\d+)?)')
    vertices = []
    codes = []
    last = (0, 0)
    for cmd, values in path_re.findall(path_string):
        points = [float(v) for v in float_re.findall(values)]
        points = np.array(points).reshape((len(points)//2, 2))
        if cmd.islower():
            points += last
        cmd = cmd.capitalize()
        if len(points) > 0:
            last = points[-1]
        codes.extend(commands[cmd])
        vertices.extend(points.tolist())
    return np.array(vertices), codes


for k in _SVG_PATHS.keys():
    _SVG_PATHS[k] = _svg_parse(_SVG_PATHS[k])


class _TextPathRenderingEffect(matplotlib.patheffects.AbstractPathEffect):
    """This class provides an effect for continuously rendering a text
    path over another path.

    """
    def __init__(self, bar, x_translation=0., y_translation=0.,
                 x_scale=1., y_scale=1.):
        """This is a class for re-rendering text paths and preserving
        their scale.

        Parameters
        ----------
        bar : matplotlib.patches.Patch
            The patch where the letter is.
        x_translation : float, optional
            Default is 0. Amount by which to translate the x coordinate.
        y_translation : float, optional
            Default is 0. Amount by which to translate the y coordinate.
        x_scale : float, optional
            Default is 1. Amount by which to scale the width.
        y_scale : float, optional
            Default is 1. Amount by which to scale the height.

        """
        self._bar = bar
        self._x_translation = x_translation
        self._y_translation = y_translation
        self._x_scale = x_scale
        self._y_scale = y_scale

    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        """Redraws the path.

        """
        b_x, b_y, b_w, b_h = self._bar.get_extents().bounds
        t_x, t_y, t_w, t_h = tpath.get_extents().bounds
        translation = [b_x - t_x, b_y - t_y]
        translation[0] += self._x_translation
        translation[1] += self._y_translation
        scale = [b_w / t_w, b_h / t_h]
        scale[0] *= self._x_scale
        scale[1] *= self._y_scale
        affine = affine.identity().scale(*scale).translate(*translation)
        renderer.draw_path(gc, tpath, affine, rgbFace)


def sequence_logo(score_matrix, order="value", width=1.0, ax=None,
                  sequence_type=Genome, font_properties=None,
                  color_scheme=None,
                  **kwargs):
    """Plots a sequence logo for visualizing motifs.

    Parameters
    ----------
    score_matrix : np.ndarray
        An :math:`L \\times N` array (where :math:`L` is the length of
        the sequence, and :math:`N` is the size of the alphabet)
        containing the scores for each base occuring at each position.
    order : {'alpha', 'value'}
        The manner by which to sort the bases stacked at each position
        in the sequence logo plot.

            * 'alpha' - Bases go in the order they are found in the\
                      sequence alphabet.
            * 'value' - Bases go in the order of their value, with the\
                        largest at the bottom.
    width : float, optional
        Default is 1. The width of each character in the plotted logo.
        A value of 1 will mean that there is no gap between each the
        characters at each position. A value of 0 will not draw any
        characters.
    ax : matplotlib.pyplot.Axes or None, optional
        Default is `None`. The axes to plot on. If left as `None`, a new
        axis will be created.
    sequence_type : class, optional
        Default is `selene_sdk.sequences.Genome`. The type of sequence that
        the *in silico* mutagenesis results are associated with. This
        should generally be a subclass of `selene_sdk.sequences.Sequence`.
    font_properties : matplotlib.font_manager.FontProperties or None, optional
        Default is `None`. A `matplotlib.font_manager.FontProperties`
        object that specifies the properties of the font to use for
        plotting the motif. If `None`, no font will be used, and the
        text will be rendered by a path. This method of rendering paths
        is  preferred, as it ensures all character heights correspond to
        the actual values, and that there are no extra gaps between
        the tops and bottoms of characters at each position in the
        sequence logo. If the user opts to use a value other
        than `None`, then no such guarantee can be made.
    color_scheme : list(str) or None, optional
        Default is `None`. A list containing the hex codes or names of
        colors to use, appearing in the order of the bases of the
        sequence type. If left as `None`, a default palette will be made
        with `seaborn.color_palette`, and will have as many
        colors as there are characters in the input sequence alphabet.

    Returns
    -------
    matplotlib.pyplot.Axes
        The axes containing the sequence logo plot.

    Raises
    ------
    ValueError
        If the number of columns in `score_matrix` does not match the
        number of characters in the alphabet of `sequence_type`.

    ValueError
        If the number of colors in `color_palette` does not match the
        number of characters in the alphabet of `sequence_type`.

    Examples
    --------
    We have included an example of the output from a`sequence_logo`
    plot below:

    .. image:: ../../docs/source/_static/img/sequence_logo_example.png

    """
    # Note that everything will break if we do not deepcopy.
    score_matrix = deepcopy(score_matrix)

    score_matrix = score_matrix.transpose()
    if font_properties is not None:
        warnings.warn(
            "Specifying a value for `font_properties` (other than `None`) "
            "will use the `matplotlib`-based character paths, and causes "
            "distortions in the plotted motif. We recommend leaving "
            "`font_properties=None`. See the documentation for details.",
            UserWarning)

    if color_scheme is None:
        color_scheme = sns.color_palette("Set1",
                                         n_colors=len(sequence_type.BASES_ARR))
        color_scheme = color_scheme.as_hex()
    if len(color_scheme) < len(sequence_type.BASES_ARR):
        raise ValueError(
            "Color scheme is shorter than number of bases in sequence.")

    if score_matrix.shape[0] != len(sequence_type.BASES_ARR):
        raise ValueError(
            "Got score with {0} bases for sequence with {1} bases.".format(
                score_matrix.shape[0], len(sequence_type.BASES_ARR)))
    if ax is None:
        _, ax = plt.subplots(figsize=score_matrix.shape)

    # Determine offsets depending on sort order.
    positive_offsets = np.zeros_like(score_matrix)
    negative_offsets = np.zeros_like(score_matrix)
    bases = np.empty(score_matrix.shape, dtype=object)
    bases[:, :] = "?"  # This ensures blanks are visually obvious.

    # Change ordering of things based on input arguments.
    if order == "alpha":
        for i in range(score_matrix.shape[0]):
            bases[i, :] = sequence_type.BASES_ARR[i]

    elif order == "value":
        if np.sum(score_matrix < 0) != 0:
            sorted_scores = np.zeros_like(score_matrix)
            for j in range(score_matrix.shape[1]):
                # Sort the negative values and put them at bottom.
                div = np.sum(score_matrix[:, j] < 0.)
                negative_idx = np.argwhere(score_matrix[:, j] < 0.).flatten()
                negative_sort_idx = np.argsort(score_matrix[negative_idx, j],
                                               axis=None)
                sorted_scores[:div, j] = score_matrix[
                    negative_idx[negative_sort_idx], j]
                bases[:div, j] = sequence_type.BASES_ARR[
                    negative_idx[negative_sort_idx]].flatten()

                # Sort the positive values and stack atop the negatives.
                positive_idx = np.argwhere(score_matrix[:, j] >= 0.).flatten()
                positive_sort_idx = np.argsort(score_matrix[positive_idx, j],
                                               axis=None)
                sorted_scores[div:, j] = score_matrix[
                    positive_idx[positive_sort_idx], j]
                bases[div:, j] = sequence_type.BASES_ARR[
                    positive_idx[positive_sort_idx]].flatten()
            score_matrix = sorted_scores
        else:
            for j in range(score_matrix.shape[1]):
                sort_idx = np.argsort(score_matrix[:, j], axis=None)[::-1]
                bases[:, j] = sequence_type.BASES_ARR[sort_idx]
                score_matrix[:, j] = score_matrix[sort_idx, j]

    # Create offsets for each bar.
    for i in range(score_matrix.shape[0] - 1):
        y_coords = score_matrix[i, :]
        if i > 0:
            negative_offsets[i + 1, :] = negative_offsets[i, :]
            positive_offsets[i + 1, :] = positive_offsets[i, :]
        neg_idx = np.argwhere(y_coords < 0.)
        pos_idx = np.argwhere(y_coords >= 0.)
        negative_offsets[i + 1, neg_idx] += y_coords[neg_idx]
        positive_offsets[i + 1, pos_idx] += y_coords[pos_idx]

    for i in range(score_matrix.shape[0]):
        x_coords = np.arange(score_matrix.shape[1]) + 0.5
        y_coords = score_matrix[i, :]

        # Manage negatives and positives separately.
        offsets = np.zeros(score_matrix.shape[1])
        negative_idx = np.argwhere(y_coords < 0.)
        positive_idx = np.argwhere(y_coords >= 0.)
        offsets[negative_idx] = negative_offsets[i, negative_idx]
        offsets[positive_idx] = positive_offsets[i, positive_idx]
        bars = ax.bar(x_coords, y_coords, color="black", width=width,
                      bottom=offsets)
        for j, bar in enumerate(bars):
            base = bases[i, j]
            bar.set_color(color_scheme[sequence_type.BASE_TO_INDEX[base]])
            bar.set_edgecolor(None)

    # Iterate over the barplot's bars and turn them into letters.
    new_patches = []
    for i, bar in enumerate(ax.patches):
        base_idx = i // score_matrix.shape[1]
        seq_idx = i % score_matrix.shape[1]
        base = bases[base_idx, seq_idx]
        # We construct a text path that tracks the bars in the barplot.
        # Thus, the barplot takes care of scaling and translation,
        #  and we just copy it.
        if font_properties is None:
            text = Path(_SVG_PATHS[base][0], _SVG_PATHS[base][1])
        else:
            text = TextPath((0., 0.), base, fontproperties=font_properties)
        b_x, b_y, b_w, b_h = bar.get_extents().bounds
        t_x, t_y, t_w, t_h = text.get_extents().bounds
        scale = (b_w / t_w, b_h / t_h)
        translation = (b_x - t_x, b_y - t_y)
        text = PathPatch(text, facecolor=bar.get_facecolor(), lw=0.)
        bar.set_facecolor("none")
        text.set_path_effects([_TextPathRenderingEffect(bar)])
        transform = transforms.Affine2D().translate(*translation).scale(*scale)
        text.set_transform(transform)
        new_patches.append(text)

    for patch in new_patches:
        ax.add_patch(patch)
    ax.set_xlim(0, score_matrix.shape[1])
    ax.set_xticks(np.arange(score_matrix.shape[1]) + 0.5)
    ax.set_xticklabels(np.arange(score_matrix.shape[1]))
    return ax


def rescale_score_matrix(score_matrix, base_scaling="identity",
                         position_scaling="identity"):
    """Performs base-wise and position-wise scaling of a score matrix for a
    feature, usually produced from an *in silico* mutagenesis experiment.

    Parameters
    ----------
    score_matrix : numpy.ndarray
        An :math:`L \\times N` matrix containing the scores for each
        position, where :math:`L` is the length of the sequence, and
        :math:`N` is the number of characters in the alphabet.
    base_scaling : {'identity', 'probability', 'max_effect'}
        The type of scaling performed on each base at a given position.

            * 'identity' - No transformation will be applied to the\
                           data.
            * 'probability' - The relative sizes of the bases will be\
                              the original input probabilities.
            * 'max_effect' - The relative sizes of the bases will be\
                             the max effect of the original input\
                             values.
    position_scaling : {'identity', 'probability', 'max_effect'}
        The type of scaling performed on each position.

            * 'identity'    - No transformation will be applied to the data.
            * 'probability' - The sum of values at a position will be\
                              equal to the sum of the original input\
                              values at that position.
            * 'max_effect'  - The sum of values at a position will be\
                              equal to the sum of the max effect values\
                              of the original input values at that\
                              position.

    Returns
    -------
    numpy.ndarray
        The transformed score matrix.

    Raises
    ------
    ValueError
        If an unsupported `base_scaling` or `position_scaling` is
        entered.

    """
    # Note that things can break if we do not deepcopy.
    score_matrix = deepcopy(score_matrix)

    score_matrix = score_matrix.transpose()
    rescaled_scores = score_matrix

    # Scale individual bases.
    if base_scaling == "identity" or base_scaling == "probability":
        pass
    elif base_scaling == "max_effect":
        rescaled_scores = score_matrix - np.min(score_matrix, axis=0)
    else:
        raise ValueError(
            "Could not find base scaling \"{0}\".".format(base_scaling))

    # Scale each position
    if position_scaling == "max_effect":
        max_effects = np.max(score_matrix, axis=0) - np.min(score_matrix,
                                                            axis=0)
        rescaled_scores /= rescaled_scores.sum(axis=0)[np.newaxis, :]
        rescaled_scores *= max_effects[np.newaxis, :]
    elif position_scaling == "probability":
        rescaled_scores /= np.sum(score_matrix, axis=0)[np.newaxis, :]
    elif position_scaling != "identity":
        raise ValueError(
            "Could not find position scaling \"{0}\".".format(
                position_scaling))
    return rescaled_scores.transpose()


def heatmap(score_matrix, mask=None, sequence_type=Genome, **kwargs):
    """Plots the input matrix of scores, generally those produced by an
    *in silico* mutagenesis experiment, on a heatmap.

    Parameters
    ----------
    score_matrix : numpy.ndarray
        An :math:`L \\times N` array (where :math:`L` is the length of
        the sequence, and :math:`N` is the size of the alphabet)
        containing the scores for each base change at each position.
    mask : numpy.ndarray, dtype=bool or None, optional
        Default is `None`. An :math:`L \\times N` array (where :math:`L`
        is the length of the sequence, and :math:`N` is the size of the
        alphabet)  containing `True` at positions in the heatmap to
        mask. If `None`, no masking will occur.
    sequence_type : class, optional
        Default is `selene_sdk.sequences.Genome`. The class of sequence that
        the *in silico* mutagenesis results are associated with. This is
        generally a sub-class of `selene_sdk.sequences.Sequence`.
    **kwargs : dict
        Keyword arguments to pass to `seaborn.heatmap`. Some useful ones
        to remember are:

            * cbar_kws - Keyword arguments to forward to the colorbar.
            * yticklabels - Manipulate the tick labels on the y axis.
            * cbar - If `False`, hide the color bar. If `True`, show\
                     the colorbar.
            * cmap - The color map to use for the heatmap.

    Returns
    -------
    matplotlib.pyplot.Axes
        The axes containing the heatmap plot.

    """
    # Note that some things can break if we do not deepcopy.
    score_matrix = deepcopy(score_matrix)

    # This flipping is so that ordering is consistent with ordering
    # in the sequence logo.
    if mask is not None:
        mask = mask.transpose()
        mask = np.flip(mask, axis=0)
    score_matrix = score_matrix.transpose()
    score_matrix = np.flip(score_matrix, axis=0)

    if "yticklabels" in kwargs:
        yticklabels = kwargs.pop("yticklabels")
    else:
        yticklabels = sequence_type.BASES_ARR[::-1]
    if "cbar_kws" in kwargs:
        cbar_kws = kwargs.pop("cbar_kws")
    else:
        cbar_kws = dict(use_gridspec=False, location="bottom", pad=0.2)
    if "cmap" in kwargs:
        cmap = kwargs.pop("cmap")
    else:
        cmap = "Blues"
    ret = sns.heatmap(score_matrix, mask=mask, yticklabels=yticklabels,
                      cbar_kws=cbar_kws, cmap=cmap, **kwargs)
    ret.set_yticklabels(labels=ret.get_yticklabels(), rotation=0)
    return ret


def load_variant_abs_diff_scores(input_path):
    """
    Loads the variant data, labels, and feature names from a diff scores
    file output from variant effect prediction.

    TODO: should we move this out of `vis.py`?

    Parameters
    ----------
    input_path : str
        Path to the input file.

    Returns
    -------
    tuple(np.ndarray, list(tuple(str)), list(str))

        * `tuple[0]` is the matrix of absolute difference scores. The rows\
          are the variants and the columns are the features for which the\
          model makes predictions.
        * `tuple[1]` is the list of variant labels. Each tuple contains\
          (chrom, pos, name, ref, alt).
        * `tuple[2]` is the list of features.

    """
    features = []
    labels = []
    diffs = []
    with open(input_path, 'r') as file_handle:
        colnames = file_handle.readline()
        features = colnames.strip().split('\t')[5:]
        for line in file_handle:
            cols = line.strip().split('\t')
            scores = [float(f) for f in cols[5:]]
            label = tuple(cols[:5])
            diffs.append(scores)
            labels.append(label)
    diffs = np.array(diffs)
    return (diffs, labels, features)


def sort_standard_chrs(chrom):
    """
    Returns the value on which the
    standard chromosomes can be sorted.

    Parameters
    ----------
    chrom : str
        The chromosome

    Returns
    -------
    int
        The value on which to sort

    """
    chrom = chrom[3:]
    if chrom.isdigit():
        return int(chrom)
    if chrom == 'X':
        return 23
    elif chrom == 'Y':
        return 24
    elif chrom == 'M':
        return 25
    else:  # unknown chr
        return 26


def ordered_variants_and_indices(labels):
    """
    Get the ordered variant labels, where the labels are sorted by chromosome
    and position, and the indices corresponding to the sort,

    Parameters
    ----------
    labels : list(tuple(str))
        The list of variant labels. Each label is a tuple of
        (chrom, pos, name, ref, alt).

    Returns
    -------
    tuple(list(tuple), list(int))
        The first value is the ordered list of labels. Each label
        is a tuple of (chrom, pos, ref, alt). The second value
        is the ordered list of label indices.

    """
    labels_dict = defaultdict(list)
    for i, l in enumerate(labels):
        chrom, pos, name, ref, alt = l
        pos = int(pos)
        info = (i, pos, ref, alt)
        labels_dict[chrom].append(info)
    for chrom, labels_list in labels_dict.items():
        labels_list.sort(key=lambda tup: tup[1:])
    ordered_keys = sorted(labels_dict.keys(),
                          key=sort_standard_chrs)

    ordered_labels = []
    ordered_label_indices = []
    for chrom in ordered_keys:
        for l in labels_dict[chrom]:
            index, pos, ref, alt = l
            ordered_label_indices.append(index)
            ordered_labels.append((chrom, pos, ref, alt))
    return (ordered_labels, ordered_label_indices)


def _label_tuple_to_text(label, diff, genes=None):
    """
    Converts the variant label tuple to a string.

    Parameters
    ----------
    label : tuple(str)
        A tuple of (chrom, pos, ref, alt).
    diff : float
        The max difference score across some or all features in the model.
    genes : list(str) or None, optional
        Default is None. If the closest protein-coding `genes` are specified,
        will display them in the label text.

    Returns
    -------
    str
        The label text.

    """
    chrom, pos, ref, alt = label
    if genes is not None:
        if len(genes) == 0:
            genes_str = "none found"
        else:
            genes_str = ', '.join(genes)
        text = ("max diff score: {0}<br />{1} {2}, {3}/{4}<br />"
                "closest protein-coding gene(s): {5}").format(
            diff, chrom, pos, ref, alt, genes_str)
    else:
        text = "max diff score: {0}<br />{1} {2}, {3}/{4}".format(
            diff, chrom, pos, ref, alt)
    return text


def _variant_closest_genes(label, tabix_fh, chrs_gene_intervals):
    chrom = label[0]
    pos = label[1]
    closest_genes = []
    try:
        overlaps = tabix_fh.query(chrom, pos, pos + 1)
        for o in overlaps:
            closest_genes.append(o[-1])
    except tabix.TabixError:
        pass
    if len(closest_genes) != 0:
        return closest_genes
    gene_intervals = chrs_gene_intervals[chrom]
    closest = None
    for (start, end, strand, gene) in gene_intervals:
        if start < pos and closest and closest == pos - end:
            closest_genes.append(gene)
        elif start < pos and (closest is None
                or closest > pos - end):
            closest = pos - end
            closest_genes = [gene]
        elif start > pos and closest and closest == start - pos:
            closest_genes.append(gene)
        elif start > pos and (closest is None
                or closest > start - pos):
            closest = start - pos
            closest_genes = [gene]
    return closest_genes


def _load_chrs_gene_intervals(gene_intervals_bed):
    chrs_gene_intervals = defaultdict(list)
    with open(gene_intervals_bed, 'r') as file_handle:
        for line in file_handle:
            cols = line.strip().split('\t')
            chrom = cols[0]
            start = int(cols[1])
            end = int(cols[2])
            strand = cols[3]
            gene = cols[4]
            chrs_gene_intervals[chrom].append((start, end, strand, gene))
    return chrs_gene_intervals


def _variants_closest_protein_coding_gene(labels, version="hg38"):
    gene_intervals_tabix = pkg_resources.resource_filename(
        "selene_sdk",
        ("interpret/data/gencode_v28_{0}/"
         "protein_coding_l12_genes.bed.gz").format(version))
    gene_intervals_bed = pkg_resources.resource_filename(
        "selene_sdk",
        ("interpret/data/gencode_v28_{0}/"
         "protein_coding_l12_genes.bed").format(version))

    tabix_fh = tabix.open(gene_intervals_tabix)
    chrs_gene_intervals = _load_chrs_gene_intervals(gene_intervals_bed)

    labels_gene_information = []
    for l in labels:
        labels_gene_information.append(_variant_closest_genes(
            l, tabix_fh, chrs_gene_intervals))

    return labels_gene_information


def variant_diffs_scatter_plot(data,
                               labels,
                               features,
                               output_path,
                               filter_features=None,
                               labels_sort_fn=ordered_variants_and_indices,
                               nth_percentile=None,
                               hg_reference_version=None,
                               threshold_line=None,
                               auto_open=False):
    """
    Displays each variant's max probability difference across features
    as a point in a scatter plot. The points in the scatter plot are
    ordered by the variant chromosome and position by default. Variants can
    be sorted differently by passing in a new `labels_sort_fn`.

    Parameters
    ----------
    data : np.ndarray
        Absolute difference scores for variants across all features that a
        model predicts. This is the first value in the tuple returned by
        `load_variant_abs_diff_scores`.
    labels : list(tuple(str))
        A list of variant labels. This is the second value in the tuple
        returned by `load_variant_abs_diff_scores`.
    features : list(str)
        A list of the features the model predicts. This is the third value
        in the tuple returned by `load_variant_abs_diff_scores`.
    output_path : str
        Path to output file. Must have '.html' extension.
    filter_features : types.FunctionType or None, optional
        Default is None. A function that takes in a `list(str)` of features
        and returns the `list(int)` of feature indices over which we would
        compute the max(probability difference) for each variant. For example,
        a user may only want to visualize the max probability difference
        for TF binding features. If `None`, uses all the features.
    labels_sort_fn : types.FunctionType, optional
        Default is `ordered_variants_and_indices`. A function that takes
        in a `list(tuple(str))` of labels corresponding to the rows in `data`
        and returns a `tuple(list(tuple), list(int))`, where the first value
        is the ordered list of variant labels and the second value is the
        ordered list of indices for those variant labels. By default,
        variants are sorted by chromosome and position.
    nth_percentile : int [0, 100] or None, optional
        Default is None. If `nth_percentile` is not None, only displays the
        variants with a max absolute difference score within the
        `nth_percentile` of scores.
    hg_reference_version : str {"hg19", "hg38"} or None, optional
        Default is None. On hover, we can display the gene(s) closest to
        each variant if `hg_reference_version` is not None, where closest
        can be a variant within a gene interval (where genes and
        their coordinates are taken from level 1 & 2 protein-coding genes
        in gencode v28) or near a gene. In the future, we will allow users
        to specify their own genome file so that this information can
        be annotated to variants from other organisms, other genome versions,
        etc.
    threshold_line : float or None, optional
        Default is None. If `threshold_line` is not None, draws a horizontal
        line at the specified threshold. Helps focus the visual on variants
        above a certain threshold.
    auto_open : bool, optional
        Default is False. If `auto_open`, will automatically open a web
        browser that displays the plotted HTML file.

    Returns
    -------
    plotly.graph_objs.graph_objs.Figure
        The generated Plotly figure.

    """
    labels_ordered, label_indices = labels_sort_fn(labels)
    variant_closest_genes = None
    if hg_reference_version is not None:
        variant_closest_genes = _variants_closest_protein_coding_gene(
            labels_ordered, version=hg_reference_version)
    ordered_data = data[label_indices, :]

    feature_indices = None
    if filter_features is not None:
        feature_indices = filter_features(features)
        ordered_data = data[:, feature_indices]
    variants_max_diff = np.amax(ordered_data, axis=1)

    display_labels = None
    if nth_percentile:
        p = np.percentile(variants_max_diff, nth_percentile)
        keep = np.where(variants_max_diff >= p)[0]
        print("{0} variants with max abs diff score above {1} are in the "
              "{2}th percentile.".format(
                  len(keep), p, nth_percentile))
        variants_max_diff = variants_max_diff[keep]
        display_labels = []
        for i, l in enumerate(labels_ordered):
            if i not in keep:
                continue
            display_labels.append(l)
    else:
        display_labels = labels_ordered

    if variant_closest_genes:
        text_labels = [
            _label_tuple_to_text(l, d, g) for l, d, g in
            zip(display_labels, variants_max_diff, variant_closest_genes)]
    else:
        text_labels = [_label_tuple_to_text(l, d) for l, d in
                       zip(display_labels, variants_max_diff)]

    label_x = [' '.join([l[0], str(l[1])]) for l in display_labels]
    data = [go.Scatter(x=label_x,
                       y=variants_max_diff,
                       mode='markers',
                       marker = dict(
                           color = "#39CCCC",
                           line = dict(width = 1)),
                       text=text_labels,
                       hoverinfo="text")]

    go_layout = {
        "title": "Max probability difference scores",
        "hovermode": "closest",
        "hoverlabel": {
                "font": {"size": 16}
            },
        "xaxis": {
            "title": "Genome coordinates",
            "showticklabels": True,
            "tickangle": 35,
            "titlefont": {"family": "Arial, sans-serif",
                          "size": 16},
            "nticks": 25,
            "tickmode": "auto",
            "automargin": True
            },
        "yaxis": {
            "title": "Absolute difference",
            "titlefont": {"family": "Arial, sans-serif",
                          "size": 16}
            }
        }
    if isinstance(threshold_line, float):
        layout = go.Layout(
            **go_layout,
            shapes=[
                {"type": "line",
                 "x0": label_x[0],
                 "y0": threshold_line,
                 "x1": label_x[-1],
                 "y1": threshold_line,
                 "line": {
                    "color": "rgba(255, 99, 71, 1)",
                    "width": 2
                    }
                }
            ])
    else:
        layout = go.Layout(**go_layout)
    fig = go.Figure(data=data, layout=layout)
    path, filename = os.path.split(output_path)
    os.makedirs(path, exist_ok=True)
    plot(fig, filename=output_path, auto_open=auto_open)
    return fig
