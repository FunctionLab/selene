"""This module provides the methods for visualizing different ouputs
from selene analysis methods.

"""
import re
import warnings
from copy import deepcopy

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.patheffects
from matplotlib.text import TextPath

from selene.sequences import Genome


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
                    " l -31.25,0 l -9.375,-25 l -25,0 z M 43.75, 50 l 12.5,0" +
                    " l -5.859375,15.625 l -5.859375,-15.625 z"),
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
    ax : matplotlib.pyplot.Axes, optional
        Default is `None`. The axes to plot on. If left as `None`, a new
        axis will be created.
    sequence_type : class, optional
        Default is `selene.sequences.Genome`. The type of sequence that
        the *in silico* mutagenesis results are associated with. This
        should generally be a subclass of `selene.sequences.Sequence`.
    font_properties : matplotlib.font_manager.FontProperties, optional
        Default is `None`. A `matplotlib.font_manager.FontProperties`
        object that specifies the properties of the font to use for
        plotting the motif. If `None`, no font will be used, and the
        text will be rendered by a path. This method of rendering paths
        is  preferred, as it ensures all character heights correspond to
        the actual values, and that there are no extra gaps between
        the tops and bottoms of characters at each position in the
        sequence logo. If the user opts to use a value other
        than `None`, then no such guarantee can be made.
    color_scheme : list(str), optional
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

    .. image:: ../../docs/source/_static/images/sequence_logo_example.png

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
    mask : numpy.ndarray, dtype=bool, optional
        Default is `None`. An :math:`L \\times N` array (where :math:`L`
        is the length of the sequence, and :math:`N` is the size of the
        alphabet)  containing `True` at positions in the heatmap to
        mask. If `None`, no masking will occur.
    sequence_type : class, optional
        Default is `selene.sequences.Genome`. The class of sequence that
        the *in silico* mutagenesis results are associated with. This is
        generally a sub-class of `selene.sequences.Sequence`.
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
        The axese containing the heatmap plot.

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
        cmap = "Blues_r"
    ret = sns.heatmap(score_matrix, mask=mask, yticklabels=yticklabels,
                      cbar_kws=cbar_kws, cmap=cmap, **kwargs)
    ret.set_yticklabels(labels=ret.get_yticklabels(), rotation=0)
    return ret
