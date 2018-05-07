from copy import deepcopy

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import transforms
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
import matplotlib.patheffects
from matplotlib.text import TextPath

from selene.sequences import Genome


class TextPathRenderingEffect(matplotlib.patheffects.AbstractPathEffect):
    """
    This is a class for re-rendering text paths and preserving their scale.
    """
    def __init__(self, bar, x_translation=0., y_translation=0., x_scale=1., y_scale=1.):
        """

        Parameters
        ----------
        bar : matplotlib.patches.Patch
            The patch where the letter is.
        x_translation : float, optional
            Default is 0.
            Amount by which to translate the x coordinate.
        y_translation : float, optional
            Default is 0.
            Amount by which to translate the y coordinate.
        x_scale : float, optional
            Default is 1.
            Amount by which to scale the width.
        y_scale : float, optional
            Default is 1.
            Amount by which to scale the height.
        """
        self._bar = bar
        self._x_translation = x_translation
        self._y_translation = y_translation
        self._x_scale = x_scale
        self._y_scale = y_scale

    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        """
        Redraws the path.
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


def sequence_logo(scores, order="value", sequence_type=Genome,
                  font_family="sans", width=1.0, font_size=180,
                  font_properties=None, ax=None, **kwargs):
    """

    Parameters
    ----------
    scores : np.ndarray
        A len(reference_sequence) x |ALPHABET| matrix containing the scores for each position.
    order : {"alpha", "value"}
        The ordering to use for the bases in the motif plots.
            alpha: Bases go in the order they are found in the sequence alphabet.
            value: Bases go in the order of their effect size, with the largest at the bottom.
    sequence_type : class, optional
        Default is selene.sequences.Genome
        The type of sequence that the ISM results are associated with.
    font_family : str, optional
        Default is `sans`.
        The font family to use. Availability of various families is controlled by system, not Selene.
    font_size : int, optional
        Default is 180.
        The size of the font to use.
    width : float, optional
        The default is 1.
        The size width of each character. A value of 1 will mean that there is no gap between each character.
    font_properties : matplotlib.font_manager.FontProperties, optional
        Default is None.
        A FontProperties object specifying the properties of the font used. If None is provided,
        a font property with the input font_size will be created.
    ax : matplotlib.pyplot.Axes, optional
        Default is None.
        An axes to plot on. If not provided, an axis will be created.
    color_scheme: list[str]
        A list containing the colors to use, appearing in the order of the bases of the sequence type.

    Returns
    -------
    matplotlib.pyplot.Axes
        An axis containing the sequence logo plot.

    """
    scores = deepcopy(scores)  # Everything will break if we do not deepcopy.
    scores = scores.transpose()

    if "colors" in kwargs:
        color_scheme = kwargs.pop("colors")
    else:
        color_scheme = ["orange", "red", "blue", "darkgreen"]
    if len(color_scheme) < len(sequence_type.BASES_ARR):
        raise ValueError("Color scheme is shorter than number of bases in sequence.")

    if scores.shape[0] != len(sequence_type.BASES_ARR):
        raise ValueError(f"Got score with {scores.shape[0]} bases for sequence"
                         f"with {len(sequence_type.BASES_ARR)} bases.")

    scores = np.flip(scores, axis=0)
    mpl.rcParams["font.family"] = font_family
    if font_properties is None:
        font_properties = FontProperties(size=font_size, weight="black")
    if ax is None:
        _, ax = plt.subplots(figsize=scores.shape)

    # Determine offsets depending on sort order.
    positive_offsets = np.zeros_like(scores)
    negative_offsets = np.zeros_like(scores)
    bases = np.empty(scores.shape, dtype=object)
    bases[:, :] = "?"  # Do not leave it as none. Should be visually obvious something happened.

    # Change ordering of things based on input arguments.
    if order == "alpha":
        for i in range(scores.shape[0]):
            bases[i, :] = sequence_type.BASES_ARR[i]

    elif order == "value":
        if np.sum(scores < 0) != 0:
            sorted_scores = np.zeros_like(scores)
            for j in range(scores.shape[1]):
                # Sort the negative values and put them at bottom.
                div = np.sum(scores[:, j] < 0.)
                negative_idx = np.argwhere(scores[:, j] < 0.).flatten()
                negative_sort_idx = np.argsort(scores[negative_idx, j], axis=None)
                sorted_scores[:div, j] = scores[negative_idx[negative_sort_idx], j]
                bases[:div, j] = sequence_type.BASES_ARR[negative_idx[negative_sort_idx]].flatten()

                # Sort the positive values and stack atop the negatives.
                positive_idx = np.argwhere(scores[:, j] >= 0.).flatten()
                positive_sort_idx = np.argsort(scores[positive_idx, j], axis=None)
                sorted_scores[div:, j] = scores[positive_idx[positive_sort_idx], j]
                bases[div:, j] = sequence_type.BASES_ARR[positive_idx[positive_sort_idx]].flatten()
            scores = sorted_scores
        else:
            for j in range(scores.shape[1]):
                sort_idx = np.argsort(scores[:, j], axis=None)[::-1]
                bases[:, j] = sequence_type.BASES_ARR[sort_idx]
                scores[:, j] = scores[sort_idx, j]

    # Create offsets for each bar.
    for i in range(scores.shape[0] - 1):
        y_coords = scores[i, :]
        if i > 0:
            negative_offsets[i + 1, :] = negative_offsets[i, :]
            positive_offsets[i + 1, :] = positive_offsets[i, :]
        neg_idx = np.argwhere(y_coords < 0.)
        pos_idx = np.argwhere(y_coords >= 0.)
        negative_offsets[i + 1, neg_idx] += y_coords[neg_idx]
        positive_offsets[i + 1, pos_idx] += y_coords[pos_idx]

    for i in range(scores.shape[0]):
        x_coords = np.arange(scores.shape[1]) + 0.5
        y_coords = scores[i, :]

        # Manage negatives and positives separately.
        offsets = np.zeros(scores.shape[1])
        negative_idx = np.argwhere(y_coords < 0.)
        positive_idx = np.argwhere(y_coords >= 0.)
        offsets[negative_idx] = negative_offsets[i, negative_idx]
        offsets[positive_idx] = positive_offsets[i, positive_idx]
        bars = ax.bar(x_coords, y_coords, color="black", width=width, bottom=offsets)
        for j, bar in enumerate(bars):
            base = bases[i, j]
            bar.set_color(color_scheme[sequence_type.BASE_TO_INDEX[base]])
            bar.set_edgecolor(None)

    # Iterate over the barplot's bars and turn them into letters.
    new_patches = []
    for i, bar in enumerate(ax.patches):
        base_idx = i // scores.shape[1]
        seq_idx = i % scores.shape[1]
        base = bases[base_idx, seq_idx]
        # We construct a text path that tracks the bars in the barplot.
        # Thus, the barplot takes care of scaling and translation, and we just copy it.
        text = TextPath((0., 0.), base, fontproperties=font_properties)
        b_x, b_y, b_w, b_h = bar.get_extents().bounds
        t_x, t_y, t_w, t_h = text.get_extents().bounds
        scale = (b_w / t_w, b_h / t_h)
        translation = (b_x - t_x, b_y - t_y)
        text = PathPatch(text, facecolor=bar.get_facecolor(), lw=0.)
        bar.set_facecolor("none")
        text.set_path_effects([TextPathRenderingEffect(bar)])  # This redraws the letters on resize.
        transform = transforms.Affine2D().translate(*translation).scale(*scale)
        text.set_transform(transform)
        new_patches.append(text)

    for patch in new_patches:
        ax.add_patch(patch)
    ax.set_xlim(0, scores.shape[1])
    return ax


def rescale_feature_matrix(scores, base_scaling="identity", position_scaling="identity"):
    """
    Performs base-wise and position-wise scaling of a feature matrix.

    Parameters
    ----------
    scores : numpy.ndarray
        A Lx|bases| matrix containing the scores for each position.
    base_scaling : {"identity", "probability", "max_effect"}
        The type of scaling performed on each base at a given position.
            identity : No transformation will be applied to the data.
            probability : The relative sizes of the bases will be the original input probabilities.
            max_effect : The relative sizes of the bases will be the max effect of the original input values.
    position_scaling : str
        The type of scaling performed on each position.
            identity: No transformation will be applied to the data.
            probability: The sum of values at a position will be equal to the
                         sum of the original input values at that position.
            max_effect: The sum of values at a position will be equal to the
                        sum of the max effect values of the original input
                        values at that position.
    kwargs : dict
        Passed to plot_sequence_logo

    Returns
    -------
    numpy.ndarray
        The transformed array.

    """
    scores = scores.transpose()
    rescaled_scores = scores

    # Scale individual bases.
    if base_scaling == "identity" or base_scaling == "probability":
        pass
    elif base_scaling == "max_effect":
        rescaled_scores = scores - np.min(scores, axis=0)
    else:
        raise ValueError(f"Could not find base scaling \"{base_scaling}\".")

    # Scale each position
    if position_scaling == "max_effect":
        max_effects = np.max(scores, axis=0) - np.min(scores, axis=0)
        rescaled_scores /= rescaled_scores.sum(axis=0)[np.newaxis, :]
        rescaled_scores *= max_effects[np.newaxis, :]
    elif position_scaling == "probability":
        rescaled_scores /= np.sum(scores, axis=0)[np.newaxis, :]
    elif position_scaling != "identity":
        raise ValueError(f"Could not find position scaling \"{position_scaling}\".")
    return rescaled_scores.transpose()


def heatmap(scores, sequence_type=Genome, mask=None, **kwargs):
    """
    Plots scores on a heatmap.

    Parameters
    ----------
    scores : numpy.ndarray
        A Lx|bases| matrix containing the scores for each position.
    sequence_type : class
        The type of sequence that the ISM results are associated with.
    mask : numpy.ndarray, None
        A Lx|bases| matrix containing 1s at positions to mask.
    kwargs : dict
        Keyword arguments to pass to seaborn.heatmap().
        Some useful ones are:
            cbar_kws: Change keyword arguments to the colorbar.
            yticklabels: Manipulate the tick labels on the y axis.
            cbar: If False, hide the colorbar, otherwise show the colorbar.
            cmap: The color map to use for the heatmap.

    Returns
    -------
    matplotlib.pytplot.Axes
        An axis containing the heatmap plot.

    """

    if mask is not None:
        mask = mask.transpose()
    scores = scores.transpose()
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
    return sns.heatmap(scores, mask=mask, yticklabels=yticklabels, cbar_kws=cbar_kws, cmap=cmap, **kwargs)
