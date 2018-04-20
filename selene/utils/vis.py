import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import transforms
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch, Rectangle
import matplotlib.patheffects
from matplotlib.text import TextPath
import numpy as np


def get_matrix_from_in_silico_mutagenesis_results(mut_encs, mut_preds, ref_enc, ref_pred=None):
    """
    Turns the results of an in silico saturated mutagenesis into a matrix, where each row is a base
    and each column is a position in the sequence. The value is the prediction value when that position
    in the sequence has been set to that base.

    If the reference prediction is not passed in (as is the case when we are doing effect relative to reference), the
    reference bases will be set to zero. If a reference prediction is passed in, its value will be used for all
    reference bases, and they will not be set to zero.

    :param ref_enc: Encoded unmutated sequence.
    :param ref_pred: Prediction for unmutated sequence. Leave as zero to not include these.
    :param mut_encs: Mutant sequences.
    :param mut_preds: Predictions for mutant sequences.
    :return: n*m matrix of predictions by base change.
    """
    # TODO (ecofer): Consider using numpy's masked array here.
    if ref_pred is not None:
        mat = ref_enc * ref_pred
    else:
        mat = np.zeros_like(ref_enc)
    for i in range(len(mut_preds)):
        tmp = (mut_encs[i] + ref_enc)
        tmp[tmp > 1] = 1.
        mat += (tmp - ref_enc) * mut_preds[i]
    return mat


def get_heatmap_from_in_silico_mutagenesis_results(mut_encs, mut_preds, ref_enc, base_arr=None, ref_pred=None):
    """
    # TODO: Change this so it can take in base sequences?
    Turns the results of an in silico saturated mutagenesis into a matrix, where each row is a base
    and each column is a position in the sequence. The value is the prediction value when that position
    in the sequence has been set to that base. Returns a plots of this as a heatmap.

    If the reference prediction is not passed in (as is the case when we are doing effect relative to reference), the
    reference bases will be masked in the resultant output. If a reference prediction is passed in, its value will be
    used for all reference bases, and they will not be masked in the output.

    :param ref_enc: Encoded unmutated sequence.
    :param ref_pred: Prediction for unmutated sequence. Leave as zero to not include these.
    :param mut_encs: Mutant sequences.
    :param mut_preds: Predictions for mutant sequences.
    :param base_arr: Bases to use as Y labels.
    :return: heatmap of in silico mutagenesis results matrix.
    """
    mat = get_matrix_from_in_silico_mutagenesis_results(mut_encs, mut_preds, ref_enc, ref_pred)
    if ref_pred is None:
        mask = ref_enc
    else:
        mask = None
    return sns.heatmap(mat, linewidths=0., yticklabels=base_arr, cmap="RdBu_r", mask=mask)


def get_sequence_logo_from_in_silico_mutagenesis_results(mut_encs, mut_preds, ref_enc, ref_pred, base_arr=None):
    """
    Produces a sequence logo from the results of the in silico mutagenesis.

    :param ref_enc: Encoded unmutated sequence.
    :param ref_pred: Prediction for unmutated sequence. Leave as zero to not include these.
    :param mut_encs: Mutant sequences.
    :param mut_preds: Predictions for mutant sequences.
    :param base_arr: Bases to use as Y labels.
    :return: n*m matrix of predictions by base change.
    :return: sequence logo of in silico mutagenesis results matrix.
    """
    pass


class TextPathRenderingEffect(matplotlib.patheffects.AbstractPathEffect):
    """
    This is a class for re-rendering text paths and preserving their scale.
    """
    def __init__(self, bar):
        """
        Constructs a new TextRenderer.
        :param translation: Translation (tuple).
        :param scale: Scale (tuple).
        """
        self._bar = bar

    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        """
        Redraws the path.
        """
        b_x, b_y, b_w, b_h = self._bar.get_extents().bounds
        t_x, t_y, t_w, t_h = tpath.get_extents().bounds
        translation = (b_x - t_x, b_y - t_y)
        scale = (b_w / t_w, b_h / t_h)
        affine = affine.identity().scale(*scale).translate(*translation)
        renderer.draw_path(gc, tpath, affine, rgbFace)


def sequence_logo(scores, bases, font_family="Arial", font_size=80, width=1., font_properties=None):
    """
    Generates a sequence logo plot for input scored sequences.
    :param scores: Scores for each base in the sequence.
    :param bases: The bases in the sequence.
    :param font_family: The font style used for the sequence logo.
    :param font_size: The scale used for the fonts. If letters are being distorted, increase or decrease this.
    :param font_properties: A FontProperties object.
    :return: Sequence logo plot.
    """
    # TODO: Add more color schemes.
    COLOR_SCHEME = {'G': "orange",
                    'A': "red",
                    'C': "blue",
                    'T': "darkgreen"}
    mpl.rcParams["font.family"] = font_family
    figure, axis = plt.subplots(figsize=scores.shape)
    if font_properties is None:
        font_properties = FontProperties(weight="bold", size=font_size)
    else:
        font_properties.set_size(font_size)

    # Create stacked barplot, stacking after each base.
    for base_idx in range(scores.shape[0]):
        base = bases[base_idx]
        x_coords = np.arange(scores.shape[1]) + 1
        y_coords = scores[base_idx, :]
        if base_idx == 0:
            bottoms = np.zeros_like(y_coords)
        else:
            bottoms = np.sum(scores[:base_idx, :], axis=0)
        plt.bar(x_coords, y_coords,
                color=COLOR_SCHEME[base],
                width=width,
                bottom=bottoms)

    # Iterate over the barplot's bars and turn them into letters.
    new_patches = []
    for i, bar in enumerate(axis.patches):
        base_idx = i // scores.shape[1]
        seq_idx = i % scores.shape[0]
        if base_idx == 0:
            bottom = 0
        else:
            bottom = scores[base_idx - 1, seq_idx]

        # We construct a text path that tracks the bars in the barplot.
        # Thus, the barplot takes care of scaling and translation, and we just copy it.
        text = TextPath((0., 0.), bases[base_idx],
                        fontproperties=font_properties)
        b_x, b_y, b_w, b_h = bar.get_extents().bounds
        t_x, t_y, t_w, t_h = text.get_extents().bounds
        translation = (b_x - t_x, b_y - t_y)
        scale = (b_w / t_w, b_h / t_h)
        text = PathPatch(text, facecolor=bar.get_facecolor(), lw=0.)
        bar.set_facecolor("none")
        text.set_path_effects([TextPathRenderingEffect(bar)]) # This redraws the letters on resize.
        transform = transforms.Affine2D().scale(*scale).translate(*translation)
        text.set_transform(transform)
        # bar.set_clip_path(text)
        new_patches.append(text)

    for patch in new_patches:
        axis.add_patch(patch)

    sns.despine(ax=axis, trim=True)
    return figure


if __name__ == "__main__":
    first_score = np.array([[0., 0., 0., 0., 0.],
                            [1., 4., 7., 10., 13.],
                            [2., 5., 8., 11., 14.],
                            [3., 6., 9., 12., 15.]])
    second_score = np.array([[7., 7., 7., 7., 7.],
                            [1., 4., 7., 10., 13.],
                            [2., 5., 8., 11., 14.],
                            [3., 6., 9., 12., 15.]])
    sequence_logo(second_score, ["G", "A", "C", "T"])
    plt.show()
