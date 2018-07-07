"""
Analyzing Complex Variants
==========================

In this tutorial, we will explore some of Selene's tools for exploring
and interpreting sequence predictions. We are generally interested in
models that predict various output labels for an input biological
sequence. For instance, we might want to predict histone marks from the
DNA sequence of a human genome. Beyond recapitulating biological
phenomena for known motifs, these models can make accurate predictions
for arbitrary sequences. An *in silico* mutagenesis experiment uses this
fact to estimate how mutations in a sequence might change our
predictions for it. In this tutorial, we will use selene's suite of
tools for *in silico* mutagenesis to predict the effects of single
nucleotide polymorphisms (SNPs).

**Prerequisites** \* `Getting Started with
Selene <https://selene.flatironinstitute.org/tutorials/getting_started_with_selene.html>`__
\* `Analyzing Mutations with Trained
Models <https://selene.flatironinstitute.org/tutorials/analyzing_mutations_with_trained_models.html>`__

Download the data
-----------------

After downloading the compressed data from `here <TODO>`__, extract it
in the current directory. The decompressed output should include the
data files ``sequences.fasta``, ``sequences.fasta.fai``,
``variants.vcf``, and ``distinct_features.txt``. We will analyze the
genomic sequences in ``sequences.fasta``. The second file,
``sequences.fasta.fai`` is a
`faidx <http://www.htslib.org/doc/faidx.html>`__ index for
``sequences.fasta``. The third file, ``variants.vcf`` is a `variant call
format (VCF) file <https://en.wikipedia.org/wiki/Variant_Call_Format>`__
with a list of SNPs, insertions, and deletions. We will predict the
effects of these variants later on in the tutorial. Lastly,
``distinct_features.txt`` contains the names of the labels predicted by
the model. There should also be two additional files, ``deepsea2.py``
and ``deepsea2_checkpoint.pth.tar``, containing the model class and a
model checkpoint for that class.

Load the trained model
----------------------

We now build the model and load the checkpoint. Note that if we do not
have access to a CUDA-enabled GPU, we add the ``map_location`` argument
to the call to ``torch.load``.

"""


######################################################################
# Variant effect prediction
# -------------------------
# 
# It is common to store genomic variants in a variant call format (VCF)
# file. As such, selene supports the ability to load variants from a VCF
# file and use them for *in silico* mutagenesis experiments.
# 

analysis = AnalyzeSequences()
analysis.variant_effect_prediction("variants.vcf",
                                   save_data=["diffs", "logits", "predictions"],
                                   indexed_fasta="hg38.fa",
                                   output_dir="./")


######################################################################
# KEEP THIS CODE:
# ===============
# 

# xticks = [x for x in ax.get_xticks() if int(x) % 5 == 0]
# ax.set_xticks(xticks)
# ax.set_xticklabels(map(int, xticks))
# selene.interpret.heatmap(m, cbar=False, ax=ax)
# selene.interpret.sequence_logo(m, ax=ax1)






# sequence_length = 20
# feature_matrix = ism_result.get_feature_matrix(feature_name)[:sequence_length, :]
# reference_sequence = Genome.sequence_to_encoding(ism_result.reference_sequence)[:sequence_length, :]

# figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
# mask=(reference_sequence > 0)
# feature_matrix = rescale_feature_matrix(feature_matrix, base_scaling="max_effect", position_scaling="max_effect")
# sequence_logo(feature_matrix, order="value", ax=ax1)
# ax1.set_xticklabels(list(ism_result.reference_sequence)[:sequence_length])
# ax1.set_yticklabels(["0", r"$1\times10^{-3}$",  r"$2\times10^{-3}$",  r"$3\times10^{-3}$", r"$4\times10^{-3}$",  r"$5\times10^{-3}$"])


# cbar_kws = {"use_gridspec": False, "location": "bottom"}
# heatmap(feature_matrix, mask=mask, cbar=True, ax=ax2, linewidths=0.5, cbar_kws=cbar_kws)
# ax2.xaxis.set_label_position('top')
# ax2.xaxis.tick_top()

# sns.despine(ax=ax1, trim=True)
# ref_mask_kwargs = dict(facecolor="#A1868A")
# ax2.patch.set(**ref_mask_kwargs)
# ax2.collections[0].colorbar.set_ticks([0, 0.5e-3, 1e-3, 1.5e-3, 2e-3, 2.5e-3])
# ax2.collections[0].colorbar.set_ticklabels(["0", r"$5\times10^{-4}$",  r"$1\times10^{-3}$",  r"$1.5\times10^{-3}$", r"$2\times10^{-3}$",  r"$2.5\times10^{-3}$"])
# ref_guide = Patch(**ref_mask_kwargs, label="Reference")
# ax2.collections[0].colorbar.ax.legend(loc="center left", ncol=2, bbox_to_anchor=(-0.009, 1.7),
#                                       handles=[ref_guide], labels=["Reference"],
#                                       fontsize=12)
# plt.show()


# figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=False)
# mask=(reference_sequence > 0)
# feature_matrix = rescale_feature_matrix(feature_matrix, base_scaling="max_effect", position_scaling="max_effect") - 0.001
# sequence_logo(feature_matrix, order="value", ax=ax1)

# ax1.set_xticks(np.asarray(range(len(ism_result.reference_sequence)))[:sequence_length] + 0.5)
# ax1.set_xticklabels(list(ism_result.reference_sequence)[:sequence_length])
# ax1.axhline(0., xmin=0.5/sequence_length, xmax=(sequence_length-0.5)/sequence_length, color="grey", linestyle="solid")
# sns.despine(ax=ax1, trim=True)
# heatmap(feature_matrix, mask=mask, cbar=True, ax=ax2)
# # heatmap(feature_matrix, mask=mask, cbar=True, ax=ax2)

# # plt.tight_layout()
# plt.show()