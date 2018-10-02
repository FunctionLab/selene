# Case 3: Applying a new model to variants 

For this case study, we do the following:

1. Make variant effect predictions for variants in the [IGAP Alzheimer's disease age at onset survival GWAS (Huang et al., 2017)](https://www.niagads.org/datasets/ng00058). We use variants in 2 groups:

- Nominally significant: p-value < 0.05
- Nonsignificant: p-value > 0.50

We use the deeper DeepSEA architecture that we trained and evaluated in case 2 for this example.

2. Next, we determine whether the predicted effect for each feature is significantly higher for variants in the nominally significant variants group compared to the nonsignificant variants group. This would indicate that our model can prioritize disease-associated variants. 

Finally, the file `download_data.sh` will download all the data and outputs required to run each of these steps.
The rest of this README assumes that you have run this script.
Any directories mentioned in the README that are not included by default should have been downloaded using `download_data.sh`.
We have included comments in that file with more information about what is downloaded. 

## Step 1: variant effect prediction

We have provided an example SLURM script [`variant_effect_prediction.sh`](https://github.com/FunctionLab/selene/blob/master/manuscript/case3/1_variant_effect_prediction/variant_effect_prediction.sh) that shows how we run variant effect prediction on a GPU node.
In this case, we are running Selene using [`../../../selene_cli.py`](https://github.com/FunctionLab/selene/blob/master/selene_cli.py) and a configuration file.
Where it is now, the CLI script runs the local version of Selene (that is, it works if you clone the entire repository and build the Cython modules using `python setup.py build_ext --inplace`).
In this case, the `selene-env` conda environment activated in the `.sh` script does not contain `selene-sdk`; instead, it contains all the dependencies of Selene (see: [`selene-gpu.yml`](https://github.com/FunctionLab/selene/blob/master/selene-gpu.yml)) as well as the `docopt` package (which parses the arguments for the CLI).  
If you want to use the installed `selene-sdk` package (through conda or pip), you can just move the top-level `selene_cli.py` script outside of the repository and run the code for a specific case.

The configuration file, [`variant_effect_prediction.yml`](https://github.com/FunctionLab/selene/blob/master/manuscript/case3/1_variant_effect_prediction/variant_effect_prediction.yml) needs to be filled out with the absolute paths of all the data files before you are able to run training. 

Specifically for variant effect prediction, we specify that we want the `abs_diffs`--the absolute difference scores between reference and alternate alleles--to be outputted as a file for each of the 2 groups of variants. 

The outputs from variant effect prediction can be found in the directory `predict_outputs`. 

## Step 2: comparing the predicted effect for the 2 variant groups (nominally significant and nonsignificant)

In this step, we run the following Python scripts:

1. [`scores_as_npz.py`](https://github.com/FunctionLab/selene/blob/master/manuscript/case3/scores_as_npz.py) to convert the scores `.tsv` files generated from variant effect prediction to compressed NumPy matrix files. The `.tsv` files contain the row (variants) and column (features) labels for the matrix. 
2. [`variant_groups_comparison.py`](https://github.com/FunctionLab/selene/blob/master/manuscript/case3/variant_groups_comparison.py) to compare the predicted effect for the 2 variant groups across the genomic features predicted by the model. The following steps are taken:
   
   - For each genomic feature, get the p-value for the 1-sided Wilcoxon rank sum test for the 2 groups (we expect that the nominally significant variants will have a larger predicted effect, or absolute difference score).
   - FDR correct the p-values using the Benjamini-Hochberg correction, alpha = 0.05.
   - Because we expect the distribution of scores to be heavy-tailed and non-Gaussian, we then quantile normalize the scores against the Gaussian distribution. We do this because we want to visualize the mean and confidence intervals (95%) for the genomic feature where the difference in predicted effect is most significant (lowest q-value).
   - Output Figure 3b. We also save the q-values and mean/CI for the 2 different groups across all the features so that we can visualize this information for all genomic features if desired.

The outputs from this step can be found in the directory `comparison_outputs`. 
