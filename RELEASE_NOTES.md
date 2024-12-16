# Release notes 

This is a document describing new functionality, bug fixes, breaking changes, etc. associated with Selene version releases from v0.5.0 onwards. 

## Version 0.6.0
- `config_utils.py`: Add additional information saved upon running Selene. Specifically, we now save the version of Selene that the latest run used, make a copy of the input configuration file, and save this along with the model architecture file in the output directory. This adds a new dependency to Selene, the package `ruamel.yaml`
- `H5Dataloader` and `_H5Dataset`: Previously `H5Dataloader` had a number of arguments that were used to then initialize `_H5Dataset` internally. One major change in this version is that we now ask that users initialize `_H5Dataset` explicitly and then pass it to `H5Dataloader` as a class argument. This makes the two classes consistent with the PyTorch specifications for `Dataset` and `DataLoader` classes, enabling them to be compatible with different data parallelization configurations supported by PyTorch and the PyTorch Lightning framework.
- `_H5Dataset` class initialization optional arguments:
	- `unpackbits` can now be specified separately for sequences and targets by way of `unpackbits_seq` and `unpackbits_tgt`
	- `use_seq_len` enables subsetting to the center `use_seq_len` length of the sequences in the dataset.
	- `shift` (particularly paired with `use_seq_len`) allows for retrieving sequences shifted from the center position by `shift` bases. Note currently `shift` only allows shifting in one direction, depending on whether you pass in a positive or negative integer.
- `GenomicFeaturesH5`: This is a new targets class to handle continuous-valued targets, stored in an HDF5 file, that can be retrieved based on genomic coordinate. Once again, genomic regions are stored in a tabix-indexed .bed file, with the main change being that the BED file now specifies for each genomic regions the index of the row in the HDF5 matrix that contains all the target values to predict. If multiple target rows are returned for a query region, the average of those rows is returned.
- `RandomPositionsSampler`:
	- `exclude_chrs`: Added a new optional argument which by default excludes all nonstandard chromosomes `exclude_chrs=['_']` by ignoring all chromosomes with an underscore in the name. Pass in a list of chromosomes or substrings to exclude. When loading possible sampling positions, the class now iterates through the `exclude_chrs` list and checks for each substring `s` in list if `s in chrom`, and if so, skips that chromosome entirely.
	- Internal function `_retrieve` now takes in an optional argument `strand` (default `None`) to enable explicit retrieval of a sequence at `chrom, position` for a specific side. The default behavior of the `RandomPositionsSampler` class remains the same, with the strand side randomly selected for each genomic position sampled.
- `PerformanceMetrics`:
	- Now supports `spearmanr` and `pearsonr` from `scipy.stats`. Room for improvement to generalize this class in the future.
	- The `update` function now has an optional argument `scores` which can pass in a subset of the metrics as `list(str)` to compute.
- `TrainModel`:
	- `self.step` starts from `self._start_step`, which is non-zero if loaded from a Selene-saved checkpoint
	- removed call to `self._test_metrics.visualize` in `evaluate` since the visualize method does not generalize well.
- `NonStrandSpecific`: Can now handle a model outputting two outputs in `forward`, will handle by taking either the mean or max of each of the two individual outputs for their forward and reverse predictions. A custom `NonStrandSpecific` class is recommended for more specific cases.


## Version 0.5.3
- Adjust dependency requirements 

## Version 0.5.2
- Fix Cython type error causing build issues with Python 3.9+

## Version 0.5.1
- PyTorch<=1.9 compatibility 

## Version 0.5.0

### New functionality
- `sampler.MultiSampler`: `MultiSampler` accepts any Selene sampler for each of the train, validation, and test partitions where previously `MultiFileSampler` only accepted `FileSampler`s. We will deprecate `MultiFileSampler` in our next major release. 
- `DataLoader`: Parallel data loading based on PyTorch's `DataLoader` class, which can be used with Selene's `MultiSampler` and `MultiFileSampler` class. (see: `sampler.SamplerDataLoader`, `sampler.H5DataLoader`) 
- To support parallelism via multiprocessing, the sampler that `SamplerDataLoader` used needs to be picklable. To enable this, opening file operations are delayed to when any method that needs the file is called. There is no change to the API and setting `init_unpicklable=True` in `__init__` for `Genome` and all `OnlineSampler` classes will fully reproduce the functionality in `selene_sdk<=0.4.8`. 
- `sampler.RandomPositionsSampler`: added support for `center_bin_to_predict` taking in a list/tuple of two integers to specify the region from which to query the targets---that is, `center_bin_to_predict` by default (`center_bin_to_predict=<int>`) queries targets based on the center bin size, but can be specified as start and end integers that are not at the center if desired. 
- `EvaluateModel`: accepts a list of metrics (by default computing ROC AUC and average precision) with which to evaluate the test dataset. 

### Usage
- **Command-line interface (CLI)**: You can now run the CLI directly with `python -m selene_sdk` (if you have cloned the repository, make sure you have locally installed `selene_sdk` via `python setup.py install`, or `selene_sdk` is in the same directory as your script / added to `PYTHONPATH`). Developers can make a copy of the `selene_sdk/cli.py` script and use it the same way that `selene_cli.py` was used in earlier versions of Selene (`python -u cli.py <config-yml> [--lr]`) 

### Bug fixes
- `EvaluateModel`: `use_features_ord` allows you to evaluate a trained model on only a subset of chromatin features (targets) predicted by the model. If you are using a `FileSampler` for your test dataset, you now have the option to pass in a subsetted matrix; however, this matrix must be ordered the same way as `features` (the original targets prediction ordering) and not in the same ordering as `use_features_ord`. However, the final model predictions and targets
  (`test_predictions.npz` and `test_targets.npz`) will be outputted according to the `use_features_ord` list and ordering.
- `MatFileSampler`: Previously the `MatFileSampler` reset the pointer to the start of the matrix too early (going back to the first sample before we had finished sampling the whole matrix). 
- CLI learning rate: Edge cases (e.g. not specifying the learning rate via CLI or config) previously were not handled correctly and did not throw an informative error. 
