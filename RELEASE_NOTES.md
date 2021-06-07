# Release notes 

This is a document describing new functionality, bug fixes, breaking changes, etc. associated with Selene version releases from v0.5.0 onwards. 

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
