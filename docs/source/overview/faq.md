# FAQ and additional resources

## Extending Selene
The main modules that users may want to extend are
- `selene_sdk.samplers.OnlineSampler`
- `selene_sdk.samplers.file_samplers.FileSampler`
- `selene_sdk.sequences.Sequence`
- `selene_sdk.targets.Target`

Please refer to the documentation for these classes.
If you are interested in contributing new modules to Selene or requesting new functionality, please submit an issue to [Selene](https://github.com/FunctionLab/selene/issues) or e-mail <kchen@flatironinstitute.org>.

## Hyperparameter optimization
Hyperparameter optimization is the process of finding the set of hyperparameters that yields an optimal model against a predefined score (e.g. minimizing a loss function). 
Hyperparameters are the variables that govern the training process (i.e. these parameters are constant during training, compared to model parameters which are optimized/"tuned" by the training process itself). 
Hyperparameter tuning works by running multiple trials of a single training run with different values for your chosen hyperparameters, set within some specified limit. Some examples of hyperparameters:
- learning rate
- number of hidden units
- convolutional kernel size

You can select hyperparameters yourself (manually) or automatically. 
For automatic hyperparameter optimization, you can look into grid search or random search. 

Some resources that may be useful:
- [Hyperopt: Distributed Asynchronous Hyper-parameter Optimization](https://github.com/hyperopt/hyperopt)
- [skorch: a scikit-learn compatible neural network library that wraps PyTorch](https://github.com/dnouri/skorch)
- [Tune: scalable hyperparameter search](https://ray.readthedocs.io/en/latest/tune.html)
- [Spearmint](https://github.com/JasperSnoek/spearmint)
- [weights & biases](https://www.wandb.com/)
- [comet.ml](https://www.comet.ml/)

To use hyperparameter optimization on models being developed with Selene, you could implement a method that runs Selene (via a command-line call) with a set of hyperparameters and then monitors the validation performance based on the output to `selene_sdk.train_model.validation.txt`. 

