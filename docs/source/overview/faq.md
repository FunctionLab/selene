# Additional resources 

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
- [skorch: a scikit-learn compatible neural network library that wraps PyTorch](https://github.com/dnouri/skorch)
- [Tune: scalable hyperparameter search](https://ray.readthedocs.io/en/latest/tune.html)
- [weights & biases](https://www.wandb.com/)
- [comet.ml](https://www.comet.ml/)

In general, automating hyperparameter optimization requires greater computational proficiency. 
You would likely have to modify or insert some code into `selene_sdk.TrainModel` for these packages/tools to work with your model and data.


