
FAQ and additional resources
============================

Extending Selene
----------------

The main modules that users may want to extend are


* ``selene_sdk.samplers.OnlineSampler``
* ``selene_sdk.samplers.file_samplers.FileSampler``
* ``selene_sdk.sequences.Sequence``
* ``selene_sdk.targets.Target``

Please refer to the documentation for these classes.
If you are encounter a bug or have a feature request, please post to our Github `issues <https://github.com/FunctionLab/selene/issues>`_. E-mail kchen@flatironinstitute.org if you are interested in being a contributor to Selene.

Join our `Google group <https://groups.google.com/forum/#!forum/selene-sdk>`_ if you have questions about the package, case studies, or model development.

Exporting a Selene-trained model to Kipoi
-----------------------------------------

We have provided an example of how to prepare a model for upload to `Kipoi's model zoo <http://kipoi.org/>`_ using a model trained during case study 2. You can use `this example <https://github.com/FunctionLab/selene/tree/master/manuscript/case2/3_kipoi_export>`_ as a starting point for preparing your own model for Kipoi. We have provided a script that can help to automate parts of the process.

We are also working on an export function that will be built into Selene and accessible through the CLI. 

Hyperparameter optimization
---------------------------

Hyperparameter optimization is the process of finding the set of hyperparameters that yields an optimal model against a predefined score (e.g. minimizing a loss function). 
Hyperparameters are the variables that govern the training process (i.e. these parameters are constant during training, compared to model parameters which are optimized/"tuned" by the training process itself). 
Hyperparameter tuning works by running multiple trials of a single training run with different values for your chosen hyperparameters, set within some specified limit. Some examples of hyperparameters:


* learning rate
* number of hidden units
* convolutional kernel size

You can select hyperparameters yourself (manually) or automatically. 
For automatic hyperparameter optimization, you can look into grid search or random search. 

Some resources that may be useful:


* `Hyperopt: Distributed Asynchronous Hyper-parameter Optimization <https://github.com/hyperopt/hyperopt>`_
* `skorch: a scikit-learn compatible neural network library that wraps PyTorch <https://github.com/dnouri/skorch>`_
* `Tune: scalable hyperparameter search <https://ray.readthedocs.io/en/latest/tune.html>`_
* `Spearmint <https://github.com/JasperSnoek/spearmint>`_
* `weights & biases <https://www.wandb.com/>`_
* `comet.ml <https://www.comet.ml/>`_

To use hyperparameter optimization on models being developed with Selene, you could implement a method that runs Selene (via a command-line call) with a set of hyperparameters and then monitors the validation performance based on the output to ``selene_sdk.train_model.validation.txt``. 
