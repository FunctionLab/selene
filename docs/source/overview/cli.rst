
Selene CLI operations and outputs
=================================

Selene provides a command-line interface (CLI) that takes in a user-specified configuration file containing the operations the user wants to run and the parameters required for these operations. It is automatically installed using a setuptools entrypoint so it can be called with ``python -m selene_sdk``. 

The sections that follow describe in detail how the various components that make up the configuration file are specified. For operation-specific sections (e.g. training, evaluation), we also explain what the expected outputs are.

We strongly recommend you read through the first 4 sections (:ref:`Overview`, :ref:`Operations`, :ref:`General configurations`, and :ref:`Model architecture`) and then pick other sections based on your use case. 

Join our `Google group <https://groups.google.com/forum/#!forum/selene-sdk>`_ if you have questions about the package, case studies, or model development.

Overview
--------

Selene's CLI accepts configuration files in the `YAML <https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html>`_ format that are composed of 4 main (high-level) groups:


#. list of operations
#. general configuration parameters
#. model configuration
#. operation-specific configurations

"Operation-specific configurations" require you to specify the input parameters for different classes and methods that we have implemented in Selene. These configurations are parsed using code adapted from the `Pylearn2 library <http://deeplearning.net/software/pylearn2/yaml_tutorial/index.html#yaml-tutorial>`_ and will instantiate the appropriate Python object or function based on your inputs. You may use Selene's `API documentation <https://selene.flatironinstitute.org>`_ to determine what parameters are accepted by the constructors/methods implemented in Selene. For your convenience, we have created this document to specifically describe the parameters necessary to build configuration files for the Selene CLI.

We recommend you start off by using one of the `example configuration files <https://github.com/FunctionLab/selene/tree/master/config_examples>`_ provided in the repository as a template for your own configuration file:


* `Training configuration <https://github.com/FunctionLab/selene/blob/master/config_examples/train.yml>`_
* `Evaluate with test BED file <https://github.com/FunctionLab/selene/blob/master/config_examples/evaluate_test_bed.yml>`_
* `Evaluate with test matrix file <https://github.com/FunctionLab/selene/blob/master/config_examples/evaluate_test_mat.yml>`_
* `Get predictions from trained model <https://github.com/FunctionLab/selene/blob/master/config_examples/get_predictions.yml>`_
* `\ *In silico* mutagenesis <https://github.com/FunctionLab/selene/blob/master/config_examples/in_silico_mutagenesis.yml>`_
* `Variant effect prediction <https://github.com/FunctionLab/selene/blob/master/config_examples/variant_effect_prediction.yml>`_

There are also various configuration files associated with the Jupyter notebook `tutorials <https://github.com/FunctionLab/selene/tree/master/tutorials>`_ and `manuscript <https://github.com/FunctionLab/selene/tree/master/manuscript>`_ case studies that you may use as a starting point.

Operations
----------

Every file should start with the operations that you want to run. 

.. code-block:: YAML

   ops: [train, evaluate, analyze]

The ``ops`` key expects one or more of ``[train, evaluate, analyze]`` to be specified as a list. In addition to the general and model architecture configurations described in the next 2 sections, each of these operations will require some additional set of configurations attached to the following keys:


* ``train``\ : ``train_model`` (see :ref:`Train`) and ``sampler`` (see :ref:`Samplers used for training (and evaluation, optionally)`)
* ``evaluate``\ : ``evaluate_model`` (see :ref:`Evaluate`) and ``sampler`` (see :ref:`Samplers used for evaluation`)
* ``analyze``\ : ``analyze_sequences`` (see :ref:`Analyze sequences`) 

**Note**\ : You should be able to use multiple operations (i.e. specify the necessary configuration keys for those operations in a single file). However, if ``[train, evaluate]`` are both specified, we expect that they will both rely on the same sampler. If you need to train and evaluate using different samplers, please create 2 separate YAML files. 

General configurations
----------------------

In addition to the ``ops`` key, you can specify the following parameters:

.. code-block:: YAML

   random_seed: 1337
   output_dir: /absolute/path/to/output/dir
   create_subdirectory: True
   lr: 0.01
   load_test_set: True

Note that there should not be any commas at the end of these lines.


* ``random_seed``\ : Set a random seed for ``torch`` and ``torch.cuda`` (if using CUDA-enabled GPUs) for reproducibility.
* ``output_dir``\ : The output directory to use for all operations. If no ``output_dir`` is specified, Selene assumes that the ``output_dir`` is specified in all relevant function-type values for operations in Selene. (More information on what function-type values are in later sections, see: :ref:`A note for the following sections`.) We recommend using this parameter for ``train`` and ``evaluate`` operations. 
* ``create_subdirectory``\ : If True, creates a directory within ``output_dir``   with the name formatted as ``%Y-%m-%d-%H-%M-%S``\ ---the date/time when Selene was run. (This is only applicable if ``output_dir`` has been specified.)
* ``lr``\ :  The learning rate. If you use the CLI (\ ``selene_sdk``\ ), you can pass this in as a command-line argument rather than having it specified in the configuration file. 
* ``load_test_set``: This is only applicable if you have specified `ops: [train, evaluate]`. You can set this parameter to True (by default it is False and the test set is only loaded when training ends) if you would like to load the test set into memory before training begins---and therefore save the test data generated by a sampler to a .bed file. You would find this useful if you want to save a test dataset and you do not know if your model will finish training and evaluation within the allotted time that your job is run. You should also be running Selene on a machine that can support such an increase in memory usage (on the order of GBs, depending on how many classes your model predicts, how large the test dataset is, etc.). 

Model architecture
------------------

For all operations, Selene requires that you specify the model architecture, loss, and optimizer as inputs.

Expected input class and methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two possible formats you can use to do this:


* 
  single Python file: We expect that most people will start using Selene with model architectures in this format. In this case, you implement your architecture as a class and include 2 static methods, ``criterion`` and ``get_optimizer`` in the same file. See our `DeepSEA model file <https://github.com/FunctionLab/selene/blob/master/models/deepsea.py>`_ as an example. 


  * The ``criterion`` method should not take any input arguments and must return a loss function object of type ``torch.nn._Loss``. 
  * The ``get_optimizer`` method should accept a single input ``lr``\ , the learning rate. (Note that this method is not used for the ``evaluate`` and ``analyze`` operations in Selene.) It returns a tuple, where ``tuple[0]`` is the optimizer class ``torch.optim.Optimizer`` and ``tuple[1]`` is a dictionary of any optional arguments with which Selene can then instantiate the class. Selene will first instantiate the model and then pass the required ``model.parameters()`` argument as input to the ``torch.optim.Optimizer`` class constructor.

* 
  Python module: For more complicated architectures, you may want to write custom PyTorch modules and use them in your final architecture. In this case, it is likely your model architecture imports other custom classes. We ask that you then specify your architecture within a Python module. That is, the directory containing your architecture, loss, and optimizer must have a ``__init__.py`` that imports the architecture class, ``criterion``\ , and ``get_optimizer``. 

Model architecture configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: YAML

   model: {
       path: /absolute/path/to/file_or_model,
       class: ModelArchitectureClassName,
       class_args: {
           arg1: val1,
           arg2: val2,
           ...
       },
       non_strand_specific: mean
   }


* ``path``\ : This can be the path to a Python file or a Python module (directory). See the `previous section <#expected-input-class-and-methods>`_ for details.
* ``class``\ : The model architecture class name.
* ``class_args``: The arguments needed to instantiate the class. In the case of `DeepSEA <https://github.com/FunctionLab/selene/blob/master/models/deepsea.py>`_, the ``class_args`` keys would be ``sequence_length`` and ``n_genomic_features``. 
* ``non_strand_specific``\ : Optional, possible values are ``mean`` or ``max`` if you want to use this parameter. (Otherwise, do not use this key in your model configuration.) If your model does not need to train on strand-specific input sequences, we have implemented a class that will pass both the forward and reverse sequence to the model and either take the ``mean`` or the ``max`` value across the two sets of predictions for a sample. 

A note for the following sections
---------------------------------

For training, evaluation, and analysis [of sequences using trained models], Selene requires that specific keys in the YAML file correspond to function-type values. The function-type value is used to construct an object that is a class in ``selene_sdk``. Our `documentation website <https://selene.flatironinstitute.org/>`_ is an important resource for debugging configuration-related errors when you run Selene via the CLI. 

We have covered the most common configurations in this document.

Train
-----

An example configuration for training:

.. code-block:: YAML

   train_model: !obj:selene_sdk.TrainModel {
       batch_size: 64,
       max_steps: 960000,
       report_stats_every_n_steps: 32000,
       save_checkpoint_every_n_steps: 1000,
       save_new_checkpoints_after_n_steps: 640000,
       n_validation_samples: 64000,
       n_test_samples: 960000,
       cpu_n_threads: 32,
       use_cuda: True,
       data_parallel: True,
       logging_verbosity: 2,
       metrics: {
           roc_auc: !import sklearn.metrics.roc_auc_score,
           average_precision: !import sklearn.metrics.average_precision_score
       },
       checkpoint_resume: False    
   }

Required parameters
^^^^^^^^^^^^^^^^^^^


* ``batch_size``\ :  Number of samples in one forward/backward pass (a single step).
* ``max_steps``\ : Total number of steps for which to train the model. 
* ``report_stats_every_n_steps``\ : The frequency with which to report summary statistics. You can set this value to be equivalent to a training epoch (\ ``n_steps * batch_size``\ ) being the total number of samples seen by the model so far. Selene evaluates the model on the validation dataset every ``report_stats_every_n_steps`` and, if the model obtains the best performance so far (based on the user-specified loss function), Selene saves the model state to a file called ``best_model.pth.tar`` in ``output_dir``.  

Optional parameters
^^^^^^^^^^^^^^^^^^^


* ``save_checkpoint_every_n_steps``\ : Default is 1000. The number of steps before Selene saves a new checkpoint model weights file. If this parameter is set to ``None``\ , we will set it to the same value as ``report_stats_every_n_steps``.
* ``save_new_checkpoints_after_n_steps``\ : Default is None. The number of steps after which Selene will continually save new checkpoint model weights files (\ ``checkpoint-<TIMESTAMP>.pth.tar``\ ) every ``save_checkpoint_every_n_steps``. Before this, the file ``checkpoint.pth.tar`` is overwritten every ``save_checkpoint_every_n_steps`` to limit the memory requirements.
* ``n_validation_samples``\ : Default is ``None``. Specify the number of validation samples in the validation set. If ``None``

  * and the data sampler you use is of type ``selene_sdk.samplers.OnlineSampler``\ , we will by default retrieve 32000 validation samples.
  * and you are using a ``selene_sdk.samplers.MultiFileSampler``\ , we will use all the validation samples available in the appropriate data file.

* 
  ``n_test_samples``\ : Default is ``None``. Specify the number of test samples in the test set. If ``None`` and


  * the sampler you specified has no test partition, you should not specify ``evaluate`` as one of the operations in the ``ops`` list. That is, Selene will not automatically evaluate your trained model on a test dataset, because the sampler you are using does not have any test data. 
  * the sampler you use is of type ``selene_sdk.samplers.OnlineSampler`` (and the test partition exists), we will retrieve 640000 test samples.
  * 
    the sampler you use is of type ``selene_sdk.samplers.MultiFileSampler`` (and the test partition exists), we will use all the test samples available in the appropriate data file.

    You can review the section on samplers for more information.

* ``cpu_n_threads``\ : Default is 1. The number of OpenMP threads used for parallelizing CPU operations in PyTorch.
* ``use_cuda``\ : Default is False. Specify whether CUDA-enabled GPUs are available for torch to use during training.  
* ``data_parallel``\ : Default is False. Specify whether multiple GPUs are available for torch to use during training.
* ``logging_verbosity``\ : Default is 2. Possible values are ``{0, 1, 2}``. Sets the logging verbosity level:

  * 0: only warnings are logged
  * 1: information and warnings are logged
  * 2: debug messages, information, and warnings are all logged

* ``metrics``: Default is a dictionary with `"roc_auc"` mapped to ``sklearn.metrics.roc_auc_score`` and `"average_precision"` mapped to ``sklearn.metrics.average_precision_score``. ``metrics`` is a dictionary that maps metric names (`str`) to metric functions. In addition to the loss function you specified with your :ref:`Model architecture`, these are the metrics that you would like to monitor during the training/evaluation process (they all get reported every ``report_stats_every_n_steps``). See the `Regression Models in Selene <https://github.com/FunctionLab/selene/blob/master/tutorials/regression_mpra_example/regression_mpra_example.ipynb>`_ tutorial for a different input to the ``metrics`` parameter. You can ``!import`` metrics from ``scipy``\ , ``scikit-learn``\ , ``statsmodels``. Each metric function should require, in order, the true values and predicted values as input arguments. For example,
  `sklearn.metrics.average_precision_score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html>`_ takes ``y_true`` and ``y_score`` as input.  

* ``checkpoint_resume``\ : Default is ``None``. If not ``None``\ , you should pass in the path to a model weights file generated by ``torch.save`` (and can now be read by ``torch.load``\ ) to resume training.  

Additional notes
^^^^^^^^^^^^^^^^

Attentive readers might have noticed that in the documentation for the `TrainModel class <https://selene.flatironinstitute.org/selene.html#trainmodel>`_ there are more input arguments than are required to instantiate the class through the CLI configuration file. This is because they are assumed to be carried through/retrieved from other configuration keys for consistency. Specifically:


* ``output_dir`` can be specified as a top-level key in the configuration. You can specify it within each function-type constructor (e.g.  ``!obj:selene_sdk.TrainModel``\ ) if you prefer. If ``output_dir`` exists as a top-level key, Selene does use the top-level ``output_dir`` and ignores all other ``output_dir`` keys. ``output_dir`` is omitted in many of the configurations for this reason.
* ``model``\ , ``loss_criterion``\ , ``optimizer_class``\ , ``optimizer_kwargs`` are all retrieved from the path in the :ref:`Model architecture` configuration. 
* ``data_sampler``\ has its own separate configuration that you will need to specify in the same YAML file. Please see :ref:`Sampler configurations` for more information.

Expected outputs for training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These outputs will be written to ``output_dir`` (a top-level parameter, can also  be specified within the function-type constructor, see above).


* ``best_model.pth.tar``: the best performing model so far. IMPORTANT: for all ``*.pth.tar`` files output by Selene right now, we save additional information beyond the model's state dictionary so that users may continue training these models through Selene if they wish. If you would like to save only the state dictionary, you can run ``out = torch.load(<*.pth.tar>)`` and then save only the ``state_dict`` key with ``torch.save(out["state_dict"], <state_dict_only.pth.tar>)``. 
* ``checkpoint.pth.tar``\ : model saved every ``save_checkpoint_every_n_steps`` steps
* ``selene_sdk.train_model.log``\ : a detailed log file containing information about how much time it takes for batches to sampled and propagated through the model, how the model is performing, etc.
* ``selene_sdk.train_model.train.txt``\ : model training loss is printed to this file every ``report_stats_every_n_steps``.

  * Visualize using ``matplotlib`` (\ ``plt.plot``\ )

* ``selene_sdk.train_model.validation.txt``\ : model validation loss and other metrics you have specified (defaults would be ROC AUC and AUPRC) are printed to this file (tab-separated) every ``report_stats_every_n_steps``. 

  * Visualize one of these columns using ``matplotlib`` (\ ``plt.plot``\ )

* saved sampled datasets (if applicable), e.g. ``test_data.bed``\ : if the ``save_datasets`` value is not an empty list, Selene periodically saves all the data sampled so far in these .bed files. The columns of these files are ``[chr, start, end, strand, semicolon_separated_class_indices]``. In the future, we will adjust this file to support non-binary labels (i.e. since we are only storing class indices in these output .bed files, we can only label sequences with 1/0, presence/absence, of a given class).

Evaluate
--------

An example configuration for evaluation:

.. code-block:: YAML

   evaluate_model: !obj:selene_sdk.EvaluateModel {
       features: !obj:selene_sdk.utils.load_features_list {
           input_path: /path/to/features_list.txt
       },
       use_features_ord: !obj:selene_sdk.utils.load_features_list {
           input_path: /path/to/features_subset_ordered.txt
       },
       trained_model_path: /path/to/trained/model.pth.tar,
       batch_size: 64,
       n_test_samples: 640000,
       report_gt_feature_n_positives: 50,
       use_cuda: True
   }

Required parameters
^^^^^^^^^^^^^^^^^^^


* ``features``\ : The list of distinct features the model predicts. (\ ``input_path`` to the function-type value that loads the features as a list.)
* ``trained_model_path``\ : Path to the trained model weights file, which should have been generated/saved using ``torch.save``. (i.e. you can pass in the saved model file generated by Selene's ``TrainModel`` class.)

Optional parameters
^^^^^^^^^^^^^^^^^^^


* ``batch_size``\ : Default is 64. Specify the batch size to process examples. Should be a power of 2.
* ``n_test_samples``\ : Default is ``None``. Use ``n_test_samples`` if you want to limit the number of samples on which you evaluate your model. If you are using a sampler of type ``selene_sdk.samplers.OnlineSampler``---you must specify a test partition in this case---it will default to 640000 test samples if ``n_test_samples = None``. If you are using a file sampler (:ref:`Multiple-file sampler`, :ref:`BED file sampler`, or :ref:`Matrix file sampler`), it will use all samples available in the file.
* ``report_gt_feature_n_positives``\ : Default is 10. In total, each class/feature must have more than ``report_gt_feature_n_positives`` positive examples in the test set to be considered in the performance computation. The output file that reports each class's performance will report 'NA' for classes that do not have enough positive samples.
* ``use_cuda``\ : Default is False. Specify whether CUDA-enabled GPUs are available for torch to use.  
* ``data_parallel``\ : Default is False. Specify whether multiple GPUs are available for torch to use.
* ``use_features_ord``\ : Default is None. Specify an ordered list of features for which to run the evaluation. The features in this list must be identical to or a subset of ``features``\ , and in the order you want the resulting ``test_targets.npz`` and ``test_predictions.npz`` to be saved.

Additional notes
^^^^^^^^^^^^^^^^

Similar to the ``train_model`` configuration, any arguments that you find in the `EvaluateModel <https://selene.flatironinstitute.org/selene.html#evaluatemodel>`_ documentation that are not present in the function-type value's arguments are automatically instantiated and passed in by Selene.

If you use a sampler with multiple data partitions with the ``evaluate_model`` configuration, please make sure that your sampler configuration's ``mode`` parameter is set to ``test``. 

Expected outputs for evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These outputs will be written to ``output_dir`` (a top-level parameter, can also  be specified within the function-type constructor).


* ``test_performance.txt``: columns are ``class`` and whatever other metrics you specified (defaults: ``roc_auc`` and ``average_precision``\ ). The breakdown of performance metrics by each class that the model predicts.
* ``test_predictions.npz``\ : The model predictions for each sample in the test set. Useful if you want to make your own visualizations/figures.
* ``test_targets.npz``\ : The actual classes for each sample in the test set. Useful if you want to make your own visualizations/figures.
* ``precision_recall_curves.svg``\ : If using AUPRC as a metric, this is an AUPRC figure that we generate for you. Each curve corresponds to one of the classes the model predicts.
* ``roc_curves.svg``\ : If using ROC AUC as a metric, this is an ROC AUC figure that we generate for you. Each curve corresponds to one of the classes the model predicts.
* ``selene_sdk.evaluate_model.log``: Note that if ``evaluate`` is run through ``TrainModel`` (that is, no ``EvaluateModel`` configuration was specified, but you used ``ops: [train, evaluate]``) you will only see ``selene_sdk.train_model.log``. ``selene_sdk.evaluate_model.log`` is only created when ``EvaluateModel`` is initialized. 

Analyze sequences
-----------------

The ``analyze`` operation allows you to apply a trained model to new sequences of interest. Currently, we support 3 "sub-operations" for ``analyze``\ :

1) Prediction on sequences: Output the model predictions for a list of sequences.
2) Variant effect prediction: Output the model predictions for sequences centered on specific variants (will output reference and alternate predictions as separate files). 
3) *In silico* mutagenesis: *In silico* mutagenesis (ISM) involves computationally "mutating" every position in the sequence to every other possible base (DNA and RNA) or amino acid (protein sequences) and examining the consequences of these "mutations". For ISM, Selene outputs the model predictions for the reference (original) sequence along with each of the mutated sequences. 

For variant effect prediction and *in silico* mutagenesis, a number of scores can be computed using the predictions from the reference and alternate alleles. You may select 1 or more of the following as outputs:


* ``predictions`` (output the predictions for each variant, as described above)
* ``diffs`` (difference scores): The difference between alt and ref predictions.
* ``abs_diffs`` (absolute difference scores): The absolute difference between alt and ref predictions.
* ``logits`` (log-fold change scores): The difference between ``logit(alt)`` and ``logit(ref)`` predictions.

You'll find examples of how these scores are specified in the :ref:`Variant effect prediction` and :ref:`In silico mutagenesis` sections. 

In all ``analyze``\ -related operations, we ask that you specify 2 configuration keys. One will always be the ``analyze_sequences`` key and the other one is dependent on which of the 3 sub-operations you use: ``prediction``\ , ``variant_effect_prediction`` or ``in_silico_mutagenesis``.

.. code-block:: YAML

   analyze_sequences: !obj:selene_sdk.predict.AnalyzeSequences {
       trained_model_path: /path/to/trained/model.pth.tar,
       sequence_length: 1000,
       features: !obj:selene_sdk.utils.load_features_list {
           input_path: /path/to/features_list.txt
       },
       batch_size: 64,
       use_cuda: False,
       reference_sequence: !obj:selene_sdk.sequences.Genome {
           input_path: /path/to/reference_sequence.fa
       },
       write_mem_limit: 5000
   }

Required parameters
^^^^^^^^^^^^^^^^^^^


* ``trained_model_path``\ : Path to the trained model weights file, which should have been generated/saved using ``torch.save``. (i.e. You can pass in the saved model file generated by Selene's ``TrainModel`` class.)
* ``sequence_length``\ : The sequence length the model is expecting for each input.
* ``features``\ : The list of distinct features the model predicts. (\ ``input_path`` to the function-type value that loads the features as a list.)

Optional parameters
^^^^^^^^^^^^^^^^^^^


* ``batch_size``\ : Default is 64. The size of the mini-batches to use.
* ``use_cuda``\ : Default is ``False``. Specify whether CUDA-enabled GPUs are available for torch to use.  
* ``reference_sequence``\ : Default is the class ``selene_sdk.sequences.Genome``. The type of sequence on which this analysis will be performed (must be type ``selene.sequences.Sequence``\ ).

  * IMPORTANT: For variant effect prediction and prediction on sequences in a BED file, the reference sequence version should correspond to the version used to specify the chromosome and position of each variant, NOT necessarily the one on which your model was trained. 
  * For prediction on sequences in a FASTA file and *in silico* mutagenesis, the only thing that matters is the sequence type---that is, Selene uses the static variables in the class for information about the sequence alphabet and encoding. One problem with our current configuration file parsing is that it asks you to pass in a valid input FASTA file even though you do not need the reference sequence for these 2 sub-operations. We aim to resolve this issue in the future.

* ``write_mem_limit``\ : Default is 5000. Specify, in MB, the amount of memory you want to allocate to storing model predictions/scores. When running one of the sub-operations in ``analyze``\ , prediction/score handlers will accumulate data in memory and write this data to files periodically. By default, Selene will write to files when the **total amount** of data (that is, across all handlers) takes up 5000MB of space. Please keep in mind that Selene will not monitor the amount of memory needed to actually carry out a sub-operation (or load the model beforehand), so ``write_mem_limit`` must always be less than the total amount of CPU memory you have available on your machine. It is hard to recommend a specific proportion of memory you would allocate for ``write_mem_limit`` because it is dependent on your input file size (we may change this soon, but Selene currently loads all variants/sequences in a file into memory before running the sub-operation), the model size, and whether the model will run on CPU or GPU.  

Prediction on sequences
^^^^^^^^^^^^^^^^^^^^^^^

For prediction on sequences, we require that a user specifies the path to a FASTA file or BED file.

An example configuration for prediction on sequences:

.. code-block:: YAML

   prediction: {
       input_path: /path/to/sequences.bed,
       output_dir: /path/to/output/dir,
       output_format: tsv,
       strand_index: 5
   }

Parameters
~~~~~~~~~~


* ``input_path``\ : Input path to the FASTA or BED file. For BED file input, we only use the genome regions specified in each row for finding the center position of the input sequence to the model. That is, the start and end of each coordinate does not need to be the same length as the expected model input sequence length--Selene will handle creating the correct sequence input for you.
* ``output_dir``\ : Output directory to write the model predictions. The resulting file will have the same filename prefix (e.g. ``example.fasta`` will output ``example_predictions.tsv``\ ).
* ``output_format``\ : Default is 'tsv'. You may specify either 'tsv' or 'hdf5'. 'tsv' is suitable if you do not have many sequences (<1000) or your model does not predict very many classes (<1000) and you want to be able to view the full set of predictions quickly and easily (via a text editor or Excel). 'hdf5' is suitable for downstream analysis. You can access the data in the HDF5 file using the Python package ``h5py``. Once the file is loaded, the full matrix is accessible under the key/name ``"data"``. Saving to TSV is much slower (more than 2x slower) than saving to HDF5. An additional .txt file with the row labels (descriptions for each sequence in the FASTA) will be output for the HDF5 format as well. It should be ordered in the same way as your input file. The matrix rows will correspond to each sequence and the columns the classes the model predicts.  
* ``strand_index``\ : Default is None. If input is BED file, you may specify the column index (0-based) that contains strand information. Otherwise we assume all sequences passed into the model will be fetched from the forward strand. The reference and alternate alleles specified in the VCF should still be for the forward strand--Selene will apply reverse complement to those alleles when strand is '-'.

Variant effect prediction
^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, we expect that all sequences passed as input to a model must be the same length ``N``. 


* For SNPs, Selene outputs the model predictions for the ref and alt sequences centered at the ``(chr, pos)`` specified. 
* For indels, sequences are centered at ``pos + (N_bases / 2)``\ , for the reference sequence of length ``N_bases``. 

An example configuration for variant effect prediction:

.. code-block:: YAML

   variant_effect_prediction: {
       vcf_files: [
           /path/to/file1.vcf,
           /path/to/file2.vcf,
           ...
       ],
       save_data: [abs_diffs],
       output_dir: /path/to/output/predictions/dir,
       output_format: hdf5,
       strand_index: 7,
       require_strand: True
   }

Parameters
~~~~~~~~~~


* ``vcf_files``\ : Path to a VCF file. Must contain the columns ``[#CHROM, POS, ID, REF, ALT]``\ , in order. Column header does not need to be present. (All other columns in the file will be ignored.)
* `save_data`: A list of the data files to output. Must input 1 or more of the following options: ``[abs_diffs, diffs, logits, predictions]``. (Note that the raw prediction values will not be outputted by default---you must specify ``predictions`` in the list if you want them.)
* ``output_dir``\ : Output directory to write the model predictions. The resulting file will have the same filename prefix.
* ``output_format``\ : Default is 'tsv'. You may specify either 'tsv' or 'hdf5'. 'tsv' is suitable if you do not have many variants (on the order of 10^4 or less) or your model does not predict very many classes (<1000) and you want to be able to view the full set of predictions quickly and easily (via a text editor or Excel). 'hdf5' is suitable for downstream analysis. You can access the data in the HDF5 file using the Python package ``h5py``. Once the file is loaded, the full matrix is accessible under the key/name ``"data"``. Saving to TSV is much slower (more than 2x slower) than saving to HDF5. When the output is in HDF5 format, an additional .txt file of row labels (corresponding to the columns (chrom, pos, id, ref, alt)) will be output so that you can match up the data matrix rows with the particular variant. Columns of the matrix correspond to the classes the model predicts.
* ``strand_index``\ : Default is None. If applicable, specify the column index (0-based) in the VCF file that contains strand information for each variant. Note that currently Selene assumes that, for multiple input VCF files, the strand column is the same for all the files. Importantly, the VCF file ref and alt alleles should still be specified for the forward strand--Selene will take the reverse complement for both if strand = '-'. 
* ``require_strand``\ : Default is False. If ``strand_index`` is not None, ``require_strand = True`` means that Selene will skip all variants with strand specified as '.' (that is, only keep variants with strand column value being '+' or '-'). If ``require_strand = False``\ , variants with strand specified as '.' will be treated as being on the '+' strand.

Additional note
~~~~~~~~~~~~~~~

You may find that there are more output files than you expect in ``output_dir`` at the end of variant effect prediction. The following cases may occur:


* **NAs:** for some variants, Selene may not be able to construct a reference sequence centered at ``pos`` of the specified sequence length. This is likely because ``pos`` is near the end or the beginning of the chromosome and the sequence length the model accepts as input is large. You will find a list of NA variants in a file that ends with the extension ``.NA``. 
* **Warnings:** Selene may detect that the ``ref`` base(s) in a variant do not match with the bases specified in the reference sequence FASTA at the ``(chrom, pos)``. In this case, Selene will use the ``ref`` base(s) specified in the VCF file in place of those in the reference genome and output predictions accordingly. These predictions will be distinguished by the row label column ``ref_match`` value ``False``. You may review these variants and determine whether you still want to use those predictions/scores. If you find that most of the variants have ``ref_match = False``\ , it may be that you have specified the wrong reference genome version---please check this before proceeding.  

*In silico* mutagenesis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example configuration for *in silico* mutagenesis of the whole sequence (i.e. rather than a subsequence), when using a single sequence as input:

.. code-block:: YAML

   in_silico_mutagenesis: {
       input_sequence: ATCGATAAAATTCTGGAG...,
       save_data: [predictions, diffs],
       output_path_prefix: /path/to/output/dir/filename_prefix,
       mutate_n_bases: 1,
       start_position: 0,
       end_position: None
   }

Parameters for a single sequence input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* ``sequence``\ : A sequence you are interested in. If the sequence length is less than or greater than the expected model's input sequence length, Selene truncates or pads (with unknown base, e.g. ``N``\ ) the sequence for you.
* `save_data`: A list of the data files to output. Must input 1 or more of the following options: ``[abs_diffs, diffs, logits, predictions]``. (Note that the raw prediction values will not be outputted by default---you must specify ``predictions`` in the list if you want them.)
* ``output_path_prefix``\ : Optional, default is "ism". The path to which the data files are written. We have specified that it should be a filename *prefix* because we will append additional information depending on what files you would like to output (e.g. ``fileprefix_logits.tsv``\ ) If directories in the path do not yet exist, they will automatically be created. 
* ``mutate_n_bases``\ : Optional, default is 1. The number of bases to mutate at any time. Standard *in silico* mutagenesis only mutates a single base at a time, so we encourage users to start by leaving this value at 1. Double/triple mutations will be more difficult to interpret and are something we may work on in the future. 
* ``start_position``\ : Optional, default is 0. The starting position of the subsequence that should be mutated. This value should be nonnegative, and less than ``end_position``. Also, the value of ``end_position - start_position`` should be at least ``mutate_n_bases``.
* ``end_position``\ : Optional, default is ``None``. If left as ``None``\ , Selene will use the ``sequence_length`` parameter from ``analyze_sequences``. This is the ending position of the subsequence that should be mutated. This value should be nonnegative, and greater than ``start_position``. The value of ``end_position -  start_position`` should be at least ``mutate_n_bases``.

An example configuration for *in silico* mutagenesis of the center 100 bases of a 1000 base sequence read from a FASTA file input:

.. code-block:: YAML

   in_silico_mutagenesis: {
       input_path: /path/to/sequences1.fa,
       save_data: [logits],
       output_dir: /path/to/output/predictions/dir,
       mutate_n_bases: 1,
       use_sequence_name: True,
       start_position: 450,
       end_position: 550
   }

Parameters for FASTA file input:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* ``input_path``\ : Input path to the FASTA file. If you have multiple FASTA files, you can replace this key with ``fa_files`` and submit an input list, the same way it is done in variant effect prediction.
* `save_data`: A list of the data files to output. Must input 1 or more of the following options: ``[abs_diffs, diffs, logits, predictions]``. 
* ``output_dir``\ : Output directory to write the model predictions.
* ``mutate_n_bases``\ : Optional, default is 1. The number of bases to mutate at any time. Standard *in silico* mutagenesis only mutates a single base at a time, so we encourage users to start by leaving this value at 1.
* ``use_sequence_name``\ : Optional, default is ``True``.

  * If ``use_sequence_name``\ , output files are prefixed by the sequence name/description corresponding to each sequence in the FASTA file. Spaces in the description are replaced with underscores '_'.
  * If not ``use_sequence_name``\ , output files are prefixed with the index ``i`` corresponding to the ``i``\ th sequence in the FASTA file.

* ``start_position``\ : Optional, default is 0. The starting position of the subsequence that should be mutated. This value should be nonnegative, and less than ``end_position``. The value of ``end_position - start_position`` should be at least ``mutate_n_bases``.
* ``end_position``\ : Optional, default is ``None``. If left as ``None``\ , Selene will use the ``sequence_length`` parameter passed to ``analyze_sequences``. This is the ending position of the subsequence that should be mutated. This value should be nonnegative, and greater than ``start_position``. The value of ``end_position -  start_position`` should be at least ``mutate_n_bases``.

Sampler configurations
----------------------

Data sampling is used during model training and evaluation. You must specify the sampler in the configuration YAML file alongside the other operation-specific configurations (i.e. ``train_model`` or ``evaluate_model``\ ). 

Samplers used for training (and evaluation, optionally)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Training requires a sampler that specifies the data for training, validation, and (optionally) testing. While Selene can directly evaluate a trained model on a test dataset when training is finished, it is not a required step and so the test dataset specification is also optional. Here, we provide examples for the samplers we have implemented that can be used for training.

There are 2 kinds of samplers implemented in Selene right now: "online" samplers and file samplers. Online samplers generate data samples on-the-fly and require you to pass in a reference sequence FASTA file and a tabix-indexed BED file so that Selene can query for an input sequence and its associated biological classes using genomic coordinates. The file sampler we use supports loading different ``.mat`` or ``.bed`` files (can support more formats upon request) for the training, validation, and test sets. 

For increased efficiency during the training of large models, we would recommend using the online sampler to create datasets (.bed or .mat) and then loading the generated data with a file sampler. We are actively working to incorporate PyTorch dataloaders and other improvements to data sampling into Selene to reduce the time and memory requirements of training. Feel free to contact us through our `Github issues <https://github.com/FunctionLab/selene/issues>`_ if you have comments or want to contribute to this effort! 

Random positions sampler
~~~~~~~~~~~~~~~~~~~~~~~~

The random positions sampler will construct data samples by randomly selecting a position in the genome and then using the sequence and classes centered at that position as the input and targets for the model to predict. 

An example configuration for the random positions sampler:

.. code-block:: YAML

   sampler: !obj:selene_sdk.samplers.RandomPositionsSampler {
       reference_sequence: !obj:selene_sdk.sequences.Genome {
           input_path: /path/to/reference_sequence.fa,
           blacklist_regions: hg19
       },
       features: !obj:selene_sdk.utils.load_features_list {
           input_path: /path/to/features_list.txt
       },
       target_path: /path/to/targets_bed.gz,
       seed: 123,
       validation_holdout: [chr6, chr7],
       test_holdout: [chr8, chr9],
       sequence_length: 1000,
       center_bin_to_predict: 200,
       feature_thresholds: 0.5,
       mode: train,
       save_datasets: [train, validate, test]
   }

Required parameters
"""""""""""""""""""


* ``reference_sequence``\ : Path to a reference sequence FASTA file we can query to create our data samples.

  * ``blacklist_regions`` is an optional argument for ``selene_sdk.sequences.Genome`` that allows you to specify the blacklist regions for the hg19 or hg38 reference sequence. The lists of blacklisted intervals are provided by `Anshul Kundaje for ENCODE <https://sites.google.com/site/anshulkundaje/projects/blacklists>`_ and support for more organisms can be included upon request.

* ``target_path``\ : Path to a tabix-indexed, compressed BED file (\ ``.bed.gz``\ ) of genomic coordinates corresponding to the measurements for genomic features/classes the model should predict. 
* ``features``\ : The list of distinct features the model predicts. (\ ``input_path`` to the function-type value that loads the file of features as a list.)

Optional parameters
"""""""""""""""""""


* ``seed``\ : Default is 436. 
* ``validation_holdout``\ : Default is ``[chr6, chr7]``. Holdout can be regional (i.e. chromosomal) or proportional.

  * If regional, expects a list where the regions must match those specified in the first column of the tabix-indexed BED file ``target_path`` (which must also match the FASTA descriptions for every record in ``reference_sequence``\ ).
  * If proportional, specify a percentage between (0.0, 1.0). Typically 0.10 or 0.20.

* ``test_holdout``\ : Default is ``[chr8, chr9]``. Holdout can be regional (i.e. chromosomal) or proportional. See description of ``validation_holdout``. 
* ``sequence_length``\ : Default is 1000. Model is trained on sequences of ``sequence_length``. 
* ``center_bin_to_predict``\ : Default is 200. Query the tabix-indexed file for a region of length ``center_bin_to_predict``\ , centered in the input sequence of ``sequence_length``. 
* ``feature_thresholds``: Default is 0.5. The threshold to pass to the ``selene_sdk.targets.Targets`` object. Because we have only implemented support for genomic features right now, we reproduce the threshold inputs for that here:

  * A genomic region is determined to be a positive sample if at least one genomic feature interval takes up some proportion of the region greater than or equal to the corresponding threshold.

    * ``float``\ : A single threshold applied to all the features in your dataset. 
    * ``dict``\ : A dictionary mapping feature names (\ ``str``\ ) to thresholds (\ ``float``\ ). This is used if you want to assign different thresholds for different features. If a feature's threshold is not specified in the dictionary, you must have the key ``default`` with a default threshold value we can use for that feature. 

* ``mode``\ : Default is 'train'. Must be one of ``{train, validate, test}``. The starting mode in which to run this sampler.
* ``save_datasets``\ : Default is ``[test]``. The list of modes for which we should save the sampled data to file. Should be one or more of ``{train, validate, test}``. 

Intervals sampler
~~~~~~~~~~~~~~~~~

The intervals sampler will construct data samples by randomly selecting positions only in the regions specified by an intervals ``.bed`` file and then using the sequence and classes centered at that position as the input and targets for the model to predict. 

An example configuration for the intervals sampler:

.. code-block:: YAML

   sampler: !obj:selene_sdk.samplers.IntervalsSampler {
       reference_sequence: !obj:selene_sdk.sequences.Genome {
           input_path: /path/to/reference_sequence.fa,
           blacklist_regions: hg38
       },
       target_path: /path/to/targets.bed.gz,
       features: !obj:selene_sdk.utils.load_features_list {
           input_path: /path/to/features_list.txt
       },
       intervals_path: /path/to/intervals.bed,
       sample_negative: False,
       seed: 436,
       validation_holdout: 0.10,
       test_holdout: 0.10,
       sequence_length: 1000,
       center_bin_to_predict: 100,
       feature_thresholds: {"feature1": 0.5, "default": 0.1},
       mode: test,
       save_datasets: [test]

Parameters
""""""""""

With the exception of ``intervals_path`` and ``sample_negative``\ , all other parameters match those for the random positions sampler. Please see the previous section for more details on the other parameters. 


* ``intervals_path``\ : The path to the intervals file. Must have the columns ``[chr, start, end]``\ , where values in ``chr`` should match the descriptions in the FASTA file. We constrain the regions from which we sample to the regions in this file instead of the using the whole genome. 
* ``sample_negative``\ : Optional, default is False. Specify whether negative examples (i.e. samples with no positive labels) should be drawn. When False, the sampler will check if the ``center_bin_to_predict`` in the input sequence contains at least 1 of the features/classes the model wants to predict. When True, no such check is made. 

Multiple-file sampler
~~~~~~~~~~~~~~~~~~~~~

The multi-file sampler loads in the training, validation, and optionally, the testing dataset.  The configuration for this therefore asks that you fill in some keys with the function-type constructors of type ``selene_sdk.samplers.file_samplers.FileSampler``. Please consult the following sections for information about these file samplers. 

An example configuration for the multiple-file sampler:

.. code-block:: YAML

   sampler: !obj:selene_sdk.samplers.MultiFileSampler {
       train_sampler: !obj:selene_sdk.samplers.file_samplers.MatFileSampler {
           ...
       },
       validate_sampler: !obj:selene_sdk.samplers.file_samplers.MatFileSampler {
           ...
       }, 
       features: !obj:selene_sdk.utils.load_features_list {
           input_path: /path/to/features_list.txt
       },
       test_sampler: !obj:selene_sdk.samplers.file_samplers.BedFileSampler {
           ...
       },
       mode: train
   }

Parameters
""""""""""


* ``train_sampler``\ : Load your training data from either a ``.bed`` file (\ ``selene_sdk.samplers.file_sampler.BedFileSampler``\ ) or ``.mat`` file (\ ``selene_sdk.samplers.file_sampler.MatFileSampler``\ ).
* ``validate_sampler``\ : Sample as ``train_sampler``.
* ``test_sampler``\ : Optional, default is ``None``. Same as ``train_sampler``.
* ``features``\ : The list of distinct features the model predicts. (\ ``input_path`` to the function-type value that loads the file of features as a list.)
* ``mode``\ : Default is 'train'. Must be one of ``{train, validate, test}``. The starting mode in which to run this sampler.

Important note
^^^^^^^^^^^^^^

If you use any of these samplers (that is, samplers with multiple data partitions) with the :ref:`Evaluate` configuration, please make sure that your ``mode`` is set to ``test``. 

Samplers used for evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use all the samplers specified for training for evaluation as well (see note above). Additionally, you can use single-file samplers, which we describe below. 

BED file sampler
~~~~~~~~~~~~~~~~

The BED file sampler loads a dataset from a ``.bed`` file. This can be generated by one of the online samplers in Selene with the ``save_dataset`` parameter. 

An example configuration for a BED file sampler:

.. code-block:: YAML

   sampler: !obj:selene_sdk.samplers.file_samplers.BedFileSampler {
       filepath: /path/to/data.bed,
       reference_sequence: !obj:selene_sdk.sequences.Genome {
           input_path: /path/to/reference_sequence.fa
       },
       n_samples: 640000,
       sequence_length: 1000,
       targets_avail: True,
       n_features: 919,
   }

Parameters
""""""""""


* ``filepath``\ : Path to the BED file.
* ``reference_sequence``\ : Path to a reference sequence FASTA file we can query to create our data samples.
* ``n_samples``\ : Number of lines in the file. (\ ``wc -l <filepath>``\ )
* ``sequence_length``\ : Optional, default is None. If the coordinates of each sample in the BED file, already account for the full sequence (that is, the columns ``end - start = sequence_length``\ , there is no need to specify this parameter. If ``sequence_length`` is not None, the length of each sample will be checked to determine whether the sample coordinates need to be adjusted to match the sequence length expected by the model architecture.
* ``targets_avail``\ : Optional, default is False. If ``targets_avail``\ , assumes that it is the last column of the ``.bed`` file. The last column should contain the indices, separated by semicolons, of features (classes) found within a given sample's coordinates (e.g. 0;1;45;60). This format assumes that we are only looking for the absence/presence of each feature within the interval.
* ``n_features``\ : Optional, default is None. If ``targets_avail`` is True, must specify ``n_features``\ , the total number of features (classes).

Matrix file sampler
~~~~~~~~~~~~~~~~~~~

The matrix file sampler loads a dataset from a matrix file.

An example configuration for a matrix file sampler:

.. code-block:: YAML

   sampler: !obj:selene_sdk.samplers.file_samplers.MatFileSampler {
       filepath: /path/to/data.mat,
       sequence_key: sequences,
       targets_key: targets,
       random_seed: 123,
       shuffle: True,
       sequence_batch_axis: 0,
       sequence_alphabet_axis: 1,
       targets_batch_axis: 0
   }

Parameters
""""""""""


* ``filepath``\ : The path to the file from which to load the data. 
* ``sequence_key``\ : The key for the sequences data matrix.
* ``targets_key``\ : Optional, default is ``None``. The key to the targets data matrix.
* ``random_seed``\ : Optional, default is 436. Sets the random seed for sampling.
* ``shuffle``\ : Optional, default is ``True``. Shuffle the order of the samples in the matrix before sampling from it.
* ``sequence_batch_axis``\ : Optional, default is 0. Specify the batch axis for the sequences matrix.
* ``sequence_alphabet_axis``\ : Optional, default is 1. Specify the alphabet axis.
* ``targets_batch_axis``\ : Optional, default is 0. Specify the batch axis for the targets matrix.

Examples of full configuration files
------------------------------------

We do have a more comprehensive set of `examples on our Github <https://github.com/FunctionLab/selene/blob/master/config_examples>`_ that you can review. We reproduce a few of these in this document to show how you can put all of the different configuration components together to create a YAML file that can be run by Selene's CLI:

Training (using intervals sampler)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: YAML

   ---
   ops: [train, evaluate]
   model: {
       path: /absolute/path/to/model/architecture.py,
       class: ModelArchitectureClassName,
       class_args: {
           arg1: val1,
           arg2: val2
       },
       non_strand_specific: mean
   }
   sampler: !obj:selene_sdk.samplers.IntervalsSampler {
       reference_sequence: !obj:selene_sdk.sequences.Genome {
           input_path: /path/to/reference_sequence.fa,
           blacklist_regions: hg19
       },
       target_path: /path/to/tabix/indexed/targets.bed.gz,
       features: !obj:selene_sdk.utils.load_features_list {
           input_path: /path/to/distinct_features.txt
       },
       intervals_path: /path/to/intervals.bed,
       sample_negative: True,
       seed: 127,
       validation_holdout: [chr6, chr7],
       test_holdout: [chr8, chr9],  # specifying a test partition
       sequence_length: 1000,
       center_bin_to_predict: 200,
       feature_thresholds: 0.5,
       mode: train,  # starting mode for sampler
       save_datasets: [test]
   }
   train_model: !obj:selene_sdk.TrainModel {
       batch_size: 64,
       max_steps: 80000,
       report_stats_every_n_steps: 16000,
       n_validation_samples: 32000,
       n_test_samples: 640000,
       cpu_n_threads: 32,
       use_cuda: True,
       data_parallel: True,
       logging_verbosity: 2,
       checkpoint_resume: False
   }
   random_seed: 133
   output_dir: /path/to/output_dir
   ...

Some notes
~~~~~~~~~~


* Ordering of the keys does not matter.
* We included many of the optional keys in this configuration. You do not need to specify these if you want to use their default values.
* In this example, we specified a test partition in our intervals sampler by assigning a list of chromosomes to ``test_holdout``. If no such holdout was specified (e.g. None or empty list), you would not be able to specify ``n_test_samples`` in ``TrainModel`` and would need to omit ``evaluate`` from the ``ops`` list. 
* ``output_dir`` is specified at the top-level and used by both the sampler and the ``TrainModel`` class. 

Evaluate (using matrix file sampler)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: YAML

   ---
   ops: [evaluate]
   model: {
       path: /absolute/path/to/model/architecture.py,
       class: ModelArchitectureClassName,
       class_args: {
           arg1: val1,
           arg2: val2
       },
       non_strand_specific: mean
   }
   sampler: !obj:selene_sdk.samplers.file_samplers.MatFileSampler {
       filepath: /path/to/test.mat,
       sequence_key: testxdata,
       targets_key: testdata,
       random_seed: 456,
       shufle: False,
       sequence_batch_axis: 0,
       sequence_alphabet_axis: 1,
       targets_batch_axis: 0
   }
   evaluate_model: !obj:selene_sdk.EvaluateModel {
       features: !obj:selene_sdk.utils.load_features_list {
           input_path: /path/to/features_list.txt
       },
       trained_model_path: /path/to/trained/model.pth.tar,
       batch_size: 64,
       report_gt_feature_n_positives: 50,
       use_cuda: True,
       data_parallel: False
   }
   random_seed: 123
   output_dir: /path/to/output_dir
   create_subdirectory: False
   ...

Some notes
~~~~~~~~~~


* For the matrix file sampler, we assume that you know ahead of time the shape of the data matrix. That is, which dimension is the batch dimension? Sequence? Alphabet (should be size 4 for DNA/RNA)? You must specify the keys that end in ``axis`` unless the shape of the sequences matrix is ``(n_samples, n_alphabet, n_sequence_length)`` and the shape of the targets matrix is ``(n_samples, n_targets)``.
* In this case, since ``create_subdirectory`` is False, all outputs from evaluate are written to ``output_dir`` directly (as opposed to being written in a timestamped subdirectory). Be careful of overwriting files.

Analyze sequences (variant effect prediction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: YAML

   ops: [analyze]
   model: {
       path: /absolute/path/to/model/architecture.py,
       class: ModelArchitectureClassName,
       class_args: {
           arg1: val1,
           arg2: val2
       },
       non_strand_specific: mean
   }
   analyze_sequences: !obj:selene_sdk.predict.AnalyzeSequences {
       trained_model_path: /path/to/trained/model.pth.tar,
       sequence_length: 1000,
       features: !obj:selene_sdk.utils.load_features_list {
           input_path: /path/to/distinct_features.txt
       },
       batch_size: 64,
       use_cuda: True,
       reference_sequence: !obj:selene_sdk.sequences.Genome {
           input_path: /path/to/reference_sequence.fa
       },
       write_mem_limit: 75000
   }
   variant_effect_prediction: {
       vcf_files: [
           /path/to/file1.vcf,
           /path/to/file2.vcf
       ],
       save_data: [predictions, abs_diffs],
       output_dir: /path/to/output/predicts/dir,
       output_format: tsv,
       strand_index: 9
   }
   random_seed: 123

Some notes
~~~~~~~~~~


* We ask that in all ``analyze`` cases, you specify the ``output_dir`` (when applicable) within the sub-operation dictionary. This is because only the sub-operation generates output, so there is no need to share this parameter across multiple configurations.
* In this variant effect prediction example, Selene will go through each VCF file and get the model predictions for each variant (ref and alt). ``analyze_sequences`` must have the parameter ``reference_sequence`` so that Selene can create sequences centered at each variant position by querying the reference sequence file. 
* The output from this operation will be 6 files: 3 for each input VCF file. This is because of what is specified in ``save_data``\ :

  * ``predictions`` will output 2 files per input VCF: the model predictions for all ``ref``\ s and the model predictions for all ``alts``. 
  * ``abs_diffs`` will output 1 file per input VCF: the absolute difference between the ``ref`` and ``alt`` model predictions. (Certainly, outputting the files from ``predictions`` is sufficient to compute ``abs_diffs`` yourself.)
