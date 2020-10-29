
Functional overview of the SDK
==============================

The software development kit (SDK), formally known as ``selene_sdk``\ , is an extensible Python package intended to ease development of new programs that leverage sequence-level models through code reuse.
The package is composed of six submodules: *sequences*\ , *samplers*\ , *targets*\ , *predict*\ , *interpret*\ , and *utils*.
It also provides two top-level classes: *TrainModel* and *EvaluateModel*.
In the following sections, we briefly discuss each submodule and top-level class. 

Sampling
--------

We start with the modules for sampling data because both training and evaluting a model in Selene will require a user to specify the kind of sampler they want to use. 

*sequences* submodule (\ `API <http://selene.flatironinstitute.org/sequences.html>`_\ )
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The *sequences* submodule defines the ``Sequence`` type, and includes implementations for several sub-classes.
These sub-classes--\ ``Genome`` and ``Proteome``\ --represent different kinds of biological sequences (e.g. DNA, RNA, amino acid sequences), and implement the ``Sequence`` interface’s methods for reading the reference sequence from files (e.g. FASTA), querying subsequences of the reference sequence, and subsequently converting those queried subsequences into a numeric representation.
Further, each sequence class specifies its own alphabet (e.g., nucleotides, amino acids) to represent query results as strings.

*targets* submodule (\ `API <http://selene.flatironinstitute.org/targets.html>`_\ )
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The *targets* submodule defines the ``Target`` class, which specifies the interface for classes to retrieve labels or “targets” for a given query sequence.
At present, we supply a single implementation of this interface: ``GenomicFeatures``.
This class takes a tabix-indexed file of intervals for each label we want our model to predict, and uses this file to identify the labels for a given sequence drawn from the reference.

*samplers* submodule (\ `API <http://selene.flatironinstitute.org/samplers.html>`_\ )
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The *samplers* submodule provides methods and classes for randomly sampling and partitioning datasets for training and evaluation.
The ``Sampler`` interface defines the minimal requirements for a class fulfilling these functions.
In particular, samplers must be able to partition data (i.e. into training, validation, and testing datasets), sample data from each partition, and, if needed, save the sampled data to a file.
Further, a file of names must be provided for the features to be predicted.
We provide several implementations adhering to the ``Sampler`` interface: the ``RandomPositionsSampler``\ , ``IntervalsSampler``\ , and ``MultiFileSampler``.

``MultiFileSampler`` draws samples from structured data files for each partition.
There is currently support for loading either .bed or .mat files via the ``FileSampler`` classes ``BedFileSampler`` and ``MatFileSampler``\ , respectively (see `API docs for file samplers <http://selene.flatironinstitute.org/samplers.file_samplers.html>`_\ ).
It is worth noting that the .bed file used by ``BedFileSampler`` includes the coordinates of each sequence, and the indices corresponding to each feature for which said sequence is a positive example.
We hope that users will request or contribute classes for other file samplers in the future.
``MultiFileSampler`` does not support saving the sampled data to a file, so calling the ``save_dataset_to_file`` method from this class will have no effect.

``RandomPositionsSampler`` and ``IntervalsSampler`` are what we call online samplers.
Online samplers generate examples from the reference sequence (e.g. genome, proteome) on-the-fly--either across the whole reference sequence (random positions sampler), or from user-specified regions (intervals sampler)--using a tabix-indexed .bed file.
These samplers automatically partition said data according to user-specified parameters (e.g. validate on a subset of chromosomes or on some percentage of the data).
Since ``OnlineSampler``\ ’s samples are randomly generated, we allow the user to save the sampled data to file.
This file can be subsequently loaded with the ``BedFileSampler``. They rely on classes from the *sequences* and *targets* submodules for retrieving each sequence and its targets in the proper matrix format. 

Training a model (\ `API <http://selene.flatironinstitute.org/selene.html#trainmodel>`_\ )
------------------------------------------------------------------------------------------

The ``TrainModel`` class may be used for training and testing of sequence-based models, and provides the core functionality of the CLI’s train command.
It relies on an ``OnlineSampler`` (or a subclass of ``OnlineSampler``\ )  to automatically partition the dataset into subsets for training, validation, and testing.
These subsets are then used to automatically train and validate performance for a user-specified number of steps.
The testing subset is used to evaluate the model performance after training is completed.
The model’s loss, area under the receiver operating characteristic curve (AUC), and area under the precision-recall curve (AUPRC) are logged during training. (In the future, we plan to support other performance metrics. Please request specific ones or use cases in our `Github issues <https://github.com/FunctionLab/selene/issues>`_.
The frequency of logging is provided by the user.
At the end of evaluation, ``TrainModel`` logs the performance metrics for each feature predicted, and produces plots of the precision recall and receiver operating characteristic curves.

Evaluating a model (\ `API <http://selene.flatironinstitute.org/selene.html#evaluatemodel>`_\ )
-----------------------------------------------------------------------------------------------

The ``EvaluateModel`` class is used to test the performance of a trained model. 
``EvaluateModel`` uses an instance of ``Sampler`` class or subclass to draw samples from a test set.
After using the provided model to predict labels for said data, ``EvaluateModel`` logs the performance measures (as described in "Training a model") and generates figures and a performance breakdown by feature.

Using a model to make predictions (\ `API <http://selene.flatironinstitute.org/predict.html>`_\ )
-------------------------------------------------------------------------------------------------

Selene’s ``predict`` submodule includes a number of methods and classes for making predictions with sequence-based models. 
The ``AnalyzeSequences`` class is the main class to use.
It leverages a user-specified trained model to make predictions for sequences sequences in a FASTA file, apply *in silico* mutagenesis to sequences in a FASTA file, or perform variant effect prediction on variants in a VCF file.
In each case, the user can specify what ``AnalyzeSequences`` should save: raw predictions, difference scores, absolute difference scores, and/or logit scores.
Note that the aforementioned “scores” can only be computed for *in silico* mutagenesis and variant effect prediction. 

Visualizing model predictions (\ `API <http://selene.flatironinstitute.org/interpret.html>`_\ )
-----------------------------------------------------------------------------------------------

The ``interpret`` submodule of ``selene_sdk`` provides methods for visualizing a sequence-based model’s predictions made with ``AnalyzeSequences``.
For example, ``interpret`` includes methods for processing variant effect predictions made with ``AnalyzeSequences`` and subsequently visualizing them with a heatmap or sequence logo.
The functionality included in the ``interpret`` submodule is not heavily incorporated into the CLI, but is instead intended for incorporation into user code.

The utilities submodule (\ `API <http://selene.flatironinstitute.org/utils.html>`_\ )
-------------------------------------------------------------------------------------

Unlike the aforementioned submodules designed around individual concepts, the ``utils`` submodule is a catch-all submodule intended to prevent cluttering of the ``selene_sdk`` top-level namespace. 
It provides diverse functionality at varying levels of flexibility. 
Some members of ``utils`` are general-purpose (e.g. configuration file parsing) while others have highly specific use cases (e.g. CLI logger initialization).

Help
----
Join our `Google group <https://groups.google.com/forum/#!forum/selene-sdk>`_ if you have questions about the package, case studies, or model development.
