# Example of exporting a model to Kipoi.

Selene enables users to experiment with modifying existing architecture or creating entirely new architectures. Selene can support research projects across different stages (development, evaluation, and application) and is most useful when a researcher wants to develop and validate a model for a new publication. 

After publication, we encourage users to archive and share their models through the [Kipoi model zoo](http://kipoi.org/) so that other researchers can access, use, and build on these models. 

Here we demonstrate the steps needed to contribute a model to Kipoi. 
This is based on the [Contributing models](http://kipoi.org/docs/contributing/01_Getting_started/) tutorial provided in the Kipoi documentation, and some content in this document will be pulled from that tutorial.

We do plan to automate much of the work needed to export a model to Kipoi so that users can run a command in the Selene CLI to generate the same output we have here.

## Requirements

To export a model to Kipoi, you should pip install `kipoi` and `kipoiseq`:

```sh
pip install -U kipoi
pip install -U kipoiseq
```

This should also install jinja2 for you, which we use in a script called `kipoi_export.py`. 

Please also install `docopt` if you have not done so already:
```sh
conda install -c anaconda docopt
```

## Running `kipoi_export.py`

In this example, we run a script `kipoi_export.py` with the following command:
```sh
python kipoi_export.py <path>/best_model.pth.tar \
                       <path>/class_names.txt \
                       <path>/config.yaml \
                       <path-to-output-dir>
```

### Parameters
- `best_model.pth.tar`: serialized dictionary containing the trained model state and other parameters, from Selene training
- `class_names.txt`: the list of distinct classes that the model predicts
- `config.yaml`: A configuration file that is used to fill out the values in `model-template.yaml`. The filled out template is output to `model.yaml`, which is a file required in Kipoi.
- `<path-to-output-dir>`: the output directory (`~/.kipoi/models/ModelName`) 

The steps taken in the script:
1. Save only the model state dictionary (`model.state_dict()`) from `best_model.pth.tar` and writes the resulting file to the output directory.
2. Copies the file of class names to the output directory.
3. Uses the config YAML to populate the values in [`model-template.yaml`](https://github.com/FunctionLab/selene/tree/master/manuscript/case2/3_kipoi_export/model-template.yaml) and writes a `model.yaml` file to the output directory.

After installing `kipoi`, you should be able to view your Kipoi model folder (default: `~/.kipoi/models`). For the model you want to submit, called `ModelName`, you should specify the output directory as `~/.kipoi/models/ModelName`. 

### The config YAML file
This is used to generate `model.yaml` from `model-template.yaml`.

#### Parameters
- `module_class`: the module class name (see [Formatting your model architecture file(s)](#formatting-your-model-architecture-files))
- `module_kwargs`: optional, specify any arguments needed to initialize the model architecture class
    - For example:
        ```YAML
        module_kwargs:
          arg1: val1
          arg2: val2
        ```
- `authors`: list of authors (each item in the list is a dictionary with `author` and `github`)
    - For example:
        ```YAML
        authors:
          - author: a1
            github: g1
          - author: a2
            github: g2
        ```
- `license`: the model license (e.g. MIT, BSD 3-Clause). Only contribute models for which you have the rights to do so and only contribute models that permit redistribution.
- `model_name`: the model name
- `trained_on_description`: describe the data on which the model was trained, what the validation and testing holdouts were, etc.
- `selene_version`: the version of Selene you used to train the model
- `tags`: optional, specify relevant tags in a list (e.g. histone modification)
    - For example:
        ```YAML
        tags:
          - Histone modification
          - DNA accessibility
        ```
- `seq_len`: the length of the sequences the model accepts as input
- `pytorch_version`: the version of PyTorch used to train the model
- `n_tasks`: the number of tasks (classes/labels) the model predicts

(List is ordered in the way the parameters appear in `model-template.yaml`)

We recommend that you run `kipoi_export.py` with your filled-out `config.yaml` and then manually make adjustments to the generated `model.yaml` file in the output directory. There will be comments in the file to highlight where you might need to change something.

Specifically, the `weights` parameter in `args` should  be updated after you upload your model file to Zenodo or Figshare. See `model-template.yaml` or the generated `model.yaml` for details. You can also see an example of the final `model.yaml` file [here](https://github.com/kipoi/models/blob/master/DeepSEA/predict/model.yaml#L5-L7).

You should also update the `cite_as` parameter with the DOI url to your publication.

### Formatting your model architecture file(s)

Move your model architecture file or module into `~/.kipoi/models/ModelName`. The next sections will explain what format is expected if your model architecture should be organized as a module.

#### If you did NOT use Selene's NonStrandSpecific module (i.e. the `non_strand_specific` parameter)
If your model architecture is implemented in a single file called `model_name_arch.py`, you can specify `module_class` to be `model_name_arch.ModelName`.

Otherwise, you can move all your model architecture files into a directory called `model_arch` and import the model `ModelName` in `__init__.py`. `model_arch` is then a Python module that Kipoi can use to import your model architecture class. You can view our example directory `model_arch` to see how this is structured. 

#### If you used Selene's NonStrandSpecific module:
We are working to automate this, but currently, we have to do a few manual steps to get our architecture formatted for export to Kipoi. This is applicable to our example, so you can refer to the files in there to see how we did this. 

    1. Create a directory called `model_arch`. This will be the Python module that Kipoi uses to import your model architecture class. 
    2. Copy the file [`non_strand_specific_module.py`](https://github.com/FunctionLab/selene/blob/master/selene_sdk/utils/non_strand_specific_module.py) from the Selene repostory to your directory.
    3. For an architecture where the main class `ModelName` is in the file `model_name_arch.py`, you should add the line `from .model_name_arch import ModelName` to `non_strand_specific_module.py`. 
    4. Next, remove the constructor input `model` from `__init__(self, model, mode="mean")` and set `self.model` to `ModelName(**kwargs)`
    5. Optional, but helpful: rename `non_strand_specific_module.py` to something more representative of your model architecture (e.g. `wrapped_model_name.py`). You can rename the class from `NonStrandSpecific` to `WrappedModelName` or something else as well. Please note that you'll need to update `super(NonStrandSpecific, self).__init__()` with the new class name too. 
    6. Finally, import your class in the file `model_arch/__init__.py` (e.g. from `.wrapped_model_name import WrappedModelName`).

How we applied these steps our example:

    - Create the directory `model_arch`.
    - The architecture file is `deeper_deepsea_arch.py`, which contains the architecture class `DeeperDeepSEA`.
    - Rename `non_strand_specific_module.py` to `wrapped_deeper_deepsea.py` and update the class name in the file to `WrappedDeeperDeepSEA`.
    - Import `DeeperDeepSEA` with the line `from .deeper_deepsea_arch import DeeperDeepSEA`. 
    - Remove `model` from `__init__` and set `self.model = DeeperDeepSEA(1000, 919)`. 
    - Create the file `__init__.py` in `model_arch` with the line `from .wrapped_deeper_deepsea import WrapperDeeperDeepSEA`. 

## Testing

The following commands assume that you are in `~/.kipoi/models/ModelName`.

Run `kipoi test .` in your model folder to test whether the general setup is correct. 

If this is successful, run `kipoi test-source dir --all` to test whether all the software dependencies of the model are set up correctly and the automated tests pass.

## Forking and submitting to Kipoi

Fork the https://github.com/kipoi/models repo on Github.

Add your fork as a git remote to `~/.kipoi/models`.
```sh
git remote add fork https://github.com/<username>/models.git
```

Push to your fork
```sh
git push fork master
```

Submit a pull request to https://github.com/kipoi/models!






