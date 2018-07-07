# Tutorials

This directory contains all of the tutorials for selene.
All of these tutorials are also available on the selene website [here](http://selene.flatironinstitute.org/tutorials/).


## Contributing tutorials

The process for adding a tutorial to selene is as follows:

1. Create a subdirectory in the tutorials directory. The name of this subdirectory should be the name of the tutorial, formatted in snake-case.
2. Write the tutorial in an [ipython notebook](https://ipython.org/notebook.html) in the subdirectory.
3. Store all data for the tutorial in the subdirectory, and create a gzipped archive (i.e. a `*.tar.gz` file) with all the data required for the tutorial.
4. Create a `*.nblink` link file in the `docs/source/tutorials` directory. This file will serve as a link to the tutorial's notebook file. Instructions for formatting this file can be found [here](https://github.com/vidartf/nbsphinx-link).
5. Add an entry for the tutorial to the list of tutorials in `docs/source/tutorials/index.rst`.
6. Rerun `make html` from the `docs` directory.

