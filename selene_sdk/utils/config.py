"""Classes and methods for loading configurations from YAML files.
Taken (with minor changes) from `Pylearn2`_.


.. _Pylearn2: \
http://github.com/lisa-lab/pylearn2/blob/master/pylearn2/config/yaml_parse.py

"""
import os
import re
import warnings
import yaml
import six
from collections import namedtuple

SCIENTIFIC_NOTATION_REGEXP = r"^[\-\+]?(\d+\.?\d*|\d*\.?\d+)?[eE][\-\+]?\d+$"
IS_INITIALIZED = False


_BaseProxy = namedtuple("_BaseProxy", ["callable", "positionals", "keywords",
                                     "yaml_src"])


class _Proxy(_BaseProxy):
    """An intermediate representation between initial YAML parse and
    object instantiation.

    Parameters
    ----------
    callable : callable
        The function/class to call to instantiate this node.
    positionals : iterable
        Placeholder for future support for positional
        arguments (`*args`).
    keywords : dict-like
        A mapping from keywords to arguments (`**kwargs`), which may be
        `_Proxy`s or `_Proxy`s nested inside `dict` or `list` instances.
        Keys must be strings that are valid Python variable names.
    yaml_src : str
        The YAML source that created this node, if available.

    Notes
    -----
    This is intended as a robust, forward-compatible intermediate
    representation for either internal consumption or external
    consumption by another tool e.g. hyperopt.
    This particular class mainly exists to  override `_BaseProxy`'s
    `__hash__` (to avoid hashing unhashable namedtuple elements).

    Taken (with minor changes) from `Pylearn2`_.

    .. _Pylearn2: \
    http://github.com/lisa-lab/pylearn2/blob/master/pylearn2/config/yaml_parse.py


    """
    __slots__ = []

    def __hash__(self):
        """Return a hash based on the object ID (to avoid hashing
         unhashable namedtuple elements).

        """
        return hash(id(self))

    def bind(self, **kwargs):
        """Sets the values for specified keys.

        """
        for k in kwargs:
            if k not in self.keywords:
                self.keywords[k] = kwargs[k]

    def pop(self, key):
        return self.keywords.pop(key)


def _do_not_recurse(value):
    """Function symbol used for wrapping an unpickled object
    (which should not be recursively expanded).

    This is recognized and respected by the instantiation parser.
    Implementationally, no-op (returns the value passed in as an
    argument).

    Parameters
    ----------
    value : object
        The value to be returned.

    Returns
    -------
    value : object
        The same object passed in as an argument.

    Notes
    -----
    Taken (with minor changes) from `Pylearn2`_.

    .. _Pylearn2: \
    http://github.com/lisa-lab/pylearn2/blob/master/pylearn2/config/yaml_parse.py

    """
    return value


def _instantiate_proxy_tuple(proxy, bindings=None):
    """ Helper function for `_instantiate` that handles objects of the
     `_Proxy` class.

    Parameters
    ----------
    proxy : _Proxy object
        A `_Proxy` object that.
    bindings : dict, optional
        A dictionary mapping previously instantiated `_Proxy` objects
        to their instantiated values.

    Returns
    -------
    obj : object
        The result object from recursively instantiating the object DAG.

    Notes
    -----
    Taken (with minor changes) from `Pylearn2`_.

    .. _Pylearn2: \
    http://github.com/lisa-lab/pylearn2/blob/master/pylearn2/config/yaml_parse.py

    """
    if proxy in bindings:
        return bindings[proxy]
    else:
        # Respect _do_not_recurse by just un-packing it (same as calling).
        if proxy.callable == _do_not_recurse:
            obj = proxy.keywords['value']
        else:
            if len(proxy.positionals) > 0:
                raise NotImplementedError('positional arguments not yet '
                                          'supported in proxy instantiation')
            kwargs = dict((k, instantiate(v, bindings))
                          for k, v in six.iteritems(proxy.keywords))
            obj = proxy.callable(**kwargs)
        try:
            obj.yaml_src = proxy.yaml_src
        except AttributeError:  # Some classes won't allow this.
            pass
        bindings[proxy] = obj
        return bindings[proxy]


def _preprocess(string, environ=None):
    """Preprocesses a string.

    Preprocesses a string, by replacing `${VARNAME}` with
    `os.environ['VARNAME']` and ~ with the path to the user's
    home directory.

    Parameters
    ----------
    string : str
        String object to _preprocess
    environ : dict, optional
        If supplied, preferentially accept values from
        this dictionary as well as `os.environ`. That is,
        if a key appears in both, this dictionary takes
        precedence.

    Returns
    -------
    rval : str
        The preprocessed string

    Notes
    -----
    Taken (with minor changes) from `Pylearn2`_.

    .. _Pylearn2: \
    http://github.com/lisa-lab/pylearn2/blob/master/pylearn2/config/yaml_parse.py

    """
    if environ is None:
        environ = {}

    split = string.split('${')

    rval = [split[0]]

    for candidate in split[1:]:
        subsplit = candidate.split('}')

        if len(subsplit) < 2:
            raise ValueError('Open ${ not followed by } before '
                             'end of string or next ${ in "' + string + '"')

        varname = subsplit[0]
        val = (environ[varname] if varname in environ
               else os.environ[varname])
        rval.append(val)

        rval.append('}'.join(subsplit[1:]))

    rval = ''.join(rval)

    string = os.path.expanduser(string)

    return rval


def instantiate(proxy, bindings=None):
    """Instantiate a hierarchy of proxy objects.

    Parameters
    ----------
    proxy : object
        A `_Proxy` object or list/dict/literal. Strings are run through
        `_preprocess`.
    bindings : dict, optional
        A dictionary mapping previously instantiated `_Proxy` objects
        to their instantiated values.

    Returns
    -------
    obj : object
        The result object from recursively instantiating the object DAG.

    Notes
    -----
    Taken (with minor changes) from `Pylearn2`_.

    .. _Pylearn2: \
    http://github.com/lisa-lab/pylearn2/blob/master/pylearn2/config/yaml_parse.py

    """
    if bindings is None:
        bindings = {}
    if isinstance(proxy, _Proxy):
        return _instantiate_proxy_tuple(proxy, bindings)
    elif isinstance(proxy, dict):
        # Recurse on the keys too, for backward compatibility.
        # Is the key instantiation feature ever actually used, by anyone?
        return dict((instantiate(k, bindings), instantiate(v, bindings))
                    for k, v in six.iteritems(proxy))
    elif isinstance(proxy, list):
        return [instantiate(v, bindings) for v in proxy]
    # In the future it might be good to consider a dict argument that provides
    # a type->callable mapping for arbitrary transformations like this.
    elif isinstance(proxy, six.string_types):
        return _preprocess(proxy)
    else:
        return proxy


def load(stream, environ=None, instantiate=True, **kwargs):
    """Loads a YAML configuration from a string or file-like object.

    Parameters
    ----------
    stream : str or object
        Either a string containing valid YAML or a file-like object
        supporting the `.read()` interface.
    environ : dict, optional
        A dictionary used for ${FOO} substitutions in addition to
        environment variables. If a key appears both in `os.environ`
        and this dictionary, the value in this dictionary is used.
    instantiate : bool, optional
        If `False`, do not actually instantiate the objects but instead
        produce a nested hierarchy of `_Proxy` objects.
    **kwargs : dict
        Other keyword arguments, all of which are passed to `yaml.load`.

    Returns
    -------
    graph : dict or object
        The dictionary or object (if the top-level element specified
        a Python object to instantiate), or a nested hierarchy of
        `_Proxy` objects.

    Notes
    -----
    Taken (with minor changes) from `Pylearn2`_.

    .. _Pylearn2: \
    http://github.com/lisa-lab/pylearn2/blob/master/pylearn2/config/yaml_parse.py

    """
    global IS_INITIALIZED
    if not IS_INITIALIZED:
        _initialize()

    if isinstance(stream, six.string_types):
        string = stream
    else:
        string = stream.read()

    proxy_graph = yaml.load(string, **kwargs)
    if instantiate:
        return instantiate(proxy_graph)
    else:
        return proxy_graph


def load_path(path, environ=None, instantiate=True, **kwargs):
    """Convenience function for loading a YAML configuration from a
    file.

    Parameters
    ----------
    path : str
        The path to the file to load on disk.
    environ : dict, optional
        A dictionary used for ${FOO} substitutions in addition to
        environment variables. If a key appears both in `os.environ`
        and this dictionary, the value in this dictionary is used.
    instantiate : bool, optional
        If `False`, do not actually instantiate the objects but instead
        produce a nested hierarchy of `_Proxy` objects.
    **kwargs : dict
        Other keyword arguments, all of which are passed to `yaml.load`.

    Returns
    -------
    graph : dict or object
        The dictionary or object (if the top-level element specified
        a Python object to instantiate), or a nested hierarchy of
        `_Proxy` objects.

    Notes
    -----
    Taken (with minor changes) from `Pylearn2`_.

    .. _Pylearn2: \
    http://github.com/lisa-lab/pylearn2/blob/master/pylearn2/config/yaml_parse.py


    """
    with open(path, 'r') as f:
        content = ''.join(f.readlines())

    # This is apparently here to avoid the odd instance where a file gets
    # loaded as Unicode instead (see 03f238c6d). It's rare instance where
    # basestring is not the right call.
    if not isinstance(content, str):
        raise AssertionError("Expected content to be of type str, got " +
                             str(type(content)))

    return load(content, instantiate=instantiate, environ=environ, **kwargs)


def _try_to_import(tag_suffix):
    components = tag_suffix.split('.')
    module_name = '.'.join(components[:-1])
    try:
        exec("import {0}".format(module_name))
    except ImportError as e:
        # We know it's an ImportError, but is it an ImportError related to
        # this path,
        # or did the module we're importing have an unrelated ImportError?
        # and yes, this test can still have false positives, feel free to
        # improve it
        pieces = module_name.split('.')
        str_e = str(e)
        found = True in [piece.find(str(e)) != -1 for piece in pieces]

        if found:
            # The yaml file is probably to blame.
            # Report the problem with the full module path from the YAML
            # file
            raise ImportError(
                "Could not import {0}; ImportError was {1}".format(
                    module_name, str_e))
        else:
            pcomponents = components[:-1]
            assert len(pcomponents) >= 1
            j = 1
            while j <= len(pcomponents):
                module_name = '.'.join(pcomponents[:j])
                try:
                    exec("import {0}".format(module_name))
                except Exception:
                    base_msg = "Could not import {0}".format(module_name)
                    if j > 1:
                        module_name = '.'.join(pcomponents[:j - 1])
                        base_msg += " but could import {0}".format(module_name)
                    raise ImportError(
                        "{0}. Original exception: {1}".format(base_msg,
                                                              str(e)))
                j += 1
    try:
        obj = eval(tag_suffix)
    except AttributeError as e:
        try:
            # Try to figure out what the wrong field name was
            # If we fail to do it, just fall back to giving the usual
            # attribute error
            pieces = tag_suffix.split('.')
            module = '.'.join(pieces[:-1])
            field = pieces[-1]
            candidates = dir(eval(module))

            msg = ("Could not evaluate {0}. "
                   "Did you mean {1}? "
                   "Original error was {2}".format(
                       tag_suffix, candidates, str(e)
                   ))

        except Exception:
            warnings.warn("Attempt to decipher AttributeError failed")
            raise AttributeError("Could not evaluate {0}. " +
                                 "Original error was {1}".format(
                                     tag_suffix, str(e)))
        raise AttributeError(msg)
    return obj


def _initialize():
    """
    Notes
    -----
    Taken (with minor changes) from `Pylearn2`_.

    .. _Pylearn2: \
    http://github.com/lisa-lab/pylearn2/blob/master/pylearn2/config/yaml_parse.py
    """
    global IS_INITIALIZED
    yaml.add_multi_constructor("!obj:", _multi_constructor_obj)
    yaml.add_multi_constructor("!import:", _multi_constructor_import)

    yaml.add_constructor("!import", _constructor_import)
    yaml.add_constructor("!float", _constructor_float)

    pattern = re.compile(SCIENTIFIC_NOTATION_REGEXP)
    yaml.add_implicit_resolver("!float",  pattern)
    IS_INITIALIZED = True


def _multi_constructor_obj(loader, tag_suffix, node):
    """
    Notes
    -----
    Taken (with minor changes) from `Pylearn2`_.

    .. _Pylearn2: \
    http://github.com/lisa-lab/pylearn2/blob/master/pylearn2/config/yaml_parse.py

    """
    yaml_src = yaml.serialize(node)
    _construct_mapping(node)
    mapping = loader.construct_mapping(node)

    assert hasattr(mapping, 'keys')
    assert hasattr(mapping, 'values')

    for key in mapping.keys():
        if not isinstance(key, six.string_types):
            raise TypeError(
                "Received non string object ({0}) as key in mapping.".format(
                    str(key)
                ))
    if '.' not in tag_suffix:
        # I'm not sure how this was ever working without eval().
        callable = eval(tag_suffix)
    else:
        callable = _try_to_import(tag_suffix)
    rval = _Proxy(callable=callable, yaml_src=yaml_src, positionals=(),
                  keywords=mapping)
    return rval


def _multi_constructor_import(loader, tag_suffix, node):
    """Callback for "!import:" tag.

    Notes
    -----
    Taken (with minor changes) from `Pylearn2`_.

    .. _Pylearn2: \
    http://github.com/lisa-lab/pylearn2/blob/master/pylearn2/config/yaml_parse.py

    """
    if '.' not in tag_suffix:
        raise yaml.YAMLError("!import: tag suffix contains no'.'")
    return _try_to_import(tag_suffix)


def _constructor_import(loader, node):
    """Callback for "!import"

    Notes
    -----
    Taken (with minor changes) from`Pylearn2`_.

    .. _Pylearn2: \
    http://github.com/lisa-lab/pylearn2/blob/master/pylearn2/config/yaml_parse.py

    """
    val = loader.construct_scalar(node)
    if '.' not in val:
        raise yaml.YAMLError("Import tag suffix contains no '.'")
    return _try_to_import(val)


def _constructor_float(loader, node):
    """Callback for "!float"

    Notes
    -----
    Taken (with minor changes) from `Pylearn2`_.

    .. _Pylearn2: \
    http://github.com/lisa-lab/pylearn2/blob/master/pylearn2/config/yaml_parse.py

    """
    val = loader.construct_scalar(node)
    return float(val)


def _construct_mapping(node, deep=False):
    """This is a modified version of
    `yaml.BaseConstructor._construct_mapping` only
    permitting unique keys.

    Notes
    -----
    Taken (with minor changes) from `Pylearn2`_.

    .. _Pylearn2: \
    http://github.com/lisa-lab/pylearn2/blob/master/pylearn2/config/yaml_parse.py

    """
    if not isinstance(node, yaml.nodes.MappingNode):
        const = yaml.constructor
        raise Exception(
            "Expected a mapping node, but found {0} {1}.".format(
                node.id, node.start_mark
            ))
    mapping = {}
    constructor = yaml.constructor.BaseConstructor()
    for key_node, value_node in node.value:
        key = constructor.construct_object(key_node, deep=False)
        try:
            hash(key)
        except TypeError as exc:
            const = yaml.constructor
            raise Exception("While constructing a mapping " +
                            "{0}, found unacceptable " +
                            "key ({1}).".format(
                                node.start_mark, (exc, key_node.start_mark)))
        if key in mapping:
            const = yaml.constructor
            raise Exception("While constructing a mapping " +
                            "{0}, found duplicate " +
                            "key ({1}).".format(node.start_mark, key))
        value = constructor.construct_object(value_node, deep=False)
        mapping[key] = value
    return mapping
