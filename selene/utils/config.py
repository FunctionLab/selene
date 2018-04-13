"""
Classes and methods for loading configurations from YAML files.
Taken (with minor changes) from: https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/config/yaml_parse.py
"""
import os
import re
import warnings
import yaml
import six
from collections import namedtuple

SCIENTIFIC_NOTATION_REGEXP = r"^[\-\+]?(\d+\.?\d*|\d*\.?\d+)?[eE][\-\+]?\d+$"
IS_INITIALIZED = False


BaseProxy = namedtuple("BaseProxy", ["callable", "positionals", "keywords", "yaml_src"])


class Proxy(BaseProxy):
    """
    An intermediate representation between initial YAML parse and object
    instantiation.
    Parameters
    ----------
    callable : callable
        The function/class to call to instantiate this node.
    positionals : iterable
        Placeholder for future support for positional arguments (`*args`).
    keywords : dict-like
        A mapping from keywords to arguments (`**kwargs`), which may be
        `Proxy`s or `Proxy`s nested inside `dict` or `list` instances.
        Keys must be strings that are valid Python variable names.
    yaml_src : str
        The YAML source that created this node, if available.
    Notes
    -----
    This is intended as a robust, forward-compatible intermediate
    representation for either internal consumption or external consumption
    by another tool e.g. hyperopt.
    This particular class mainly exists to  override `BaseProxy`'s `__hash__`
    (to avoid hashing unhashable namedtuple elements).
    """
    __slots__ = []

    def __hash__(self):
        """
        Return a hash based on the object ID (to avoid hashing unhashable
        namedtuple elements).
        """
        return hash(id(self))

    def bind(self, **kwargs):
        for k in kwargs:
            if k not in self.keywords:
                self.keywords[k] = kwargs[k]

    def pop(self, key):
        return self.keywords.pop(key)


def do_not_recurse(value):
    """
    Function symbol used for wrapping an unpickled object (which should
    not be recursively expanded). This is recognized and respected by the
    instantiation parser. Implementationally, no-op (returns the value
    passed in as an argument).
    Parameters
    ----------
    value : object
        The value to be returned.
    Returns
    -------
    value : object
        The same object passed in as an argument.
    """
    return value


def _instantiate_proxy_tuple(proxy, bindings=None):
    """
    Helper function for `_instantiate` that handles objects of the `Proxy`
    class.
    Parameters
    ----------
    proxy : Proxy object
        A `Proxy` object that.
    bindings : dict, optional
        A dictionary mapping previously instantiated `Proxy` objects
        to their instantiated values.
    Returns
    -------
    obj : object
        The result object from recursively instantiating the object DAG.
    """
    if proxy in bindings:
        return bindings[proxy]
    else:
        # Respect do_not_recurse by just un-packing it (same as calling).
        if proxy.callable == do_not_recurse:
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


def preprocess(string, environ=None):
    """
    Preprocesses a string, by replacing `${VARNAME}` with
    `os.environ['VARNAME']` and ~ with the path to the user's
    home directory
    Parameters
    ----------
    string : str
        String object to preprocess
    environ : dict, optional
        If supplied, preferentially accept values from
        this dictionary as well as `os.environ`. That is,
        if a key appears in both, this dictionary takes
        precedence.
    Returns
    -------
    rval : str
        The preprocessed string
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
    """
    Instantiate a (hierarchy of) Proxy object(s).
    Parameters
    ----------
    proxy : object
        A `Proxy` object or list/dict/literal. Strings are run through
        `preprocess`.
    bindings : dict, optional
        A dictionary mapping previously instantiated `Proxy` objects
        to their instantiated values.
    Returns
    -------
    obj : object
        The result object from recursively instantiating the object DAG.
    Notes
    -----
    This should not be considered part of the stable, public API.
    """
    if bindings is None:
        bindings = {}
    if isinstance(proxy, Proxy):
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
        return preprocess(proxy)
    else:
        return proxy


def load(stream, environ=None, instantiate=True, **kwargs):
    """
    Loads a YAML configuration from a string or file-like object.
    Parameters
    ----------
    stream : str or object
        Either a string containing valid YAML or a file-like object
        supporting the .read() interface.
    environ : dict, optional
        A dictionary used for ${FOO} substitutions in addition to
        environment variables. If a key appears both in `os.environ`
        and this dictionary, the value in this dictionary is used.
    instantiate : bool, optional
        If `False`, do not actually instantiate the objects but instead
        produce a nested hierarchy of `Proxy` objects.
    Returns
    -------
    graph : dict or object
        The dictionary or object (if the top-level element specified
        a Python object to instantiate), or a nested hierarchy of
        `Proxy` objects.
    Notes
    -----
    Other keyword arguments are passed on to `yaml.load`.
    """
    global IS_INITIALIZED
    if not IS_INITIALIZED:
        initialize()

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
    """
    Convenience function for loading a YAML configuration from a file.
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
        produce a nested hierarchy of `Proxy` objects.
    Returns
    -------
    graph : dict or object
        The dictionary or object (if the top-level element specified
        a Python object to instantiate), or a nested hierarchy of
        `Proxy` objects.
    Notes
    -----
    Other keyword arguments are passed on to `yaml.load`.
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


def try_to_import(tag_suffix):
    components = tag_suffix.split('.')
    module_name = '.'.join(components[:-1])
    try:
        exec(f"import {module_name}")
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
            raise ImportError(f"Could not import {module_name}; ImportError was {str_e}")
        else:
            pcomponents = components[:-1]
            assert len(pcomponents) >= 1
            j = 1
            while j <= len(pcomponents):
                module_name = '.'.join(pcomponents[:j])
                try:
                    exec(f"import {module_name}")
                except Exception:
                    base_msg = f"Could not import {module_name}"
                    if j > 1:
                        module_name = '.'.join(pcomponents[:j - 1])
                        base_msg += f" but could import {module_name}"
                    raise ImportError(f"{base_msg}. Original exception: {str(e)}")
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

            msg = (f"Could not evaluate {tag_suffix}. " +
                   f"Did you mean {match(field, candidates)}? " +
                   f"Original error was {str(e)}")

        except Exception:
            warnings.warn("Attempt to decipher AttributeError failed")
            raise AttributeError(f"Could not evaluate {tag_suffix}. " +
                                      f"Original error was {str(e)}")
        raise AttributeError(msg)
    return obj


def initialize():
    global IS_INITIALIZED
    yaml.add_multi_constructor("!obj:", multi_constructor_obj)
    yaml.add_multi_constructor("!import:", multi_constructor_import)

    yaml.add_constructor("!import", constructor_import)
    yaml.add_constructor("!float", constructor_float)

    pattern = re.compile(SCIENTIFIC_NOTATION_REGEXP)
    yaml.add_implicit_resolver("!float",  pattern)
    IS_INITIALIZED = True


def multi_constructor_obj(loader, tag_suffix, node):
    yaml_src = yaml.serialize(node)
    construct_mapping(node)
    mapping = loader.construct_mapping(node)

    assert hasattr(mapping, 'keys')
    assert hasattr(mapping, 'values')

    for key in mapping.keys():
        if not isinstance(key, six.string_types):
            raise TypeError(f"Received non string object ({str(key)}) as key in mapping.")
    if '.' not in tag_suffix:
        # TODO: I'm not sure how this was ever working without eval().
        callable = eval(tag_suffix)
    else:
        callable = try_to_import(tag_suffix)
    rval = Proxy(callable=callable, yaml_src=yaml_src, positionals=(),
                 keywords=mapping)
    return rval


def multi_constructor_import(loader, tag_suffix, node):
    """
    Callback for "!import:" tag.
    """
    if '.' not in tag_suffix:
        raise yaml.YAMLError("!import: tag suffix contains no'.'")
    return try_to_import(tag_suffix)


def constructor_import(loader, node):
    """
    Callback for "!import"
    """
    val = loader.construct_scalar(node)
    if '.' not in val:
        raise yaml.YAMLError("Import tag suffix contains no '.'")
    return try_to_import(val)


def constructor_float(loader, node):
    """
    Callback for "!float"
    """
    val = loader.construct_scalar(node)
    return float(val)


def construct_mapping(node, deep=False):
    """
    This is a modified version of yaml.BaseConstructor.construct_mapping only permitting unique keys.
    """
    if not isinstance(node, yaml.nodes.MappingNode):
        const = yaml.constructor
        raise Exception(f"Expected a mapping node, but found {node.id} {node.start_mark}.")
    mapping = {}
    constructor = yaml.constructor.BaseConstructor()
    for key_node, value_node in node.value:
        key = constructor.construct_object(key_node, deep=False)
        try:
            hash(key)
        except TypeError as exc:
            const = yaml.constructor
            raise Exception(f"While constructing a mapping {node.start_mark}, found unacceptable " +
                            f"key ({(exc, key_node.start_mark)}).")
        if key in mapping:
            const = yaml.constructor
            raise Exception(f"While constructing a mapping {node.start_mark}, found duplicate " +
                            f"key ({key}).")
        value = constructor.construct_object(value_node, deep=False)
        mapping[key] = value
    return mapping
