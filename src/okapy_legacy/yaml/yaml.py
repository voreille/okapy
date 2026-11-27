import yaml
from yaml import *
import logging
import os.path
import functools

log = logging.getLogger(__name__)


class ExtLoaderMeta(type):

    def __new__(metacls, __name__, __bases__, __dict__):
        """Add include constructer to class."""

        # register the include constructor on the class
        cls = super().__new__(metacls, __name__, __bases__, __dict__)
        cls.add_constructor('!include', cls.construct_include)

        return cls


class ExtLoader(yaml.SafeLoader, metaclass=ExtLoaderMeta):
    """Safe YAML Loader with an `!include` constructor.

    Built on ``yaml.SafeLoader``, so ``!!python/...`` tags are rejected.
    ``!include`` paths are resolved relative to the including file's
    directory and must stay inside it.
    """

    def __init__(self, stream):
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

    def construct_include(self, node):
        """Include file referenced at node."""

        include = self.construct_scalar(node)
        root = os.path.realpath(self._root)
        filename = os.path.realpath(os.path.join(root, include))
        if os.path.commonpath([root, filename]) != root:
            raise yaml.constructor.ConstructorError(
                None, None,
                f"!include path {include!r} resolves outside the directory "
                f"of the including file ({root})",
                node.start_mark)
        extension = os.path.splitext(filename)[1].lstrip('.')

        with open(filename, 'r') as f:
            if extension in ('yaml', 'yml'):
                return yaml.load(f, ExtLoader)
            else:
                return ''.join(f.readlines())


# Set MyLoader as default.
load = functools.partial(yaml.load, Loader=ExtLoader)
