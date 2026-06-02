import contextlib
from shutil import rmtree
from tempfile import mkdtemp


@contextlib.contextmanager
def make_temp_directory():
    temp_dir = mkdtemp()
    try:
        yield temp_dir
    finally:
        rmtree(temp_dir)
