import os
from pkg_resources import resource_filename


def ensure_dir_exists(path):
    """Make sure a directory exists, create if necessary."""
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def add_prefix(filename, prefix=None):
    if prefix is not None and prefix != "":
        filename = '_'.join((str(prefix), filename))
    return filename


def print_config_dir():
    """Get directory of panga configs using pkg_resources"""
    print resource_filename('panga', os.path.join('data', 'config'))
