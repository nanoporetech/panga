import os
import re
import numpy
from setuptools import setup, find_packages, Command, Extension
from distutils.core import setup
from Cython.Build import cythonize

__pkg_name__ = 'panga'

# Get the version number from __init__.py
verstrline = open(os.path.join(__pkg_name__, '__init__.py'), 'r').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "{}/__init__.py".'.format(__pkg_name__))


# extensions


class CythonCommand(Command):
    """ Additional command for building pyx files into c code.
    """

    description = 'Build all Cython code into C code.'
    user_options = []

    def initialize_options(self):
        return

    def finalize_options(self):
        return

    def run(self):
        import Cython
        from Cython.Build import cythonize
        print 'Found Cython, version {}'.format(Cython.__version__)
        Cython.Compiler.Options.annotate = True
        print 'Cythonizing pyx files.'
        cythonize(cython_modules, force=True)
        return


extensions = []
OPTIMISATION = [ '-O3', '-DNDEBUG', '-fstrict-aliasing' ]
pyx_compile_args = ['-w'] + OPTIMISATION

cython_modules = [
    Extension('panga.algorithm.read_bounds_from_delta',
              [os.path.join('panga','algorithm','read_bounds_from_delta.pyx')],
              include_dirs=[numpy.get_include()],
              extra_compile_args=pyx_compile_args),
]

cython_extensions = [
    Extension('panga.algorithm.read_bounds_from_delta',
              [os.path.join('panga','algorithm','read_bounds_from_delta.c')],
              include_dirs=[numpy.get_include()],
              extra_compile_args=pyx_compile_args),
]
extensions.extend(cython_extensions)

dir_path = os.path.dirname(__file__)
with open(os.path.join(dir_path, 'requirements.txt')) as fh:
    install_requires = fh.read().splitlines()

extra_requires = {
}

setup(
    name=__pkg_name__,
    version=__version__,
    url='https://git.oxfordnanolabs.local/research/{}'.format(__pkg_name__),
    author='cwright',
    author_email='cwright@nanoporetech.com',
    maintainer='cwright',
    maintainer_email='cwright@nanoporetech.com',
    description='Generic system for classifying channel states.',
    dependency_links=[],
    install_requires=install_requires,
    tests_require=['nose>=1.3.7'].extend(install_requires),
    extras_require=extra_requires,
    packages=find_packages(exclude=['*.test', '*.test.*', 'test.*', 'test']),
    package_data={__pkg_name__:[os.path.join('data', 'config', '*.yml')]},
    zip_safe=True,
    test_suite='discover_tests',
    entry_points={
        'console_scripts': [
            'read_builder = panga.read_builder:main',
            'panga_config_dir = panga.util:print_config_dir',
        ]
    },
    cmdclass={'use_cython': CythonCommand},
    ext_modules=extensions,
)
