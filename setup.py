from distutils.core import setup
from Cython.Build import cythonize

setup(name="data_utils.fastloop", ext_modules=cythonize('data_utils/fastloop.pyx'),)
