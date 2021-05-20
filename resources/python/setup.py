from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("train_c.pyx"))

