# -*- coding: utf-8 -*-
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='pimtorch',
      version='1.0',
      description='Python API for pimtorch',
      author='Heng Zhou',
      author_email='heng_zhou@hust.edu.cn',
      ext_modules=[cpp_extension.CppExtension('pimtorch', ['pimtorch.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})