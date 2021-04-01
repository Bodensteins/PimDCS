# -*- coding: utf-8 -*-
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='logic_array_interface_cpp',
      ext_modules=[cpp_extension.CppExtension('logic_array_interface_cpp', ['logic_array_interface.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})