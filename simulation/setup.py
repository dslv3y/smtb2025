from setuptools import setup, Extension
import numpy
import pybind11

extension_mod = Extension(
    "diffusion_cpp",
    ["diffusion_cpp.cpp"],
    include_dirs=[numpy.get_include(), pybind11.get_include()],
    language="c++",
    extra_compile_args=["-std=c++17", "-O3"],
)

setup(
    name="diffusion_cpp",
    version="1.0",
    description="A C++ extension for diffusion simulation",
    ext_modules=[extension_mod],
)
