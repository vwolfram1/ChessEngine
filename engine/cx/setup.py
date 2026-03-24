"""
Build script for NexusChess Cython extension.

Usage (from engine/cx/ directory):
    python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import sys
import os

# Aggressive optimisation flags
extra_compile_args = ["/O2", "/arch:AVX2"] if sys.platform == "win32" else ["-O3", "-march=native", "-ffast-math"]
extra_link_args    = [] if sys.platform == "win32" else []

extensions = cythonize(
    [Extension(
        "cchess",
        sources=["cchess.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )],
    compiler_directives={
        "language_level": "3",
        "boundscheck":      False,
        "wraparound":       False,
        "nonecheck":        False,
        "cdivision":        True,
        "initializedcheck": False,
        "profile":          False,
    },
    annotate=False,
)

setup(
    name="nexus_cchess",
    ext_modules=extensions,
)
