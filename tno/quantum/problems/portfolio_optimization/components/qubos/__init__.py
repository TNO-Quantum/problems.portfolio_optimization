"""This subpackage contains QUBO related classes and methods.

The ``QuboCompiler`` can create a verity of QUBO formulation by combining different
objectives and constraints with their corresponding penalty or preference parameters.

The ``QuboFactory`` class provides a convenient interface for constructing intermediate
QUBO matrices for different objectives and constraints.
"""
from ._qubo_compiler import QuboCompiler
from ._qubo_factory import QuboFactory

__all__ = ["QuboCompiler", "QuboFactory"]
