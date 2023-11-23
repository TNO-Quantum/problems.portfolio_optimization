"""This package provides Python code that converts the multi-objective portfolio
optimization problem into a `QUBO`_ problem. The transformed problem can then be solved
using quantum annealing techniques.

The following objectives can be considered

- `return on capital`, indicated by ROC,
- `diversification`, indicated by the `Herfindahl-Hirschman Index`_ HHI.

Additionally, we allow for capital growth factor and arbitrary emission reduction 
constraints to be considered.

The `Pareto front`_, the set of solutions where one objective can't be improved without
worsening the other objective, can be computed for return on capital and diversification. 

The codebase is based on the following paper:

- Aguilera et al., - Multi-objective Portfolio Optimisation Using the Quantum Annealer (2023)

.. _Pareto front: https://en.wikipedia.org/wiki/Pareto_front
.. _QUBO: https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization"""

from tno.quantum.problems.portfolio_optimization._portfolio_optimizer import (
    PortfolioOptimizer,
)
from tno.quantum.problems.portfolio_optimization.components import (
    plot_front,
    plot_points,
)

__all__ = ["PortfolioOptimizer", "plot_front", "plot_points"]

__version__ = "0.0.1"
