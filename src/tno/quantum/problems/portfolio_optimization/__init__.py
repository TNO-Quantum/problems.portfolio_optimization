"""Multi-objective portfolio optimization using QUBO formulation.

This package provides Python code that converts a multi-objective portfolio optimization
problem into a `QUBO`_ problem. The transformed problem can then be solved using quantum
annealing techniques.

The following objectives can be considered

- `return on capital`, indicated by ROC,
- `diversification`, indicated by the `Herfindahl-Hirschman Index`_ HHI.

Additionally, we allow for capital growth factor and arbitrary emission reduction
constraints to be considered.

The `Pareto front`_, the set of solutions where one objective can't be improved without
worsening the other objective, can be computed for return on capital and diversification.

Usage example:

>>> import numpy as np
>>> from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer
>>> from tno.quantum.optimization.qubo.solvers import SimulatedAnnealingSolver
>>>
>>> # Choose sampler for solving qubo
>>> solver = SimulatedAnnealingSolver(num_reads=20, num_sweeps=200)
>>>
>>> # Set up penalty coefficients for the constraints
>>> lambdas1 = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
>>> lambdas2 = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
>>> lambdas3 = np.array([1])
>>>
>>> # Create portfolio optimization problem
>>> portfolio_optimizer = PortfolioOptimizer("benchmark_dataset")
>>> portfolio_optimizer.add_minimize_hhi(weights=lambdas1)
>>> portfolio_optimizer.add_maximize_roc(formulation=1, weights_roc=lambdas2)
>>> portfolio_optimizer.add_emission_constraint(
...     weights=lambdas3,
...     emission_now="emis_intens_now",
...     emission_future="emis_intens_future",
...     name="emission",
... )
>>>
>>> # Solve the portfolio optimization problem
>>> results = portfolio_optimizer.run(solver, verbose=True)
>>> print(results.head())  # doctest: +SKIP
                                  outstanding amount  diff ROC  diff diversification  diff outstanding  diff emission
0  (14.0, 473.0, 26.666666666666668, 1410.0, 74.0...  4.105045             -6.102454          1.514694     -29.999998
1  (19.0, 473.0, 28.0, 1196.6666666666667, 68.0, ...  2.574088             -2.556330          1.520952     -29.999992
2  (17.333333333333332, 509.6666666666667, 24.0, ...  2.979830             -6.397679          1.566499     -29.999988
3  (15.666666666666666, 491.3333333333333, 25.333...  1.875721             -4.025964          1.531100     -30.000023
4  (15.666666666666666, 491.3333333333333, 24.0, ...  2.697235             -7.117611          1.555159     -29.999977

The codebase is based on the following paper:

- `Aguilera et al., - Multi-objective Portfolio Optimisation Using the Quantum Annealer (2024)`_

.. _Aguilera et al., - Multi-objective Portfolio Optimisation Using the Quantum Annealer (2024): https://www.mdpi.com/2227-7390/12/9/1291
.. _Pareto front: https://en.wikipedia.org/wiki/Pareto_front
.. _QUBO: https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization
"""  # noqa: E501

from tno.quantum.problems.portfolio_optimization._components import (
    PortfolioData,
    QuboFactory,
    plot_front,
    plot_points,
)
from tno.quantum.problems.portfolio_optimization._portfolio_optimizer import (
    PortfolioOptimizer,
)

__all__ = [
    "PortfolioData",
    "PortfolioOptimizer",
    "QuboFactory",
    "plot_front",
    "plot_points",
]

__version__ = "1.0.1-beta.1"
