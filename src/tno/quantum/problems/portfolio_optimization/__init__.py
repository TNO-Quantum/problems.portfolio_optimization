"""Multi-objective portfolio optimization using QUBO formulation.

This package provides Python code that converts a multi-objective portfolio optimization
problem into a `QUBO`_ problem. The transformed problem can then be solved using quantum
annealing techniques.

The following objectives can be considered

- `return on capital`, indicated by ROC,
- `diversification`, indicated by the `Herfindahl-Hirschman Index`_ HHI.

Additionally, we allow for capital growth factor and arbitrary emission reduction
constraints to be considered.

Usage example:
--------------

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

The `Pareto front`_, the set of solutions where one objective can't be improved without
worsening the other objective, can be computed for return on capital and diversification.

>>> import matplotlib.pyplot as plt
>>> from tno.quantum.problems.portfolio_optimization import plot_front, plot_points
>>>
>>> (x1, y1), (x2, y2) = results.slice_results()
>>> fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
>>>
>>> # Plot data points
>>> plot_points(x2, y2, color="orange", label="QUBO constraint not met", ax=ax1)  # doctest: +SKIP
>>> plot_points(x1, y1, color="green", label="QUBO constraint met", ax=ax1)  # doctest: +SKIP
>>> ax1.set_title("Points")  # doctest: +SKIP
>>>
>>> # Plot Pareto front
>>> plot_front(x2, y2, color="orange", label="QUBO constraint not met", ax=ax2)  # doctest: +SKIP
>>> plot_front(x1, y1, color="green", label="QUBO constraint met", ax=ax2)  # doctest: +SKIP
>>> ax2.set_title("Pareto Front")  # doctest: +SKIP
>>> fig.tight_layout()
>>> plt.show()  # doctest: +SKIP

.. image:: img/example.png
    :width: 1200
    :align: center
    :alt: (Diversification, ROC)-Graph

Data input
----------

The data used for the portfolio optimization can be imported via an excel file, csv file,
json file or as a Pandas :class:`~pandas.DataFrame`.
The data needs to contain at least the following columns:

    - ``asset``: The name of the asset.
    - ``outstanding_now``: Current outstanding amount per asset.
    - ``min_outstanding_future``: Lower bound outstanding amount in the future per asset.
    - ``max_outstanding_future``: Upper bound outstanding amount in the future per asset.
    - ``income_now``: Current income per asset, corresponds to return multiplied by the current outstanding amount.
    - ``regcap_now``: Current regulatory capital per asset.

The table below shows an example dataset with the correct structure.
Note that this is the least amount of columns that need to be present.
More columns are allowed and required for some functionalities.

.. list-table:: Example Dataset
   :widths: 25 25 25 25 25 25
   :header-rows: 1

   * - asset
     - outstanding_now
     - min_outstanding_future
     - max_outstanding_future
     - income_now
     - regcap_now
   * - Sector 1 COUNTRY 1
     - 10
     - 14
     - 19
     - 5
     - 5
   * - Sector 2 COUNTRY 1
     - 600
     - 473
     - 528
     - 70
     - 40
   * - Sector 3 COUNTRY 1
     - 20
     - 24
     - 28
     - 5
     - 10
   * - Sector 4 COUNTRY 1
     - 800
     - 1090
     - 1410
     - 1
     - 2
   * - Sector 1 COUNTRY 2
     - 40
     - 56
     - 74
     - 10
     - 5
   * - Sector 2 COUNTRY 2
     - 200
     - 291
     - 397
     - 40
     - 20
   * - ...
     - ...
     - ...
     - ...
     - ...
     - ...

If the input data file contains the correct information but has different column names,
you can rename the columns without modifying the input file. For more details and
examples, refer to the documentation of
:py:class:`~tno.quantum.problems.portfolio_optimization._components._io.PortfolioData`.

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

__version__ = "2.0.0-beta.1"
