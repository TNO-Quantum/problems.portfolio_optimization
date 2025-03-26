Portfolio optimization
======================

Real-world investment decisions involve multiple, often conflicting, objectives that needs to be balanced.
Primary goals typically revolve around maximizing returns while minimizing risks.
At the same time, one might want to require additional constraints such as demanding a minimum carbon footprint reduction. 
Finding a portfolio that balances these objectives is a challenging task and can be solved using multi-objective portfolio optimization. 


This repository provides Python code that converts the multi-objective portfolio optimization problem
into a `QUBO`_ problem. The transformed problem can then be solved using quantum annealing techniques.

The following objectives can be considered

- `return on capital`, indicated by ROC,
- `diversification`, indicated by the `Herfindahl-Hirschman Index`_ HHI.

Additionally, we allow for a capital growth factor and arbitrary emission reduction constraints to be considered.

The `Pareto front`_, the set of solutions where one objective can't be improved without worsening the other objective,
can be computed for the objectives return on capital and diversification. 

The codebase is based on the following paper:

- `Aguilera et al., - Multi-objective Portfolio Optimisation Using the Quantum Annealer (2024)`_

.. _Aguilera et al., - Multi-objective Portfolio Optimisation Using the Quantum Annealer (2024): https://www.mdpi.com/2227-7390/12/9/1291
.. _Herfindahl-Hirschman Index: https://en.wikipedia.org/wiki/Herfindahl%E2%80%93Hirschman_index
.. _Pareto front: https://en.wikipedia.org/wiki/Pareto_front
.. _QUBO: https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization


**Funding:** This research was funded by Rabobank and Stichting TKI High Tech Systems
and Materials, under a program by Brightland's Techruption.

Quick Install
-------------
The portfolio optimization module can be installed using pip as follows::

    pip install tno.quantum.problems.portfolio_optimization

Examples
--------

Here's an example of how the :py:class:`~portfolio_optimization.PortfolioOptimizer` class 
can be used to define an portfolio optimization problem, and subsequently, how the Pareto front can be computed 
using a simulated annealing QUBO solver. 


.. code-block:: python

  import numpy as np

  from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer
  from tno.quantum.optimization.qubo import SolverConfig

  # Choose sampler for solving qubo
  solver_config = SolverConfig(
      name="simulated_annealing_solver", options={"num_reads": 20, "num_sweeps": 200}
  )
  solver = solver_config.get_instance()

  # Set up penalty coefficients for the constraints
  lambdas1 = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
  lambdas2 = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
  lambdas3 = np.array([1])

  # Create portfolio optimization problem
  portfolio_optimizer = PortfolioOptimizer("benchmark_dataset")
  portfolio_optimizer.add_minimize_hhi(weights=lambdas1)
  portfolio_optimizer.add_maximize_roc(formulation=1, weights_roc=lambdas2)
  portfolio_optimizer.add_emission_constraint(
      weights=lambdas3,
      emission_now="emis_intens_now",
      emission_future="emis_intens_future",
      name="emission",
  )

  # Solve the portfolio optimization problem
  results = portfolio_optimizer.run(solver, verbose=True)

  print(results.head())


The results can be inspected in more detail by looking at the Pandas results DataFrame
`results.results_df`.

Alternatively, the results can be plotted in a `(Diversification, ROC)`-graph. The
following example first slices the results in data points that do and do not satisfy the
constraints using the method :py:meth:`~portfolio_optimization.components.results.Results.slice_results()`. 

Note that:

- Individual data points can subsequently be plotted using :py:func:`~portfolio_optimization.components.visualization.plot_points()`
- The Pareto front can be plotted using :py:func:`~portfolio_optimization.components.visualization.plot_front()`

.. code-block:: python

    import matplotlib.pyplot as plt

    from tno.quantum.problems.portfolio_optimization import plot_front, plot_points

    (x1, y1), (x2, y2) = results.slice_results()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    # Plot data points
    plot_points(x2, y2, color="orange", label="QUBO constraint not met", ax=ax1)
    plot_points(x1, y1, color="green", label="QUBO constraint met", ax=ax1)
    ax1.set_title("Points")

    # Plot Pareto front
    plot_front(x2, y2, color="orange", label="QUBO constraint not met", ax=ax2)
    plot_front(x1, y1, color="green", label="QUBO constraint met", ax=ax2)
    ax2.set_title("Pareto Front")
    fig.tight_layout()
    plt.show()

.. image:: ../images_for_docs/example.png
    :width: 1200
    :align: center
    :alt: (Diversification, ROC)-Graph

More elaborate examples can be found in our `examples repository`_.

.. _examples repository: https://github.com/TNO-Quantum/examples 

Data input
----------

The data used for the portfolio optimization can be imported via an excel file, csv file,
json file or as a Pandas DataFrame.
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

If the input datafile contains all the correct information, but has different column
names, it is possible to rename the columns without altering the input file.
Details and examples can be found in the documentation of
:py:class:`~portfolio_optimization.components.io.PortfolioData`.


Using Quantum Annealing Solvers
-------------------------------

By default, the portfolio optimization QUBO is solved using simulated annealing.
Any TNO QUBO ``Solver`` is however supported and can be provided to the
:py:meth:`~portfolio_optimization.PortfolioOptimizer.run` method.
 

Below is an example how to initialise a quantum annealing sampler that uses `100` micro seconds annealing time per sample.
The example assumes a proper `configuration setup`_ to the D-Wave's Solver API.

.. code-block:: python

    from tno.quantum.optimization.qubo import SolverConfig

    # Instantiate QPU D-Wave Solver
    solver_config = SolverConfig(
        name="dwave_sampler_solver", options={"annealing_time": 100}
    )
    solver = solver_config.get_instance()


We refer to the `tno.quantum.optimization.qubo.solvers documentation`_ for information on usage of different samplers and their sampler arguments.

.. _configuration setup: https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html
.. _tno.quantum.optimization.qubo.solvers documentation: https://github.com/TNO-Quantum