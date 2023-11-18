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

Additionally, we allow for capital growth factor and arbitrary emission reduction constraints to be considered.

The `Pareto front`_, the set of solutions where one objective can't be improved without worsening the other objective,
can be computed for return on capital and diversification. 

The codebase is based on the following paper:

- Aguilera et al., - Multi-objective Portfolio Optimisation Using the Quantum Annealer (2023)

.. _Herfindahl-Hirschman Index: https://en.wikipedia.org/wiki/Herfindahl%E2%80%93Hirschman_index
.. _Pareto front: https://en.wikipedia.org/wiki/Pareto_front
.. _QUBO: https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization


Quick Install
-------------
The portfolio optimization module can be installed using pip as follows::

    pip install tno.quantum.problems.portfolio_optimization

Examples
--------

Here's an example of how the :py:class:`~portfolio_optimization.portfolio_optimizer.PortfolioOptimizer` class 
can be used to define an portfolio optimization problem, and subsequently, how the Pareto front be computed 
using the simulated annealing sampler from D-Wave. 


.. code-block:: python

    import numpy as np
    from dwave.samplers import SimulatedAnnealingSampler

    from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer

    # Choose sampler and solve qubo.
    sampler = SimulatedAnnealingSampler()
    sampler_kwargs = {"num_reads": 20, "num_sweeps": 200}

    # Set up penalty coefficients for the constraints
    lambdas1 = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
    lambdas2 = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
    lambdas3 = np.array([1])

    # Create portfolio optimization problem
    portfolio_optimizer = PortfolioOptimizer("rabobank")
    portfolio_optimizer.add_minimize_hhi(weights=lambdas1)
    portfolio_optimizer.add_maximize_roc(formulation=1, weights_roc=lambdas1)
    portfolio_optimizer.add_emission_constraint(
        weights=lambdas3,
        variable_now="emis_intens_now",
        variable_future="emis_intens_future",
    )

    # Solve the portfolio optimization problem
    results = portfolio_optimizer.run(sampler, sampler_kwargs)


Data input
----------

TODO: Explain dataset constrains + refer to docs.


Using Quantum Annealing Solvers
-------------------------------

By default, the portfolio optimization QUBO is solved using simulated annealing.
Any D-Wave ``Sampler`` is however supported and can be provided to the :py:meth:`~portfolio_optimization.portfolio_optimizer.PortfolioOptimizer.run` method.
 

Below is an example how to initialise a quantum annealing sampler that uses `100` micro seconds annealing time per sample.
The example assumes a proper `configuration setup`_ to the D-Wave's Solver API.

.. code-block:: python

    from dwave.system import DWaveSampler, EmbeddingComposite

    # Define QPU D-Wave Sampler
    qpu = DWaveSampler()
    sampler = EmbeddingComposite(qpu)
    sampler_kwargs = {"annealing_time": 100}


We refer to the `D-Wave Sampler documentation`_ for information on usage of different samplers and their sampler arguments.

.. _configuration setup: https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html
.. _D-Wave Sampler documentation: https://docs.ocean.dwavesys.com/projects/system/en/stable/reference/samplers.html