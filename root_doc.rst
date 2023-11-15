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

Additionally, we allow for arbitrary constraints to be considered.

The `Pareto front`_, the set of solutions where one objective can't be improved without worsening the other objective,
can be computed for return on capital and diversification. 

The codebase is based on the following TNO papers:

- Phillipson et al.. - Portfolio Optimisation Using the D-Wave Quantum Annealer. (2021) (doi: `10.1007/978-3-030-77980-1 4`_)
- Phillipson et al., - Multi-objective Portfolio Optimisation Using the D-Wave Quantum Annealer (2023) (doi: TODO)


.. _10.1007/978-3-030-77980-1 4: https://doi.org/10.1007/978-3-030-77980-1_4
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
can be used to define an portfolio optimization problem, and subsequently, how its Pareto front be computed 
using a simulated annealing sampler from D-Wave. 


.. code-block:: python

    import numpy as np
    from dwave.samplers import SimulatedAnnealingSampler

    from tno.quantum.problems.portfolio_optimization import (
        PortfolioOptimizer,
        plot_front,
    )

    # Define the precision of the portfolio sizes.
    kmax = 2  # 2 # number of values
    kmin = 0  # minimal value 2**kmin


    # Choose sampler to solve qubo with
    sampler = SimulatedAnnealingSampler()
    sampler_kwargs = {"num_reads": 20, "num_sweeps": 200}


    # Set up penalty coefficients for the constraints
    lambdas1 = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
    lambdas2 = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
    lambdas3 = np.array([1])

    # Define the problem
    portfolio_optimizer = PortfolioOptimizer("rabobank", kmin, kmax)
    portfolio_optimizer.add_minimize_HHI(weights=lambdas1)
    portfolio_optimizer.add_maximize_ROC(formulation=1, weights_roc=lambdas1)
    portfolio_optimizer.add_emission_constraint(weights=lambdas3)

    # Solve the multi-objective optimization problem.
    results = portfolio_optimizer.run(sampler, sampler_kwargs)
    results.slice_results()

    # Make a plot of the results.
    fig = plot_front(results, "green", "orange", "red")
    fig.savefig(f"figures/pareto_front.png")


Data input
----------

TODO: Explain dataset constrains + refer to docs.


Different Solvers
-----------------

TODO: Explain how different QUBO solvers from D-Wave can be used + refer to DWave sampler docs.