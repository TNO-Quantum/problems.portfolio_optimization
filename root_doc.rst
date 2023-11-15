Portfolio optimization
======================

This repository provides python code perform multi-objective portfolio optimization. 


A Pareto front is the set of Pareto efficient solutions. In multi-objective optimisation, there does
not typically exists a feasible solution that optimises all objective functions simultaneously.

In portfolio optimization, the efficient frontier or portfolio frontier, is the set of portfolios that
have the highest return on capital given the risk of the portfolio. Additional constrains on the carbon footprint.


The codebase is based on the following TNO papers:

- Phillipson et al.. - Portfolio Optimisation Using the D-Wave Quantum Annealer. (2021) (doi: `10.1007/978-3-030-77980-1 4`_)
- Phillipson et al., - Multi-objective Portfolio Optimisation Using the D-Wave Quantum Annealer 


.. _10.1007/978-3-030-77980-1 4: https://doi.org/10.1007/978-3-030-77980-1_4



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