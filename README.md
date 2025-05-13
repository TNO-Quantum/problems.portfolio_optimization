# Portfolio optimization

Real-world investment decisions involve multiple, often conflicting, objectives that needs to be balanced.
Primary goals typically revolve around maximizing returns while minimizing risks.
At the same time, one might want to require additional constraints such as demanding a minimum carbon footprint reduction. 
Finding a portfolio that balances these objectives is a challenging task and can be solved using multi-objective portfolio optimization. 

This repository provides Python code that converts the multi-objective portfolio optimization problem
into a [QUBO](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization) problem. The transformed problem can then be solved using quantum annealing techniques.

The following objectives can be considered

- `return on capital`, indicated by `ROC`,
- `diversification`, indicated by the [Herfindahl-Hirschman Index](https://en.wikipedia.org/wiki/Herfindahl%E2%80%93Hirschman_inde) `HHI`.

Additionally, we allow for a capital growth factor and arbitrary emission reduction constraints to be considered.

The Pareto front, the set of solutions where one objective can't be improved without worsening the other objective,
can be computed for the objectives return on capital and diversification. 

The codebase is based on the following paper:

- [Aguilera et al., - Multi-objective Portfolio Optimisation Using the Quantum Annealer (2024)](https://www.mdpi.com/2227-7390/12/9/1291)

**Funding:** This research was funded by Rabobank and Stichting TKI High Tech Systems
and Materials, under a program by Brightland's Techruption.


## Documentation

Documentation of the `tno.quantum.problems.portfolio_optimization` package can be found [here](https://tno-quantum.github.io/documentation/).

## Install

Easily install the `tno.quantum.problems.portfolio_optimization` package using pip:

```console
$ python -m pip install tno.quantum.problems.portfolio_optimization
```

If you wish to run the tests you can use:
```console
$ python -m pip install tno.quantum.problems.portfolio_optimization[tests]
```

Usage examples can be found in the [documentation](https://tno-quantum.github.io/documentation/).

Data input
----------

The data used for the portfolio optimization can be imported via an excel file, csv file,
json file or as a Pandas DataFrame.
The data needs to contain at least the following columns:

- `asset`: The name of the asset.
- `outstanding_now`: Current outstanding amount per asset.
- `min_outstanding_future`: Lower bound outstanding amount in the future per asset.
- `max_outstanding_future`: Upper bound outstanding amount in the future per asset.
- `income_now`: Current income per asset, corresponds to return multiplied by the current outstanding amount.
- `regcap_now`: Current regulatory capital per asset.


If the input datafile contains all the correct information, but has different column
names, it is possible to rename the columns without altering the input file.

The data that was used for the publication can be found in the `src/tno/quantum/problems/portfolio_optimization/datasets/` folder.


## (End)use limitations
The content of this software may solely be used for applications that comply with international export control laws.
