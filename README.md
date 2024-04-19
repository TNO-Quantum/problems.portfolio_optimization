# Comming soon!

This repository will be released in the near future.

For more details you can contact the developers of this repository at tnoquantum@tno.nl.

# Portfolio optimization

Real-world investment decisions involve multiple, often conflicting, objectives that needs to be balanced.
Primary goals typically revolve around maximizing returns while minimizing risks.
At the same time, one might want to require additional constraints such as demanding a minimum carbon footprint reduction. 
Finding a portfolio that balances these objectives is a challenging task and can be solved using multi-objective portfolio optimization. 


This repository provides Python code that converts the multi-objective portfolio optimization problem
into a `QUBO`_ problem. The transformed problem can then be solved using quantum annealing techniques.

The following objectives can be considered

- `return on capital`, indicated by `ROC`,
- `diversification`, indicated by the Herfindahl-Hirschman Index `HHI`.

Additionally, we allow for a capital growth factor and arbitrary emission reduction constraints to be considered.

The `Pareto front`, the set of solutions where one objective can't be improved without worsening the other objective,
can be computed for the objectives return on capital and diversification. 

The codebase is based on the following paper:

- Aguilera et al., - Multi-objective Portfolio Optimisation Using the Quantum Annealer (2024)
