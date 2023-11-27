"""This module contains the ``PortfolioOptimizer`` class."""
from __future__ import annotations

import itertools
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from dimod.core.sampler import Sampler
from dwave.samplers import SimulatedAnnealingSampler
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame
from tqdm import tqdm

from tno.quantum.problems.portfolio_optimization.components import (
    Decoder,
    PortfolioData,
    QuboCompiler,
    Results,
)


class PortfolioOptimizer:
    """The ``PortfolioOptimizer`` class is used to convert multi-objective portfolio
    optimization problems into QUBO problems which can then be solved using QUBO solving
    techniques such as simulated or quantum annealing.

    The following objectives can be considered

    - `return on capital`, indicated by ROC,
    - `diversification`, indicated by the `Herfindahl-Hirschman Index`_ HHI.

    The following constraints can be added

    - `capital growth`, demand a minimum increase in outstanding assets.
    - `emission reduction`, demand a minimum reduction for an arbitrary emission type.

    Usage example:

    .. code-block::

        import numpy as np
        from dwave.samplers import SimulatedAnnealingSampler

        from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer

        # Choose sampler for solving qubo
        sampler = SimulatedAnnealingSampler()
        sampler_kwargs = {"num_reads": 20, "num_sweeps": 200}

        # Set up penalty coefficients for the constraints
        lambdas1 = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
        lambdas2 = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
        lambdas3 = np.array([1])

        # Create portfolio optimization problem
        portfolio_optimizer = PortfolioOptimizer("benchmark_dataset")
        portfolio_optimizer.add_minimize_hhi(weights=lambdas1)
        portfolio_optimizer.add_maximize_roc(formulation=1, weights_roc=lambdas1)
        portfolio_optimizer.add_emission_constraint(
            weights=lambdas3,
            emission_now="emis_intens_now",
            emission_future="emis_intens_future",
            name="emission",
        )

        # Solve the portfolio optimization problem
        results = portfolio_optimizer.run(sampler, sampler_kwargs)
        print(results.head())

    .. _Herfindahl-Hirschman Index: https://nl.wikipedia.org/wiki/Herfindahl-index
    """

    def __init__(
        self,
        portfolio_data: PortfolioData | DataFrame | str | Path,
        k: int = 2,
        columns_rename: dict[str, str] | None = None,
    ) -> None:
        """Init ``PortfolioOptimizer``.

        Args:
            portfolio_data: Portfolio data represented by a ``PortfolioData`` object,
                a pandas ``DataFrame`` or a path to where portfolio data is stored. See
                the docstring of
                :py:class:`~portfolio_optimization.components.io.PortfolioData` for data
                input conventions.
            k: The number of bits that are used to represent the outstanding amount for
                each asset. A fixed point representation is used to represent `$2^k$`
                different equidistant values in the range `$[LB_i, UB_i]$` for asset i.
            column_rename: can be used to rename data columns. See the docstring of
                :py:class:`~portfolio_optimization.components.io.PortfolioData` for
                example.

        Raises:
            TypeError: If the provided ``portfolio_data`` input has the wrong type.
        """
        if isinstance(portfolio_data, PortfolioData):
            self.portfolio_data = portfolio_data
        elif isinstance(portfolio_data, DataFrame):
            self.portfolio_data = PortfolioData(portfolio_data, columns_rename)
        elif isinstance(portfolio_data, (Path, str)):
            self.portfolio_data = PortfolioData.from_file(
                portfolio_data, columns_rename
            )
        else:
            raise TypeError(
                "`portfolio_data` must be of type `PortfolioData`, `DataFrame`, `Path` "
                f"or `str`, but was of type {type(portfolio_data)}"
            )
        self._qubo_compiler = QuboCompiler(self.portfolio_data, k)
        self.decoder = Decoder(self.portfolio_data, k)
        self._all_lambdas: list[NDArray[np.float_]] = []
        self._provided_emission_constraints: list[tuple[str, str, float, str]] = []
        self._provided_growth_target: float | None = None

    def add_minimize_hhi(self, weights: ArrayLike | None = None) -> None:
        r"""Adds the minimize HHI objective to the portfolio optimization problem.

        The HHI objective is given by

        $$HHI(x) = \sum_{i=1}^N\left(\frac{x_i}{\sum_{j=1}^N x_j}\right)^2,$$

        where

            - `$N$` is the total number of assets,
            - `$x_i$` is the future outstanding amount for asset `$i$`.

        As the objective contains non-quadratic terms, a QUBO formulation requires
        approximations. For the QUBO formulation, see the docs of
        :py:class:`~portfolio_optimization.components.qubos.QuboFactory`.
        :py:meth:`~portfolio_optimization.components.qubos.QuboFactory.calc_minimize_hhi`.
        
        Usage example:

        >>> from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer
        >>> import numpy as np
        >>> portfolio_optimizer = PortfolioOptimizer(filename="benchmark_dataset")
        >>> lambdas = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
        >>> portfolio_optimizer.add_minimize_hhi(weights=lambdas)

        Args:
            weights: The coefficients that are considered as penalty parameter.
        """
        self._all_lambdas.append(self._parse_weight(weights))
        self._qubo_compiler.add_minimize_hhi()

    def add_maximize_roc(
        self,
        formulation: int,
        weights_roc: ArrayLike | None = None,
        ancilla_variables: int = 0,
        weights_stabilize: ArrayLike | None = None,
    ) -> None:
        r"""Adds the maximize ROC objective to the portfolio optimization problem.

        The ROC objective is given by

        .. math::

            ROC(x) =
            \frac{\sum_{i=1}^N \frac{x_i \cdot r_i}{y_i}}
            {\sum_{i=1}^N \frac{x_i \cdot c_i}{y_i}},

        where

            - `$N$` is the total number of assets,
            - `$x_i$` is the future outstanding amount for asset `$i$`,
            - `$y_i$` is the current outstanding amount for asset `$i$`,
            - `$r_i$` is the return for asset `$i$`,
            - `$c_i$` is the regulatory capital for asset `$i$`.

        As the ROC is not a quadratic function, it is approximated using two different
        formulations:

        formulation 1:
            $$ROC_1(x)=\sum_{i=1}^N\frac{x_i\cdot r_i}{c_i\cdot y_i}$$

            Adds 1 qubo term, use ``weights_roc`` to scale.

        formulation 2:
            $$ROC_2(x)=\frac{1}{G_C \cdot C_{21}}\sum_{i=1}^N x_i\frac{r_i}{y_i}$$

            In this formulation, $G_C \cdot C_{21}$ approximates a fixed regulatory
            capital growth which is equal for all assets, where

                - `$1≤G_C<2$` is a growth factor to be estimated using ancilla variables,
                - `$C_{21} = \sum_{i=1}^N c_{i}$` is the sum of all assets' regulatory capital.

            This formulation adds 2 qubo terms, one for the ROC term, and one to stabilize the 
            capital growth. The stabilize qubo requires an extra argument ``ancilla_variables``.
            Use ``weights_roc`` and ``weights_stabilize`` to scale both qubo's accordingly.

        For the different QUBO formulations, see the docs of
        :py:class:`~portfolio_optimization.components.qubos.QuboFactory`.

        Usage example:

        >>> from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer
        >>> import numpy as np
        >>> portfolio_optimizer = PortfolioOptimizer(filename="benchmark_dataset")
        >>> lambdas = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
        >>> portfolio_optimizer.add_maximize_roc(...)

        Args:
            formulation: the ROC QUBO formulation that is being used.
                Possible options are: [1, 2].
            weights_roc: The coefficients that are considered as penalty parameter for
                maximizing the roc objective.
            ancilla_variables: The number of ancillary variables that are used to
                represent ``G_C`` using fixed point representation. Only relevant for
                roc formulation ``2``.
            weights_stabilize: The coefficients that are considered as penalty parameter
                for the stabilizing constraint. Only relevant for roc formulation ``2``.

        Raises:
            ValueError: If invalid formulation is provided.
        """
        allowed_formulation_options = [1, 2]
        if formulation not in allowed_formulation_options:
            raise ValueError(
                "Invalid formulation input provided, "
                f"choose from {allowed_formulation_options}."
            )

        self._all_lambdas.append(self._parse_weight(weights_roc))
        if formulation == 2:
            self._all_lambdas.append(self._parse_weight(weights_stabilize))
        self._qubo_compiler.add_maximize_roc(formulation, ancilla_variables)

    def add_emission_constraint(
        self,
        emission_now: str,
        emission_future: str | None = None,
        reduction_percentage_target: float = 0.7,
        name: str | None = None,
        weights: ArrayLike | None = None,
    ) -> None:
        r"""Adds emission constraint to the portfolio optimization problem.

        The constraint is given by

        .. math::

            \frac{\sum_{i=1}^Nf_i \cdot x_i}{\sum_{i=1}^N x_i}
            =
            g_e \frac{\sum_{i=1}^Ne_i \cdot y_i}{\sum_{i=1}^N y_i},

        where:

            - `$N$` is the total number of assets,
            - `$x_i$` is the future outstanding amount for asset `$i$`,
            - `$y_i$` is the current outstanding amount for asset `$i$`,
            - `$e_i$` is the current emission intensity for asset `$i$`,
            - `$f_i$` is the expected emission intensity at the future for asset `$i$`,
            - `$g_e$` is the target value for the relative emission reduction.

        Usage example:

        >>> from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer
        >>> import numpy as np
        >>> portfolio_optimizer = PortfolioOptimizer(filename="benchmark_dataset")
        >>> lambdas = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
        >>> portfolio_optimizer.add_emission_constraint(
        ...   emission_now="emis_intens_now", weights=lambdas
        ... )

        For the QUBO formulation, see the docs of
        :py:class:`~portfolio_optimization.components.qubos.QuboFactory`.
        :py:meth:`~portfolio_optimization.components.qubos.QuboFactory.calc_emission_constraint`.

        Args:
            emission_now: Name of the column in the portfolio dataset corresponding to
                the variables emission intensity at current time.
            emission_future: Name of the column in the portfolio dataset corresponding
                to the variables emission intensity at future time. If no value is
                provided, it is assumed that the emission intensity is constant over
                time, i.e., the variable ``emission_now`` will be used.
            reduction_percentage_target: target value for reduction percentage amount.
            name: Name that will be used for emission constraint in the results df.
            weights: The coefficients that are considered as penalty parameter.
        """
        # Store emission constraint information
        if name is None:
            name = emission_now
        if emission_future is None:
            emission_future = emission_now
        self._provided_emission_constraints.append(
            (emission_now, emission_future, reduction_percentage_target, name)
        )
        self._all_lambdas.append(self._parse_weight(weights))

        self._qubo_compiler.add_emission_constraint(
            emission_now=emission_now,
            emission_future=emission_future,
            reduction_percentage_target=reduction_percentage_target,
        )

    def add_growth_factor_constraint(
        self, growth_target: float, weights: ArrayLike | None = None
    ) -> None:
        # pylint: disable=line-too-long
        r"""Adds an outstanding amount growth factor constraint to the portfolio
        optimization problem.

        The constraint is given by

        $$\frac{\sum_{i=1}^N x_i}{\sum_{i=1}^N y_i} = g_c,$$

        where

            - `$N$` is the total number of assets,
            - `$x_i$` is the future outstanding amount for asset `$i$`,
            - `$y_i$` is the current outstanding amount for asset `$i$`,
            - `$g_c$` is the target value for the total growth factor.

        This constraint can only be added once.

        Usage example:

        >>> from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer
        >>> import numpy as np
        >>> portfolio_optimizer = PortfolioOptimizer(filename="benchmark_dataset")
        >>> lambdas = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
        >>> portfolio_optimizer.add_emission_constraint(growth_target=1.2, weights=lambdas)

        For the QUBO formulation, see the docs of
        :py:class:`~portfolio_optimization.components.qubos.QuboFactory`.
        :py:meth:`~portfolio_optimization.components.qubos.QuboFactory.calc_growth_factor_constraint`.

        Args:
            growth_target: target value for growth factor total outstanding amount.
            weights: The coefficients that are considered as penalty parameter.

        Raises:
            ValueError: If constraint has been added before.
        """
        # pylint: enable=line-too-long
        if self._provided_growth_target is not None:
            raise ValueError("Growth factor constraint has been set before.")

        self._all_lambdas.append(self._parse_weight(weights))
        self._provided_growth_target = growth_target
        self._qubo_compiler.add_growth_factor_constraint(growth_target)

    def run(
        self,
        sampler: Sampler | None = None,
        sampler_kwargs: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> Results:
        # pylint: disable=line-too-long
        """
        Optimizes a portfolio given the set of provided constraints.

        Usage example:

        >>> from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer
        >>> from dwave.samplers import SimulatedAnnealingSampler
        >>> portfolio_optimizer = PortfolioOptimizer(filename="benchmark_dataset")
        >>> portfolio_optimizer.add_minimize_HHI()
        >>> portfolio_optimizer.run(sampler=SimulatedAnnealingSampler(), verbose=False)

        Args:
            sampler: Instance of a D-Wave Sampler that can be used to solve the QUBO.
                More information can be found in the `D-Wave Ocean Documentation`_.
                By default the ``SimulatedAnnealingSampler`` is being used.
            sampler_kwargs: The sampler specific key-word arguments.
            verbose: If True, print detailed information during execution

        Returns:
            results

        Raises:
            ValueError: if constraints are not set

        .. _D-Wave Ocean Documentation: https://docs.ocean.dwavesys.com/projects/system/en/stable/reference/samplers.html

        """
        # pylint: enable=line-too-long
        if verbose:
            self.portfolio_data.print_portfolio_info()

        sampler = SimulatedAnnealingSampler() if sampler is None else sampler
        sampler_kwargs = {} if sampler_kwargs is None else sampler_kwargs

        if verbose:
            print("Status: creating model")
            if self._provided_growth_target is not None:
                print(f"Growth target: {self._provided_growth_target - 1:.1%}")

            for _, _, target_value, name in self._provided_emission_constraints:
                print(
                    f"Emission constraint: {name}, "
                    f"target reduction percentage: {target_value - 1:.1%}"
                )

        self._qubo_compiler.compile()

        results = Results(
            portfolio_data=self.portfolio_data,
            provided_emission_constraints=self._provided_emission_constraints,
            provided_growth_target=self._provided_growth_target,
        )

        if verbose:
            print("Status: calculating")
            starttime = datetime.now()

        total_steps = math.prod(map(len, self._all_lambdas))
        lambdas_iterator = tqdm(
            itertools.product(*self._all_lambdas), total=total_steps
        )

        for lambdas in lambdas_iterator:
            # Compile the model and generate QUBO
            qubo, _ = self._qubo_compiler.make_qubo(*lambdas)
            # Solve the QUBO
            response = sampler.sample_qubo(qubo, **sampler_kwargs)
            # Postprocess solution. Iterate over all found solutions. (Compute future portfolios)
            outstanding_future_samples = self.decoder.decode_sampleset(response)
            results.add_result(outstanding_future_samples)

        if verbose:
            print("Drop duplicate samples in results.")
        results.drop_duplicates()

        if verbose:
            print("Number of unique samples: ", len(results))
            print("Time consumed:", datetime.now() - starttime)

        return results

    @staticmethod
    def _parse_weight(weights: ArrayLike | None = None) -> NDArray[np.float_]:
        """Converts weights into NumPy array and if needed set default weights to [1.0]

        Args:
            weights: penalty coefficients.

        Returns:
            Numpy array of weights
        """
        if weights is None:
            return np.array([1.0])
        return np.asarray(weights, dtype=np.float_)
