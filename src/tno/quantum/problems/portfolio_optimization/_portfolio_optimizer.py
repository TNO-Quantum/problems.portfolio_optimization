"""This module contains the ``PortfolioOptimizer`` class."""

from __future__ import annotations

import itertools
import logging
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame
from tqdm import tqdm

from tno.quantum.optimization.qubo.components import SolverConfig
from tno.quantum.problems.portfolio_optimization._components import (
    Decoder,
    PortfolioData,
    QuboCompiler,
    Results,
)


class PortfolioOptimizer:
    """Class to perform portfolio optimization.

    The :py:class:`~PortfolioOptimizer` class is used to convert
    multi-objective portfolio optimization problems into QUBO problems which can then be
    solved using QUBO solving techniques such as simulated or quantum annealing.

    The following objectives can be considered

    - `return on capital`, indicated by ROC,
    - `diversification`, indicated by the `Herfindahl-Hirschman Index`_ HHI.

    The following constraints can be added

    - `capital growth`, demand a minimum increase in outstanding assets.
    - `emission reduction`, demand a minimum reduction for an arbitrary emission type.

    .. _Herfindahl-Hirschman Index: https://nl.wikipedia.org/wiki/Herfindahl-index
    """

    def __init__(
        self,
        portfolio_data: PortfolioData | DataFrame | str | Path,
        k: int = 2,
        columns_rename: dict[str, str] | None = None,
    ) -> None:
        """Init :py:class:`~PortfolioOptimizer`.

        Args:
            portfolio_data: Portfolio data represented by either the portfolio data
                object, pandas dataframe or path to where portfolio data is stored. See
                the docstring of :py:class:`~tno.quantum.problems.portfolio_optimization._components._io.PortfolioData`
                for data input conventions.
            k: The number of bits that are used to represent the outstanding amount for
                each asset. A fixed point representation is used to represent `$2^k$`
                different equidistant values in the range `$[LB_i, UB_i]$` for asset i.
            columns_rename: can be used to rename data columns. See the docstring of
                :py:class:`~tno.quantum.problems.portfolio_optimization._components._io.PortfolioData`
                for an example.

        Raises:
            TypeError: If the provided portfolio data input has the wrong type.
        """  # noqa: E501
        if isinstance(portfolio_data, PortfolioData):
            self.portfolio_data = portfolio_data
        elif isinstance(portfolio_data, DataFrame):
            self.portfolio_data = PortfolioData(portfolio_data, columns_rename)
        elif isinstance(portfolio_data, (Path, str)):
            self.portfolio_data = PortfolioData.from_file(
                portfolio_data, columns_rename
            )
        else:
            error_msg = (
                "`portfolio_data` must be of type `PortfolioData`, `DataFrame`, `Path` "
                f"or `str`, but was of type {type(portfolio_data)}"
            )
            raise TypeError(error_msg)
        self._qubo_compiler = QuboCompiler(self.portfolio_data, k)
        self.decoder = Decoder(self.portfolio_data, k)
        self._all_lambdas: list[NDArray[np.float64]] = []
        self._provided_emission_constraints: list[tuple[str, str, float, str]] = []
        self._provided_growth_target: float | None = None

        # Create logger
        self._logger = logging.getLogger("PortfolioOptimizer")
        self._logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        self._logger.addHandler(handler)

    def add_minimize_hhi(self, weights: ArrayLike | None = None) -> None:
        r"""Adds the minimize HHI objective to the portfolio optimization problem.

        The HHI objective is given by

        $$HHI(x) = \sum_{i=1}^N\left(\frac{x_i}{\sum_{j=1}^N x_j}\right)^2,$$

        where

            - `$N$` is the total number of assets,
            - `$x_i$` is the future outstanding amount for asset `$i$`.

        As the objective contains non-quadratic terms, a QUBO formulation requires
        approximations. For the QUBO formulation, see the docs of
        :py:meth:`~tno.quantum.problems.portfolio_optimization.QuboFactory.calc_minimize_hhi`.

        Usage example:

        >>> from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer
        >>> import numpy as np
        >>> portfolio_optimizer = PortfolioOptimizer(portfolio_data="benchmark_dataset")
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

        For the QUBO formulation, see the docs of
        :py:meth:`~tno.quantum.problems.portfolio_optimization.QuboFactory.calc_maximize_roc1` and
        :py:meth:`~tno.quantum.problems.portfolio_optimization.QuboFactory.calc_maximize_roc2`.

        Usage example:

        >>> from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer
        >>> import numpy as np
        >>> portfolio_optimizer = PortfolioOptimizer(portfolio_data="benchmark_dataset")
        >>> lambdas = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
        >>> portfolio_optimizer.add_maximize_roc(formulation=1, weights_roc=lambdas)

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
        """  # noqa: E501
        allowed_formulation_options = [1, 2]
        if formulation not in allowed_formulation_options:
            error_msg = (
                "Invalid formulation input provided, "
                f"choose from {allowed_formulation_options}."
            )
            raise ValueError(error_msg)

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
        >>> portfolio_optimizer = PortfolioOptimizer(portfolio_data="benchmark_dataset")
        >>> lambdas = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
        >>> portfolio_optimizer.add_emission_constraint(
        ...   emission_now="emis_intens_now", weights=lambdas
        ... )

        For the QUBO formulation, see the docs of
        :py:meth:`~tno.quantum.problems.portfolio_optimization.QuboFactory.calc_emission_constraint`.

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
        r"""Adds outstanding amount growth factor constraint to optimization problem.

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
        >>> portfolio_optimizer = PortfolioOptimizer(portfolio_data="benchmark_dataset")
        >>> lambdas = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
        >>> portfolio_optimizer.add_growth_factor_constraint(growth_target=1.2, weights=lambdas)

        For the QUBO formulation, see the docs of
        :py:meth:`~tno.quantum.problems.portfolio_optimization.QuboFactory.calc_growth_factor_constraint`.

        Args:
            growth_target: target value for growth factor total outstanding amount.
            weights: The coefficients that are considered as penalty parameter.

        Raises:
            ValueError: If constraint has been added before.
        """  # noqa: E501
        if self._provided_growth_target is not None:
            error_msg = "Growth factor constraint has been set before."
            raise ValueError(error_msg)

        self._all_lambdas.append(self._parse_weight(weights))
        self._provided_growth_target = growth_target
        self._qubo_compiler.add_growth_factor_constraint(growth_target)

    def run(
        self,
        solver_config: SolverConfig | Mapping[str, Any] | None = None,
        *,
        verbose: bool = True,
    ) -> Results:
        """Optimizes a portfolio given the set of provided constraints.

        Usage example:

        >>> from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer
        >>> portfolio_optimizer = PortfolioOptimizer(portfolio_data="benchmark_dataset")
        >>> portfolio_optimizer.add_minimize_hhi()
        >>> portfolio_optimizer.run() # doctest: +SKIP

        Args:
            solver_config: Configuration for the qubo solver to use. Must be a
                ``SolverConfig`` or a mapping with ``"name"`` and ``"options"`` keys. If
                ``None`` (default) is provided, the :py:class:`~tno.quantum.optimization.qubo.solvers.SimulatedAnnealingSolver
                will be used, i.e. ``{"name": "simulated_annealing_solver", "options": {}}``.
            verbose: If True, print detailed information during execution

        Returns:
            Results.

        Raises:
            ValueError: if constraints are not set
        """  # noqa: E501
        solver_config = (
            SolverConfig.from_mapping(solver_config)
            if solver_config is not None
            else SolverConfig(name="simulated_annealing_solver", options={})
        )
        solver = solver_config.get_instance()

        if verbose:
            self.portfolio_data.print_portfolio_info()
            self._logger.info("Status: Creating model.")

            if self._provided_growth_target is not None:
                self._logger.info(
                    "  Setup: Growth target: %.1f", self._provided_growth_target - 1
                )

            for _, _, target_value, name in self._provided_emission_constraints:
                self._logger.info(
                    "Setup: Emission constraint: %s, target reduction percentage=%.1f.",
                    name,
                    target_value - 1,
                )

        self._qubo_compiler.compile()

        results = Results(
            portfolio_data=self.portfolio_data,
            provided_emission_constraints=self._provided_emission_constraints,
            provided_growth_target=self._provided_growth_target,
        )

        if verbose:
            self._logger.info("Status: Calculating")
            starttime = time.time()

        total_steps = math.prod(map(len, self._all_lambdas))
        lambdas_iterator = tqdm(
            itertools.product(*self._all_lambdas), total=total_steps
        )

        for lambdas in lambdas_iterator:
            # Compile the model and generate QUBO
            qubo = self._qubo_compiler.make_qubo(*lambdas)

            # Solve the QUBO
            result = solver.solve(qubo)

            # Postprocess solution. Iterate over all found solutions.
            outstanding_future_samples = self.decoder.decode_result(result)
            results.add_result(outstanding_future_samples)

        if verbose:
            self._logger.info("Status: Dropping duplicate samples in results.")
        results.drop_duplicates()

        if verbose:
            self._logger.info(
                "Status: Computation finished in %.1f seconds.",
                time.time() - starttime,
            )
            self._logger.info("Number of unique samples: %d", len(results))
        return results

    @staticmethod
    def _parse_weight(weights: ArrayLike | None = None) -> NDArray[np.float64]:
        """Converts weights into NumPy array and if needed set default weights to [1.0].

        Args:
            weights: penalty coefficients.

        Returns:
            Numpy array of weights
        """
        if weights is None:
            return np.array([1.0])
        return np.asarray(weights, dtype=np.float64)
