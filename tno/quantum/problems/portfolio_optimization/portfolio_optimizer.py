"""This module contains the ``PortfolioOptimizer`` class.

Example script:



"""
from __future__ import annotations

import itertools
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from dimod.core.sampler import Sampler
from dwave.samplers import SimulatedAnnealingSampler
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from tno.quantum.problems.portfolio_optimization.components import (
    Decoder,
    QuboCompiler,
    Results,
    print_portfolio_info,
    read_portfolio_data,
)


class PortfolioOptimizer:
    """Portfolio Optimizer"""

    def __init__(
        self,
        filename: str | Path,
        k: int = 2,
        columns_rename: Optional[dict[str, str]] = None,
    ) -> None:
        """Init ``PortfolioOptimizer``.

        Args:
            filename: path to where portfolio data is stored. See the docstring of
                :py:func:`~portfolio_optimization.components.io.read_portfolio_data`
                for data input conventions.
            k: The number of bits that are used to represent the outstanding amount for
                each asset. A fixed point representation is used to represent `$2^k$`
                different equidistant values in the range `$[LB_i, UB_i]$` for asset i.
            column_rename: can be used to rename data columns. See the docstring of
                :py:func:`~portfolio_optimization.components.io.read_portfolio_data` for
                example.
        """
        portfolio_data = read_portfolio_data(filename, columns_rename)
        self.portfolio_data = portfolio_data
        self._qubo_compiler = QuboCompiler(portfolio_data, k)
        self.decoder = Decoder(portfolio_data, k)
        self._all_lambdas: list[NDArray[np.float_]] = []
        self._growth_target: float

    def add_minimize_hhi(self, weights: Optional[ArrayLike] = None) -> None:
        r"""Adds the minimize HHI objective to the portfolio optimization problem.

        The HHI objective is given by

        $$HHI(x) = \sum_{i=1}^N\left(\frac{x_i}{\sum_{j=1}^N x_j}\right)^2,$$

        where

            - `$N$` is the total number of assets,
            - `$x_i$` is the future outstanding amount for asset `$i$`.

        usage example:

        >>> from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer
        >>> import numpy as np
        >>> portfolio_optimizer = PortfolioOptimizer(filename="rabobank")
        >>> lambdas = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
        >>> portfolio_optimizer.add_minimize_hhi(weights=lambdas)

        For the QUBO formulation, see the docs of
        :py:class:`~portfolio_optimization.components.qubos.qubo_factory.QuboFactory`.
        :py:meth:`~portfolio_optimization.components.qubos.qubo_factory.QuboFactory.calc_minimize_hhi`.

        Args:
            weights: The coefficients that are considered as penalty parameter.

        """
        self._all_lambdas.append(self._parse_weight(weights))
        self._qubo_compiler.add_minimize_hhi()

    def add_maximize_roc(
        self,
        formulation: int,
        ancilla_variables: int = 0,
        weights_roc: Optional[ArrayLike] = None,
        weights_stabilize: Optional[ArrayLike] = None,
    ) -> None:
        r"""Adds the maximize ROC objective to the portfolio optimization problem.

        The ROC objective is given by

        .. math::

            ROC(x) =
            \frac{\sum_{i=1}^N \frac{1}{y_i} x_i \cdot r_i}
            {\sum_{i=1}^N \frac{1}{y_i} x_i \cdot c_i},

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

            where

                - `$G_C$` is ...,
                - `$C_{21}$` is ...,
        
            Adds 2 qubo terms, requires extra arg ``ancilla_variables``. Use ``weights_roc``
            and ``weights_stabilize`` to scale.
        
        For the different QUBO formulations, see the docs of
        :py:class:`~portfolio_optimization.components.qubos.qubo_factory.QuboFactory`.

        usage example:

        >>> from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer
        >>> import numpy as np
        >>> portfolio_optimizer = PortfolioOptimizer(filename="rabobank")
        >>> lambdas = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
        >>> portfolio_optimizer.add_maximize_roc(...)

        Args:
            formulation: the ROC QUBO formulation that is being used.
                Possible options are: [1, 2].
            ancilla_variables:
            weights_roc:
            weights_stabilize:

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
        variable_now: str,
        variable_future: Optional[str] = None,
        reduction_percentage_target: float = 0.7,
        weights: Optional[ArrayLike] = None,
    ) -> None:
        r"""Add emission constraint to the portfolio optimization problem.

        The constraint is given by

        .. math::

            \frac{\sum_{i=1}^Nf_i \cdot x_i}{\sum_i x_i}
            =
            g \frac{\sum_{i=1}^Ne_i \cdot y_i}{\sum_{i=1}^N y_i},

        where:

            - `$N$` is the total number of assets,
            - `$x_i$` is the future outstanding amount for asset `$i$`,
            - `$y_i$` is the current outstanding amount for asset `$i$`,
            - `$e_i$` is the current emission intensity for asset `$i$`,
            - `$f_i$` is the expected emission intensity at the future for asset `$i$`,
            - `$g$` is the target value for the relative emission reduction.

        usage example:

        >>> from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer
        >>> import numpy as np
        >>> portfolio_optimizer = PortfolioOptimizer(filename="rabobank")
        >>> lambdas = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
        >>> portfolio_optimizer.add_emission_constraint(
        ...   variable_now="emis_intens_now", weights=lambdas
        ... )

        For the QUBO formulation, see the docs of
        :py:class:`~portfolio_optimization.components.qubos.qubo_factory.QuboFactory`.
        :py:meth:`~portfolio_optimization.components.qubos.qubo_factory.QuboFactory.calc_emission_constraint`.

        Args:
            variable_now: Name of the column in the portfolio dataset corresponding to
                the variables at current time.
            variable_future: Name of the column in the portfolio dataset corresponding
                to the variables at future time. If no value is provided, it is assumed
                that the value is constant over time, i.e., the variable
                ``variable_now`` will be used.
            reduction_percentage_target: target value for reduction percentage amount.
            weights: The coefficients that are considered as penalty parameter.
        """
        self._all_lambdas.append(self._parse_weight(weights))
        self._qubo_compiler.add_emission_constraint(
            variable_now=variable_now,
            variable_future=variable_future,
            reduction_percentage_target=reduction_percentage_target,
        )

    def add_growth_factor_constraint(
        self, growth_target: float, weights: Optional[ArrayLike] = None
    ) -> None:
        r"""Add an growth factor constraint to the portfolio optimization problem.

        The constraint is given by

        $$\frac{\sum_i x_i}{\sum_i y_i} = g,$$

        where

            - `$x_i$` is the future outstanding amount for asset `$i$`,
            - `$y_i$` is the current outstanding amount for asset `$i$`,
            - `$g$` is the target value for the total growth factor.

        usage example:

        >>> from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer
        >>> import numpy as np
        >>> portfolio_optimizer = PortfolioOptimizer(filename="rabobank")
        >>> lambdas = np.logspace(-16, 1, 25, endpoint=False, base=10.0)
        >>> portfolio_optimizer.add_emission_constraint(growth_target=1.2, weights=lambdas)

        For the QUBO formulation, see the docs of
        :py:class:`~portfolio_optimization.components.qubos.qubo_factory.QuboFactory`.
        :py:meth:`~portfolio_optimization.components.qubos.qubo_factory.QuboFactory.calc_growth_factor_constraint`.

        Args:
            growth_target:
            weights: The coefficients that are considered as penalty parameter.
        """
        self._growth_target = growth_target
        self._all_lambdas.append(self._parse_weight(weights))
        self._qubo_compiler.add_growth_factor_constraint(growth_target)

    def run(
        self,
        sampler: Optional[Sampler] = None,
        sampler_kwargs: Optional[dict[str, Any]] = None,
        verbose: bool = True,
    ) -> Results:
        """
        Optimize a portfolio given the set of provided constraints.

        usage example:

        >>> from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer
        >>> from dwave.samplers import SimulatedAnnealingSampler
        >>> portfolio_optimizer = PortfolioOptimizer(filename="rabobank")
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
        if verbose:
            print_portfolio_info(self.portfolio_data)

        sampler = SimulatedAnnealingSampler() if sampler is None else sampler
        sampler_kwargs = {} if sampler_kwargs is None else sampler_kwargs

        if verbose:
            print("Status: creating model")
            if hasattr(self, "_growth_target"):
                print(f"Growth target: {self._growth_target - 1:.1%}")
        self._qubo_compiler.compile()

        results = Results(self.portfolio_data)

        if verbose:
            print("Status: calculating")
            starttime = datetime.now()

        total_steps = math.prod(map(len, self._all_lambdas))
        lambdas_iterator = tqdm(
            itertools.product(*self._all_lambdas), total=total_steps
        )

        for lambdas in lambdas_iterator:
            # Compile the model and generate QUBO
            qubo, offset = self._qubo_compiler.make_qubo(*lambdas)
            # Solve the QUBO
            response = sampler.sample_qubo(qubo, **sampler_kwargs)
            # Postprocess solution. Iterate over all found solutions. (Compute future portfolios)
            out_future = self.decoder.decode_sampleset(response)
            results.add_result(out_future)

        if verbose:
            print("Number of generated samples: ", len(results))
            print("Time consumed:", datetime.now() - starttime)

        results.aggregate()
        return results

    @staticmethod
    def _parse_weight(weights: Optional[ArrayLike] = None) -> NDArray[np.float_]:
        """Convert weights into NumPy array and if needed set default weights to [1.0]

        Args:
            weights: penalty coefficients.

        Returns:
            Numpy array of weights
        """
        if weights is None:
            return np.array([1.0])
        return np.asarray(weights, dtype=np.float_)
