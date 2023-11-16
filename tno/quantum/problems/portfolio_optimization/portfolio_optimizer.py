"""This module contains the ``PortfolioOptimizer`` class."""
from __future__ import annotations

import itertools
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from dimod import Sampler
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
        """Init PortfolioOptimizer

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
        self._growth_target = None

    def add_minimize_HHI(self, weights: Optional[ArrayLike] = None) -> None:
        """
        Adds the minimize HHI objective.

        Args:
            weights: The coefficients that are considered as penalty parameter.

        """
        self._all_lambdas.append(self._parse_weight(weights))
        self._qubo_compiler.add_minimize_HHI()

    def add_maximize_ROC(
        self,
        formulation: int,
        capital_growth_factor: float = 0,
        ancilla_qubits: int = 0,
        weights_roc: Optional[ArrayLike] = None,
        weights_stabilize: Optional[ArrayLike] = None,
    ) -> None:
        """
        Adds the maximize ROC objective.

        formulation 1:
            add 1 qubo term, use weights_roc to scale
        formulation 2:
            add 2 qubo terms, requires extra arg capital_growth_factor. Use weights_roc
            and weights_stabilize to scale
        formulation 3:
            add 2 qubo terms, requires extra arg ancilla_qubits. Use weights_roc and
            weights_stabilize to scale
        formulation 4:
            add 1 qubo term, use weights_roc to scale

        Args:
            formulation: the ROC QUBO formulation that is being used.
                Possible options are: [1, 2, 3, 4].
            capital_growth_factor:
            ancilla_qubits:
            weights_roc:
            weights_stabilize:

        Raises:
            ValueError: If invalid formulation is provided.
        """
        allowed_formulation_options = [1, 2, 3, 4]
        if formulation not in allowed_formulation_options:
            raise ValueError(
                f"Invalid formulation input provided, "
                f"choose from {allowed_formulation_options}."
            )

        self._all_lambdas.append(self._parse_weight(weights_roc))
        if formulation in [2, 3]:
            self._all_lambdas.append(self._parse_weight(weights_stabilize))
        self._qubo_compiler.add_maximize_ROC(
            formulation, capital_growth_factor, ancilla_qubits
        )

    def add_emission_constraint(self, weights: Optional[ArrayLike] = None) -> None:
        """Add emission constraint

        Args:
            weights: The coefficients that are considered as penalty parameter.
        """
        self._all_lambdas.append(self._parse_weight(weights))
        self._qubo_compiler.add_emission_constraint()

    def add_growth_factor_constraint(
        self, growth_target: float, weights: Optional[ArrayLike] = None
    ) -> None:
        """Add constraint: total_outstanding_future/total_outstanding_now = growth_target

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
        Optimize a portfolio given the set constraints.


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
            if self._growth_target is not None:
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
