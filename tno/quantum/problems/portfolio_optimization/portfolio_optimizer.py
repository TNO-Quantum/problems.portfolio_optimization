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
        kmin: int,
        kmax: int,
        columns_rename: Optional[dict[str, str]] = None,
    ) -> None:
        """Init PortfolioOptimizer

        Args:
            filename: path to portfolio data
            kmin:
            kmax:
            column_rename:
        """
        portfolio_data = read_portfolio_data(filename, columns_rename)
        self.portfolio_data = portfolio_data
        self._qubo_compiler = QuboCompiler(portfolio_data, kmin, kmax)
        self.decoder = Decoder(portfolio_data, kmin, kmax)
        self._all_labdas: list[NDArray[np.float_]] = []
        self._growth_target = None

    def add_minimize_HHI(self, weights: Optional[ArrayLike] = None) -> None:
        """
        Adds the minimize HHI objective.

        Args:
            weights:

        """
        self._all_labdas.append(self._parse_weight(weights))
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
            formulation:
            capital_growth_factor:
            ancilla_qubits:
            weights_roc:
            weights_stabilize:
        """
        self._all_labdas.append(self._parse_weight(weights_roc))
        if formulation in [2, 3]:
            self._all_labdas.append(self._parse_weight(weights_stabilize))
        self._qubo_compiler.add_maximize_ROC(
            formulation, capital_growth_factor, ancilla_qubits
        )

    def add_emission_constraint(self, weights: Optional[ArrayLike] = None) -> None:
        """Add emission constraint

        Args:
            weights: penalty parameter coefficients
        """
        self._all_labdas.append(self._parse_weight(weights))
        self._qubo_compiler.add_emission_constraint()

    def add_growth_factor_constraint(
        self, growth_target: float, weights: Optional[ArrayLike] = None
    ) -> None:
        """Add constraint: total_out_future/total_out_now = growth_target

        Args:
            growth_target:
            weights:
        """
        self._growth_target = growth_target
        self._all_labdas.append(self._parse_weight(weights))
        self._qubo_compiler.add_growth_factor_constraint(growth_target)

    def run(
        self,
        sampler: Optional[Sampler] = None,
        sampler_kwargs: Optional[dict[str, Any]] = None,
        verbose: bool = True,
    ) -> Results:
        """
        Optimize portfolio given constraints

        Args:
            sampler:
            sampler_kwargs:
            verbose:

        Returns:
            results

        Raises:
            ValueError: if constraints are not set
        """
        if verbose:
            print_portfolio_info(self.portfolio_data)

        sampler = SimulatedAnnealingSampler() if sampler is None else sampler
        sampler_kwargs = {} if sampler_kwargs is None else sampler_kwargs

        if verbose:
            print("Status: creating model")
            if self._growth_target is not None:
                print(f"Growth target: {self._growth_target - 1:.1%})")
        self._qubo_compiler.compile()

        results = Results(self.portfolio_data)

        if verbose:
            print("Status: calculating")
            starttime = datetime.now()

        total_steps = math.prod(map(len, self._all_labdas))
        labdas_iterator = tqdm(itertools.product(*self._all_labdas), total=total_steps)

        for labdas in labdas_iterator:
            # Compile the model and generate QUBO
            qubo, offset = self._qubo_compiler.make_qubo(*labdas)
            # Solve the QUBO
            response = sampler.sample_qubo(qubo, **sampler_kwargs)
            # Postprocess solution. Iterate over all found solutions. (Compute future portfolios)
            out_future = self.decoder.decode_sampleset(response)
            results.add_result(out_future)

        if verbose:
            print(
                "Number of generated samples: ",
                len(results.x1),
                len(results.x2),
                len(results.x3),
            )
            print("Time consumed:", datetime.now() - starttime)

        results.aggregate()
        return results

    @staticmethod
    def _parse_weight(weights: Optional[ArrayLike] = None) -> NDArray[np.float_]:
        """Convert weights into NumPy array and if needed set default weights to [1.0]

        Args:
            weights: penalty coefficients

        Returns:
            Numpy array of weights
        """
        if weights is None:
            return np.array([1.0])
        return np.asarray(weights, dtype=np.float_)
