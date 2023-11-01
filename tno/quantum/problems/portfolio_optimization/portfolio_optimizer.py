from __future__ import annotations

import itertools
import math
from typing import Any, Optional

import numpy as np
from dimod import Sampler
from dwave.samplers import SimulatedAnnealingSampler
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame
from tqdm import tqdm

from tno.quantum.problems.portfolio_optimization.containers import Results
from tno.quantum.problems.portfolio_optimization.new_qubo_factories import QuboCompiler
from tno.quantum.problems.portfolio_optimization.postprocess import Decoder


class PortfolioOptimizer:
    def __init__(
        self,
        portfolio_data: DataFrame,
        kmin: int,
        kmax: int,
        growth_target: float = 0,
    ) -> None:
        self._qubo_compiler = QuboCompiler(portfolio_data, kmin, kmax)
        self.decoder = Decoder(portfolio_data, kmin, kmax)
        self._all_labdas: list[NDArray[np.float_]] = []
        if growth_target <= 0:
            self.results = Results(portfolio_data)
        else:
            self.results = Results(portfolio_data, growth_target)

    def add_minimize_HHI(self, weights: Optional[ArrayLike] = None) -> None:
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
        """
        self._all_labdas.append(self._parse_weight(weights_roc))
        if formulation in [2, 3]:
            self._all_labdas.append(self._parse_weight(weights_stabilize))
        self._qubo_compiler.add_maximize_ROC(
            formulation, capital_growth_factor, ancilla_qubits
        )

    def add_emission_constraint(self, weights: Optional[ArrayLike] = None) -> None:
        self._all_labdas.append(self._parse_weight(weights))
        self._qubo_compiler.add_emission_constraint()

    def add_growth_factor_constraint(
        self, growth_target: float, weights: Optional[ArrayLike] = None
    ) -> None:
        """Add constaint: total_out2030/total_out2021 = growth_target"""
        self._all_labdas.append(self._parse_weight(weights))
        self._qubo_compiler.add_growth_factor_constraint(growth_target)

    def run(
        self,
        sampler: Optional[Sampler] = None,
        sampler_kwargs: Optional[dict[str, Any]] = None,
    ) -> Results:
        sampler = SimulatedAnnealingSampler() if sampler is None else sampler
        sampler_kwargs = {} if sampler_kwargs is None else sampler_kwargs

        total_steps = math.prod(map(len, self._all_labdas))
        labdas_iterator = tqdm(itertools.product(*self._all_labdas), total=total_steps)
        self._qubo_compiler.compile()
        for labdas in labdas_iterator:
            # Compile the model and generate QUBO
            qubo, offset = self._qubo_compiler.make_qubo(*labdas)
            # Solve the QUBO
            response = sampler.sample_qubo(qubo, **sampler_kwargs)
            # Postprocess solution. Iterate over all found solutions. (Compute 2030 portfolios)
            out2030 = self.decoder.decode_sampleset(response)
            self.results.add_result(out2030)
        return self.results

    @staticmethod
    def _parse_weight(weights: Optional[ArrayLike]) -> NDArray[np.float_]:
        if weights is None:
            return np.array([1.0])
        return np.asarray(weights, dtype=np.float_)
