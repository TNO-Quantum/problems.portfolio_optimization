from __future__ import annotations

import itertools
from typing import Any, Optional

import numpy as np
from dimod import Sampler
from numpy.typing import NDArray
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
        labdas1: NDArray[np.float_],
        labdas2: NDArray[np.float_],
        labdas3: NDArray[np.float_],
        labdas4: Optional[NDArray[np.float_]] = None,
        growth_target: float = 0,
    ) -> None:
        self._qubo_compiler = QuboCompiler(portfolio_data, kmin, kmax)
        if labdas4 is None:
            total_steps = len(labdas1) * len(labdas2) * len(labdas3)
            self.labdas_iterator = tqdm(
                itertools.product(labdas1, labdas2, labdas3), total=total_steps
            )
        else:
            total_steps = len(labdas1) * len(labdas2) * len(labdas3) * len(labdas4)
            self.labdas_iterator = tqdm(
                itertools.product(labdas1, labdas2, labdas3, labdas4), total=total_steps
            )
        self.decoder = Decoder(portfolio_data, kmin, kmax)
        if growth_target <= 0:
            self.results = Results(portfolio_data)
        else:
            self.results = Results(portfolio_data, growth_target)

    def add_minimize_HHI(self) -> None:
        self._qubo_compiler.add_minimize_HHI()

    def add_maximize_ROC(
        self,
        formulation: int,
        capital_growth_factor: float = 0,
        ancilla_qubits: int = 0,
    ) -> None:
        """
        formulation 1:
            add 1 qubo term
        formulation 2:
            add 2 qubo terms, requires extra arg capital_growth_factor
        formulation 3:
            add 2 qubo terms, requires extra arg ancilla_qubits
        formulation 4:
            add 1 qubo term
        """
        self._qubo_compiler.add_maximize_ROC(
            formulation, capital_growth_factor, ancilla_qubits
        )

    def add_emission_constraint(self) -> None:
        self._qubo_compiler.add_emission_constraint()

    def add_growth_factor_constraint(self, growth_target: float) -> None:
        """Add constaint: total_out2030/total_out2021 = growth_target"""
        self._qubo_compiler.add_growth_factor_constraint(growth_target)

    def run(self, sampler: Sampler, sampler_kwargs: dict[str, Any]) -> Results:
        self._qubo_compiler.compile()
        for labdas in self.labdas_iterator:
            # Compile the model and generate QUBO
            qubo, offset = self._qubo_compiler.make_qubo(*labdas)
            # Solve the QUBO
            response = sampler.sample_qubo(qubo, **sampler_kwargs)
            # Postprocess solution. Iterate over all found solutions. (Compute 2030 portfolios)
            out2030 = self.decoder.decode_sampleset(response)
            self.results.add_result(out2030)
        return self.results
