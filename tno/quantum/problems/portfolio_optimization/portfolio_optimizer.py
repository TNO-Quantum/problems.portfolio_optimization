from __future__ import annotations

import itertools
from typing import Any, Optional

import numpy as np
from dimod import Sampler
from numpy.typing import NDArray
from pandas import DataFrame
from tqdm import tqdm

from tno.quantum.problems.portfolio_optimization.containers import Results
from tno.quantum.problems.portfolio_optimization.postprocess import Decoder
from tno.quantum.problems.portfolio_optimization.qubo_factories import BaseQUBOFactory


class PortfolioOptimizer:
    def __init__(
        self,
        portfolio_data: DataFrame,
        kmin: int,
        kmax: int,
        qubo_factory: BaseQUBOFactory,
        sampler: Sampler,
        sampler_kwargs: dict[str, Any],
        labdas1: NDArray[np.float_],
        labdas2: NDArray[np.float_],
        labdas3: NDArray[np.float_],
        labdas4: Optional[NDArray[np.float_]] = None,
        growth_target: float = 0,
    ) -> None:
        self.qubo_factory = qubo_factory
        self.sampler = sampler
        self.sampler_kwargs = sampler_kwargs
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

    def run(self):
        for labdas in self.labdas_iterator:
            # Compile the model and generate QUBO
            qubo, offset = self.qubo_factory.make_qubo(*labdas)
            # Solve the QUBO
            response = self.sampler.sample_qubo(qubo, **self.sampler_kwargs)
            # Postprocess solution. Iterate over all found solutions. (Compute 2030 portfolios)
            out2030 = self.decoder.decode_sampleset(response)
            self.results.add_result(out2030)
        return self.results
