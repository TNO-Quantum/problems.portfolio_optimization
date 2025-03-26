"""This module contains tests for the ``PortfolioOptimizer`` class."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import ArrayLike

from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer


@pytest.mark.parametrize(
    ("weight", "expected_outcome"), [(None, [1.0]), ([1, 2, 3], [1, 2, 3])]
)
def test_parse_weight(weight: ArrayLike | None, expected_outcome: ArrayLike) -> None:
    parsed_weights = PortfolioOptimizer._parse_weight(weight)
    np.testing.assert_array_equal(parsed_weights, expected_outcome)


def test_portfolio_optimizer() -> None:
    portfolio_optimizer = PortfolioOptimizer("benchmark_dataset", k=2)
    portfolio_optimizer.add_minimize_hhi(weights=[1])
    portfolio_optimizer.add_maximize_roc(formulation=1, weights_roc=[1])
    portfolio_optimizer.add_emission_constraint(
        weights=[1],
        emission_now="emis_intens_now",
        emission_future="emis_intens_future",
        reduction_percentage_target=0.7,
    )
    portfolio_optimizer.run()


def test_multiple_growth_factor_constraints() -> None:
    portfolio_optimizer = PortfolioOptimizer("benchmark_dataset", k=2)

    portfolio_optimizer.add_growth_factor_constraint(
        growth_target=1.2,
        weights=[1],
    )
    portfolio_optimizer.run()

    expected_message = "Growth factor constraint has been set before."
    with pytest.raises(ValueError, match=expected_message):
        portfolio_optimizer.add_growth_factor_constraint(
            growth_target=1.4,
            weights=[1],
        )
