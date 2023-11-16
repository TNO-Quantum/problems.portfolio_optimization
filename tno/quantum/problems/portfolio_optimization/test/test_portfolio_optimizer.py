"""This module contains tests for the ``PortfolioOptimizer`` class."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import ArrayLike

from tno.quantum.problems.portfolio_optimization import PortfolioOptimizer

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access


@pytest.mark.parametrize(
    "weight,expected_outcome", [(None, [1.0]), ([1, 2, 3], [1, 2, 3])]
)
def test_parse_weight(weight: ArrayLike | None, expected_outcome: ArrayLike) -> None:
    parsed_weights = PortfolioOptimizer._parse_weight(weight)
    np.testing.assert_array_equal(parsed_weights, expected_outcome)


def test_portfolio_optimizer() -> None:
    portfolio_optimizer = PortfolioOptimizer("rabobank", k=2)
    portfolio_optimizer.add_minimize_HHI(weights=[1])
    portfolio_optimizer.add_maximize_ROC(formulation=1, weights_roc=[1])
    portfolio_optimizer.add_emission_constraint(
        weights=[1],
        variable_now="emis_intens_now",
        variable_future="emis_intens_future",
        reduction_percentage_target=0.7,
    )
    portfolio_optimizer.run()
