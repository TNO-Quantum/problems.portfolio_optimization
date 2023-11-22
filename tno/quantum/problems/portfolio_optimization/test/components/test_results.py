"""This module contains tests for the Results module."""
import numpy as np
import pytest

from tno.quantum.problems.portfolio_optimization.components import Results
from tno.quantum.problems.portfolio_optimization.test import make_test_dataset

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access


@pytest.fixture(name="results")
def results_fixture() -> Results:
    portfolio_data = make_test_dataset()
    provided_constraints = []
    return Results(portfolio_data, provided_constraints)


def test_add_result(results: Results) -> None:
    assert len(results) == 0

    outstanding_future_samples = np.array([[19, 39], [10, 30]])
    results.add_result(outstanding_future_samples)
    assert len(results) == 2

    # Add duplicate result
    results.add_result(outstanding_future_samples)
    results.drop_duplicates()
    assert len(results) == 2

    # Add different results
    for i in range(10):
        outstanding_future_samples = np.array([[i, 1]])
        results.add_result(outstanding_future_samples)
        results.drop_duplicates()
        assert len(results) == i + 3
