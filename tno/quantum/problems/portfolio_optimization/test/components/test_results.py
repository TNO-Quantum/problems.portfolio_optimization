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
    outstanding_future = np.array([[19, 39], [10, 30]])

    for i in range(2, 10, 2):
        results.add_result(outstanding_future)
        assert len(results) == i


def test_aggregate(results: Results) -> None:
    outstanding_future = np.array([[19, 39], [10, 30]])
    for _ in range(100):
        results.add_result(outstanding_future)

    assert len(results._x) == 200
    assert len(results._y) == 200
    assert len(results._outstanding_future) == 200
    results.aggregate()
    assert len(results._x) == 2
    assert len(results._y) == 2
    assert len(results._outstanding_future) == 2
