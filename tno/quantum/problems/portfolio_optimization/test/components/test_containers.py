"""This module contains tests for the containers module."""
from collections import deque

import numpy as np
import pytest
from pandas import DataFrame

from tno.quantum.problems.portfolio_optimization.components import Results


@pytest.fixture(name="results")
def results_fixture() -> Results:
    columns = [
        "out_now",
        "out_future_min",
        "out_future_max",
        "emis_intens_now",
        "emis_intens_future",
        "income_now",
        "regcap_now",
    ]
    index = ["asset 1", "asset 2"]
    data = [
        [1.0, 10.0, 19.0, 100.0, 76.0, 1.0, 1.0],
        [2.0, 30.0, 39.0, 200.0, 152.0, 1.0, 1.0],
    ]
    portfolio_data = DataFrame(data=data, columns=columns, index=index)
    return Results(portfolio_data)


def test_add_result(results: Results) -> None:
    assert len(results._x) == 0
    assert len(results._y) == 0
    assert len(results._out_future) == 0

    out_future = np.array([[19, 39], [10, 30]])

    for i in range(2, 10, 2):
        results.add_result(out_future)
        assert len(results._x) == i
        assert len(results._y) == i
        assert len(results._out_future) == i


def test_aggregate(results: Results) -> None:
    out_future = np.array([[19, 39], [10, 30]])
    for _ in range(100):
        results.add_result(out_future)

    assert len(results._x) == 200
    assert len(results._y) == 200
    assert len(results._out_future) == 200
    results.aggregate()
    assert len(results._x) == 2
    assert len(results._y) == 2
    assert len(results._out_future) == 2
