"""This module contains tests for the pareto_front module."""
from __future__ import annotations

import numpy as np
import pytest

from tno.quantum.problems.portfolio_optimization.components import pareto_front


@pytest.fixture(name="x_y_points")
def x_y_points_fixture() -> tuple[list[int], list[int]]:
    # Create point with a square in a square inside another square
    x = [0, 0, 5, 5, 1, 1, 4, 4, 2, 2, 3, 3]
    y = [0, 5, 0, 5, 1, 4, 1, 4, 2, 3, 2, 3]
    return x, y


@pytest.mark.parametrize(
    "min_points,expected_points",
    [
        (4, {(0, 0), (0, 5), (5, 0), (5, 5)}),
        (8, {(0, 0), (0, 5), (5, 0), (5, 5), (1, 1), (1, 4), (4, 1), (4, 4)}),
    ],
)
def test_pareto_front(
    x_y_points: tuple[list[float], list[float]],
    min_points: int,
    expected_points: set[tuple[int, int]],
) -> None:
    x, y = x_y_points
    x_par, y_par = pareto_front(x, y, min_points=min_points)
    assert len(x_par) == len(expected_points)
    assert len(y_par) == len(expected_points)
    for point in zip(x_par, y_par):
        assert point in expected_points


def test_small_front() -> None:
    x = [1, 2]
    y = [3, 4]
    x_par, y_par = pareto_front(x, y)
    np.testing.assert_array_equal(x, x_par)
    np.testing.assert_array_equal(y, y_par)


def test_small_front2() -> None:
    x = [0, 0, 5, 5, 1]
    y = [0, 5, 0, 5, 1]
    expected_points = {(0, 0), (0, 5), (5, 0), (5, 5)}
    x_par, y_par = pareto_front(x, y)

    assert len(x_par) == len(expected_points)
    assert len(y_par) == len(expected_points)
    for point in zip(x_par, y_par):
        assert point in expected_points
