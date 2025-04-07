"""This module contains tests for post-processing of results."""

from __future__ import annotations

import numpy as np
import pytest

from tno.quantum.optimization.qubo.components import BasicResult, Freq
from tno.quantum.problems.portfolio_optimization.components import Decoder, pareto_front
from tno.quantum.problems.portfolio_optimization.test import make_test_dataset

# region Decoder


@pytest.fixture(name="decoder")
def decoder_fixture() -> Decoder:
    portfolio_data = make_test_dataset()
    return Decoder(portfolio_data, k=2)


def test_decode_sampleset(decoder: Decoder) -> None:
    freq = Freq(
        bitvectors=[
            "0000",
            "0001",
            "0010",
            "0011",
            "0100",
            "0101",
            "0110",
            "0111",
            "1000",
            "1001",
            "1010",
            "1011",
            "1100",
            "1101",
            "1110",
            "1111",
        ],
        energies=[int(i) for i in range(16)],
        num_occurrences=[1 for _ in range(16)],
    )
    result = BasicResult(best_bitvector="0000", best_value=0.0, freq=freq)

    expected_result = np.array(
        [
            [10, 30],
            [10, 36],
            [10, 33],
            [10, 39],
            [16, 30],
            [16, 36],
            [16, 33],
            [16, 39],
            [13, 30],
            [13, 36],
            [13, 33],
            [13, 39],
            [19, 30],
            [19, 36],
            [19, 33],
            [19, 39],
        ]
    )

    np.testing.assert_array_equal(decoder.decode_result(result), expected_result)


# region Pareto Front
@pytest.fixture(name="x_y_points")
def x_y_points_fixture() -> tuple[list[int], list[int]]:
    # Create point with a square in a square inside another square
    x_vals = [0, 0, 5, 5, 1, 1, 4, 4, 2, 2, 3, 3]
    y_vals = [0, 5, 0, 5, 1, 4, 1, 4, 2, 3, 2, 3]
    return x_vals, y_vals


@pytest.mark.parametrize(
    ("min_points", "expected_points"),
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
    x_vals, y_vals = x_y_points
    x_par, y_par = pareto_front(x_vals, y_vals, min_points, upper_right_quadrant=False)
    assert len(x_par) == len(expected_points)
    assert len(y_par) == len(expected_points)
    for point in zip(x_par, y_par):
        assert point in expected_points  # type: ignore[comparison-overlap]


def test_small_front() -> None:
    x_vals = [1, 2]
    y_vals = [3, 4]
    x_par, y_par = pareto_front(x_vals, y_vals)
    np.testing.assert_array_equal(x_vals, x_par)
    np.testing.assert_array_equal(y_vals, y_par)


def test_small_front2() -> None:
    x_vals = [0, 0, 5, 5, 1]
    y_vals = [0, 5, 0, 5, 1]
    expected_points = {(0, 0), (0, 5), (5, 0), (5, 5)}
    x_par, y_par = pareto_front(x_vals, y_vals, upper_right_quadrant=False)

    assert len(x_par) == len(expected_points)
    assert len(y_par) == len(expected_points)
    for point in zip(x_par, y_par):
        assert point in expected_points  # type: ignore[comparison-overlap]
