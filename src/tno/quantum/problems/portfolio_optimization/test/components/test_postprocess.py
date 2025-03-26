"""This module contains tests for post-processing of results."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from dimod import SampleSet
from numpy.typing import NDArray

from tno.quantum.problems.portfolio_optimization.components import Decoder, pareto_front
from tno.quantum.problems.portfolio_optimization.test import make_test_dataset


@pytest.fixture(name="decoder")
def decoder_fixture() -> Decoder:
    portfolio_data = make_test_dataset()
    return Decoder(portfolio_data, k=2)


@pytest.mark.parametrize(
    ("sample", "expected_results"),
    [
        ({0: 0, 1: 0, 2: 0, 3: 0}, np.array([10, 30])),
        ({0: 0, 1: 0, 2: 0, 3: 1}, np.array([10, 36])),
        ({0: 0, 1: 0, 2: 1, 3: 0}, np.array([10, 33])),
        ({0: 0, 1: 0, 2: 1, 3: 1}, np.array([10, 39])),
        ({0: 0, 1: 1, 2: 0, 3: 0}, np.array([16, 30])),
        ({0: 0, 1: 1, 2: 0, 3: 1}, np.array([16, 36])),
        ({0: 0, 1: 1, 2: 1, 3: 0}, np.array([16, 33])),
        ({0: 0, 1: 1, 2: 1, 3: 1}, np.array([16, 39])),
        ({0: 1, 1: 0, 2: 0, 3: 0}, np.array([13, 30])),
        ({0: 1, 1: 0, 2: 0, 3: 1}, np.array([13, 36])),
        ({0: 1, 1: 0, 2: 1, 3: 0}, np.array([13, 33])),
        ({0: 1, 1: 0, 2: 1, 3: 1}, np.array([13, 39])),
        ({0: 1, 1: 1, 2: 0, 3: 0}, np.array([19, 30])),
        ({0: 1, 1: 1, 2: 0, 3: 1}, np.array([19, 36])),
        ({0: 1, 1: 1, 2: 1, 3: 0}, np.array([19, 33])),
        ({0: 1, 1: 1, 2: 1, 3: 1}, np.array([19, 39])),
    ],
)
def test_decode_sample(
    decoder: Decoder, sample: Mapping[int, int], expected_results: NDArray[np.float64]
) -> None:
    np.testing.assert_array_equal(decoder.decode_sample(sample), expected_results)


def test_decode_sampleset(decoder: Decoder) -> None:
    samples = [
        {0: 0, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 0, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: 1},
        {0: 0, 1: 1, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 1, 3: 0},
        {0: 1, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 1, 1: 1, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 1, 3: 0},
        {0: 1, 1: 1, 2: 1, 3: 1},
    ]
    sampleset = SampleSet.from_samples(samples, "BINARY", [0] * 16)  # type: ignore[no-untyped-call]
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

    np.testing.assert_array_equal(decoder.decode_sampleset(sampleset), expected_result)


# ---- TESTS FOR THE PARETO FRONT ------
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
        assert point in expected_points


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
        assert point in expected_points
