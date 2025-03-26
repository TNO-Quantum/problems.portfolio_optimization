"""This module contains tests for the ``QuboFactory`` class."""

import numpy as np
import pytest

from tno.quantum.problems.portfolio_optimization.components import QuboFactory
from tno.quantum.problems.portfolio_optimization.test import make_test_dataset


@pytest.fixture(name="qubo_factory")
def qubo_factory_fixture() -> QuboFactory:
    portfolio_data = make_test_dataset()
    return QuboFactory(portfolio_data, k=2)


def test_calc_minimize_hhi(qubo_factory: QuboFactory) -> None:
    expected_matrix = np.array(
        [[69, 36, 0, 0], [0, 156, 0, 0], [0, 0, 189, 36], [0, 0, 0, 396]]
    )
    expected_offset = 1000
    qubo = qubo_factory.calc_minimize_hhi()

    np.testing.assert_almost_equal(qubo.offset, expected_offset)
    np.testing.assert_almost_equal(qubo.matrix, expected_matrix)


def test_calc_emission_constraint(qubo_factory: QuboFactory) -> None:
    expected_matrix = (
        np.array(
            [
                [-433_588, 178_608, -77_592, -155_184],
                [0, -777_872, -155_184, -310_368],
                [0, 0, 449_228, 134_832],
                [0, 0, 0, 965_872],
            ]
        )
        / 3
    )
    expected_offset = 3_841_600 / 9
    qubo = qubo_factory.calc_emission_constraint(
        emission_now="emis_intens_now", emission_future="emis_intens_future"
    )

    np.testing.assert_almost_equal(qubo.offset, expected_offset)
    np.testing.assert_almost_equal(qubo.matrix, expected_matrix)


def test_calc_growth_factor_constraint(qubo_factory: QuboFactory) -> None:
    expected_matrix = [
        [25, 4, 2, 4],
        [0, 52, 4, 8],
        [0, 0, 25, 4],
        [0, 0, 0, 52],
    ]
    expected_offset = 144
    qubo = qubo_factory.calc_growth_factor_constraint(4 / 3)

    np.testing.assert_almost_equal(qubo.offset, expected_offset)
    np.testing.assert_almost_equal(qubo.matrix, expected_matrix)


def test_calc_maximize_roc1(qubo_factory: QuboFactory) -> None:
    expected_matrix = np.diag([-3, -6, -3 / 4, -3 / 2])
    expected_offset = -17.5
    qubo = qubo_factory.calc_maximize_roc1()

    np.testing.assert_almost_equal(qubo.offset, expected_offset)
    np.testing.assert_almost_equal(qubo.matrix, expected_matrix)


def test_calc_maximize_roc2(qubo_factory: QuboFactory) -> None:
    # Use 3 ancilla variables
    qubo_factory.n_vars += 3

    expected_matrix = [
        [-3, 0, 0, 0, 3 / 4, 9 / 16, 21 / 64],
        [0, -6, 0, 0, 3 / 2, 9 / 8, 21 / 32],
        [0, 0, -3 / 4, 0, 3 / 16, 9 / 64, 21 / 256],
        [0, 0, 0, -3 / 2, 3 / 8, 9 / 32, 21 / 128],
        [0, 0, 0, 0, 35 / 8, 0, 0],
        [0, 0, 0, 0, 0, 105 / 32, 0],
        [0, 0, 0, 0, 0, 0, 245 / 128],
    ]
    expected_offset = -17.5
    qubo = qubo_factory.calc_maximize_roc2()

    np.testing.assert_almost_equal(qubo.offset, expected_offset)
    np.testing.assert_almost_equal(qubo.matrix, expected_matrix)


def test_calc_stabilize_c(qubo_factory: QuboFactory) -> None:
    # Use 3 ancilla variables
    qubo_factory.n_vars += 3

    expected_matrix = np.array(
        [
            [147, 36, 9, 18, -6, -3, -3 / 2],
            [0, 312, 18, 36, -12, -6, -3],
            [0, 0, 71.25, 9, -3, -3 / 2, -3 / 4],
            [0, 0, 0, 147, -6, -3, -3 / 2],
            [0, 0, 0, 0, -45, 1, 1 / 2],
            [0, 0, 0, 0, 0, -22.75, 1 / 4],
            [0, 0, 0, 0, 0, 0, -11.4375],
        ]
    )
    expected_offset = 529
    qubo = qubo_factory.calc_stabilize_c()

    np.testing.assert_almost_equal(qubo.offset, expected_offset)
    np.testing.assert_almost_equal(qubo.matrix, expected_matrix)
