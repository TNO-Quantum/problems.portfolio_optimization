"""This module contains tests for the ``QuboFactory`` class."""
import numpy as np
import pytest

from tno.quantum.problems.portfolio_optimization.components import QuboFactory
from tno.quantum.problems.portfolio_optimization.test import make_test_dataset

# pylint: disable=missing-function-docstring


@pytest.fixture(name="qubo_factory")
def qubo_factory_fixture() -> QuboFactory:
    portfolio_data = make_test_dataset()
    return QuboFactory(portfolio_data, k=2)


def test_calc_minimize_hhi(qubo_factory: QuboFactory) -> None:
    expected_qubo = np.array(
        [[69, 36, 0, 0], [0, 156, 0, 0], [0, 0, 189, 36], [0, 0, 0, 396]]
    )
    expected_offset = 1000
    qubo, offset = qubo_factory.calc_minimize_hhi()

    np.testing.assert_almost_equal(offset, expected_offset)
    np.testing.assert_almost_equal(qubo, expected_qubo)


def test_calc_emission_constraint(qubo_factory: QuboFactory) -> None:
    expected_qubo = (
        np.array(
            [
                [-325_191, 133_956, -58_194, -116_388],
                [0, -583_404, -116_388, -232_776],
                [0, 0, 336_921, 101_124],
                [0, 0, 0, 724_404],
            ]
        )
        / 562_500
    )
    expected_offset = 960_400 / 562_500
    qubo, offset = qubo_factory.calc_emission_constraint(
        emission_now="emis_intens_now", emission_future="emis_intens_future"
    )

    np.testing.assert_almost_equal(offset, expected_offset)
    np.testing.assert_almost_equal(qubo, expected_qubo)


def test_calc_growth_factor_constraint(qubo_factory: QuboFactory) -> None:
    expected_qubo = [
        [25, 4, 2, 4],
        [0, 52, 4, 8],
        [0, 0, 25, 4],
        [0, 0, 0, 52],
    ]
    expected_offset = 144
    qubo, offset = qubo_factory.calc_growth_factor_constraint(4 / 3)

    np.testing.assert_almost_equal(offset, expected_offset)
    np.testing.assert_almost_equal(qubo, expected_qubo)


def test_calc_maximize_roc1(qubo_factory: QuboFactory) -> None:
    expected_qubo = np.diag([-3, -6, -3 / 4, -3 / 2])
    expected_offset = -17.5
    qubo, offset = qubo_factory.calc_maximize_roc1()

    np.testing.assert_almost_equal(offset, expected_offset)
    np.testing.assert_almost_equal(qubo, expected_qubo)


def test_calc_maximize_roc2(qubo_factory: QuboFactory) -> None:
    # Use 3 ancilla variables
    qubo_factory.n_vars += 3

    expected_qubo = [
        [-3, 0, 0, 0, 3 / 4, 9 / 16, 21 / 64],
        [0, -6, 0, 0, 3 / 2, 9 / 8, 21 / 32],
        [0, 0, -3 / 4, 0, 3 / 16, 9 / 64, 21 / 256],
        [0, 0, 0, -3 / 2, 3 / 8, 9 / 32, 21 / 128],
        [0, 0, 0, 0, 35 / 8, 0, 0],
        [0, 0, 0, 0, 0, 105 / 32, 0],
        [0, 0, 0, 0, 0, 0, 245 / 128],
    ]
    expected_offset = -17.5
    qubo, offset = qubo_factory.calc_maximize_roc2()

    np.testing.assert_almost_equal(offset, expected_offset)
    np.testing.assert_almost_equal(qubo, expected_qubo)


def test_calc_stabilize_c(qubo_factory: QuboFactory) -> None:
    # Use 3 ancilla variables
    qubo_factory.n_vars += 3

    expected_qubo = np.array(
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
    qubo, offset = qubo_factory.calc_stabilize_c()

    np.testing.assert_almost_equal(offset, expected_offset)
    np.testing.assert_almost_equal(qubo, expected_qubo)
