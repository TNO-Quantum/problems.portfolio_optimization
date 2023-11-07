import numpy as np
import pytest
from pandas import DataFrame

from tno.quantum.problems.portfolio_optimization.components import QuboFactory


@pytest.fixture(name="qubo_factory")
def qubo_factory_fixture() -> QuboFactory:
    columns = [
        "out_2021",
        "out_2030_min",
        "out_2030_max",
        "emis_intens_2021",
        "emis_intens_2030",
        "income_2021",
        "regcap_2021",
    ]
    index = ["asset 1", "asset 2"]
    data = [
        [1.0, 10.0, 19.0, 100.0, 76.0, 0.0, 0.0],
        [2.0, 30.0, 39.0, 200.0, 152.0, 0.0, 0.0],
    ]
    portfolio_data = DataFrame(data=data, columns=columns, index=index)
    return QuboFactory(portfolio_data, 0, 2)


def test_calc_minimize_hhi(qubo_factory: QuboFactory) -> None:
    expected_qubo = (
        np.array([[69, 36, 0, 0], [0, 156, 0, 0], [0, 0, 189, 36], [0, 0, 0, 396]])
        / 2401
    )
    expected_offset = 1000 / 2401
    qubo, offset = qubo_factory.calc_minimize_HHI()

    np.testing.assert_almost_equal(offset, expected_offset)
    np.testing.assert_almost_equal(qubo, expected_qubo)


def test_calc_minimize_hhi(qubo_factory: QuboFactory) -> None:
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
    qubo, offset = qubo_factory.calc_emission_constraint()

    np.testing.assert_almost_equal(offset, expected_offset)
    np.testing.assert_almost_equal(qubo, expected_qubo)
