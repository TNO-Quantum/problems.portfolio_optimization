"""This module contains test for the ``QuboCompiler`` class."""
from __future__ import annotations

from typing import Any, MutableSequence

import numpy as np
import pytest
from pandas import DataFrame

from tno.quantum.problems.portfolio_optimization.components import (
    QuboCompiler,
    QuboFactory,
)


@pytest.fixture(name="qubo_compiler")
def qubo_compiler_fixture() -> QuboCompiler:
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
    return QuboCompiler(portfolio_data, 0, 2)


def test_init(qubo_compiler: QuboCompiler) -> None:
    assert isinstance(qubo_compiler._qubo_factory, QuboFactory)
    assert isinstance(qubo_compiler._to_compile, MutableSequence)
    assert isinstance(qubo_compiler._compiled_qubos, MutableSequence)

    assert len(qubo_compiler._to_compile) == 0
    assert len(qubo_compiler._compiled_qubos) == 0


@pytest.mark.parametrize(
    "method_name,method_args",
    [
        ("add_minimize_HHI", []),
        ("add_emission_constraint", []),
        ("add_growth_factor_constraint", [5]),
        ("add_maximize_ROC", [1]),
        ("add_maximize_ROC", [4]),
    ],
)
def test_add_single_qubo(
    qubo_compiler: QuboCompiler, method_name: str, method_args: list[Any]
) -> None:
    assert len(qubo_compiler._to_compile) == 0
    assert len(qubo_compiler._compiled_qubos) == 0

    assert hasattr(qubo_compiler, method_name)
    getattr(qubo_compiler, method_name)(*method_args)

    assert len(qubo_compiler._to_compile) == 1
    assert len(qubo_compiler._compiled_qubos) == 0

    qubo_compiler.compile()

    assert len(qubo_compiler._to_compile) == 1
    assert len(qubo_compiler._compiled_qubos) == 1

    qubo, _ = qubo_compiler.make_qubo(1)
    assert len(qubo) == 4
    np.testing.assert_array_equal(qubo, qubo_compiler._compiled_qubos[0])


@pytest.mark.parametrize(
    "method_name,method_args,len_qubo",
    [
        ("add_maximize_ROC", [2, 1], 4),
        ("add_maximize_ROC", [3, 0, 5], 9),
    ],
)
def test_add_double_qubo(
    qubo_compiler: QuboCompiler, method_name: str, method_args: list[Any], len_qubo: int
) -> None:
    assert len(qubo_compiler._to_compile) == 0
    assert len(qubo_compiler._compiled_qubos) == 0

    assert hasattr(qubo_compiler, method_name)
    getattr(qubo_compiler, method_name)(*method_args)

    assert len(qubo_compiler._to_compile) == 2
    assert len(qubo_compiler._compiled_qubos) == 0

    qubo_compiler.compile()

    assert len(qubo_compiler._to_compile) == 2
    assert len(qubo_compiler._compiled_qubos) == 2

    qubo, _ = qubo_compiler.make_qubo(1, 2)
    assert len(qubo) == len_qubo
    np.testing.assert_array_equal(
        qubo, qubo_compiler._compiled_qubos[0] + 2 * qubo_compiler._compiled_qubos[1]
    )


def test_mixing(qubo_compiler: QuboCompiler) -> None:
    assert len(qubo_compiler._to_compile) == 0
    assert len(qubo_compiler._compiled_qubos) == 0

    qubo_compiler.add_minimize_HHI()
    assert len(qubo_compiler._to_compile) == 1
    assert len(qubo_compiler._compiled_qubos) == 0

    qubo_compiler.add_maximize_ROC(formulation=2, capital_growth_factor=1)
    assert len(qubo_compiler._to_compile) == 3
    assert len(qubo_compiler._compiled_qubos) == 0

    qubo_compiler.add_growth_factor_constraint(growth_target=1.5)
    assert len(qubo_compiler._to_compile) == 4
    assert len(qubo_compiler._compiled_qubos) == 0

    qubo_compiler.add_emission_constraint()
    assert len(qubo_compiler._to_compile) == 5
    assert len(qubo_compiler._compiled_qubos) == 0

    qubo_compiler.compile()
    assert len(qubo_compiler._to_compile) == 5
    assert len(qubo_compiler._compiled_qubos) == 5

    qubo, _ = qubo_compiler.make_qubo(1, 2, 3, 4, 5)
    assert len(qubo) == 4
    np.testing.assert_array_equal(
        qubo, sum((i + 1) * qubo_compiler._compiled_qubos[i] for i in range(5))
    )


def test_incorrect_lambdas(qubo_compiler: QuboCompiler) -> None:
    qubo_compiler.add_emission_constraint()
    qubo_compiler.compile()
    with pytest.raises(ValueError):
        qubo_compiler.make_qubo(1, 2, 3)
