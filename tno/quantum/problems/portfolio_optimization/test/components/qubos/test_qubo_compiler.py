"""This module contains test for the ``QuboCompiler`` class."""
from __future__ import annotations

from collections.abc import MutableSequence
from typing import Any

import numpy as np
import pytest

from tno.quantum.problems.portfolio_optimization.components import (
    QuboCompiler,
    QuboFactory,
)
from tno.quantum.problems.portfolio_optimization.test import make_test_dataset

# pylint: disable=missing-function-docstring


@pytest.fixture(name="qubo_compiler")
def qubo_compiler_fixture() -> QuboCompiler:
    portfolio_data = make_test_dataset()
    return QuboCompiler(portfolio_data, k=2)


def test_init(qubo_compiler: QuboCompiler) -> None:
    assert isinstance(qubo_compiler._qubo_factory, QuboFactory)
    assert isinstance(qubo_compiler._to_compile, MutableSequence)
    assert isinstance(qubo_compiler._compiled_qubos, MutableSequence)

    assert len(qubo_compiler._to_compile) == 0
    assert len(qubo_compiler._compiled_qubos) == 0


@pytest.mark.parametrize(
    "method_name,method_args",
    [
        ("add_minimize_hhi", []),
        ("add_emission_constraint", ["emis_intens_now", "emis_intens_future", 0.7]),
        ("add_growth_factor_constraint", [5]),
        ("add_maximize_roc", [1]),
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
        ("add_maximize_roc", [2, 5], 9),
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

    qubo_compiler.add_minimize_hhi()
    assert len(qubo_compiler._to_compile) == 1
    assert len(qubo_compiler._compiled_qubos) == 0

    qubo_compiler.add_maximize_roc(formulation=2, ancilla_variables=5)
    assert len(qubo_compiler._to_compile) == 3
    assert len(qubo_compiler._compiled_qubos) == 0

    qubo_compiler.add_growth_factor_constraint(growth_target=1.5)
    assert len(qubo_compiler._to_compile) == 4
    assert len(qubo_compiler._compiled_qubos) == 0

    qubo_compiler.add_emission_constraint(emission_now="emis_intens_now")
    assert len(qubo_compiler._to_compile) == 5
    assert len(qubo_compiler._compiled_qubos) == 0

    qubo_compiler.compile()
    assert len(qubo_compiler._to_compile) == 5
    assert len(qubo_compiler._compiled_qubos) == 5

    qubo, _ = qubo_compiler.make_qubo(1, 2, 3, 4, 5)
    assert len(qubo) == 9
    np.testing.assert_array_equal(
        qubo, sum((i + 1) * qubo_compiler._compiled_qubos[i] for i in range(5))
    )


def test_ordering() -> None:
    """Test order"""
    portfolio_data = make_test_dataset()

    qubo_compiler1 = QuboCompiler(portfolio_data, k=2)
    qubo_compiler1.add_minimize_hhi()
    qubo_compiler1.add_maximize_roc(formulation=2)
    qubo1, offset1 = qubo_compiler1.compile().make_qubo(1, 1, 1)

    qubo_compiler2 = QuboCompiler(portfolio_data, k=2)
    qubo_compiler2.add_maximize_roc(formulation=2)
    qubo_compiler2.add_minimize_hhi()
    qubo2, offset2 = qubo_compiler2.compile().make_qubo(1, 1, 1)
    assert np.allclose(qubo1, qubo2)
    assert np.equal(offset1, offset2) | (np.isnan(offset1) & np.isnan(offset2))


def test_incorrect_lambdas(qubo_compiler: QuboCompiler) -> None:
    qubo_compiler.add_emission_constraint(emission_now="emis_intens_now")
    qubo_compiler.compile()
    with pytest.raises(ValueError):
        qubo_compiler.make_qubo(1, 2, 3)
