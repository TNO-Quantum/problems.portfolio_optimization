"""This module contains the ``QuboCompiler`` class.

The ``QuboCompiler`` can create a variety of QUBO formulation by combining different
objectives and constraints with their corresponding penalty or preference parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, TypeVar

from tno.quantum.problems.portfolio_optimization._components.qubos._qubo_factory import (
    QuboFactory,
)

if TYPE_CHECKING:
    from tno.quantum.optimization.qubo.components import QUBO
    from tno.quantum.problems.portfolio_optimization._components.io import PortfolioData

QuboCompilerT = TypeVar("QuboCompilerT", bound="QuboCompiler")


class QuboCompiler:
    """QuboCompiler - A compiler class for creating QUBO instances.

    This class provides a convenient interface for combining different QUBO formulations
    without needing to worry about the qubo size.

    Methods:
    - `add_minimize_hhi`: Adds the to minimize HHI QUBO to the compile list.
    - `add_maximize_roc`: Adds a ROC and optionally a stabilizing QUBO to the compile
      list.
    - `add_emission_constraint`: Adds an emission constraint QUBO to the compile list.
    - `add_growth_factor_constraint`: Adds the growth factor constraint QUBO to the
      compile list.

    """

    def __init__(self, portfolio_data: PortfolioData, k: int) -> None:
        """Init of the ``QuboCompiler`` class.

        The ``QuboCompiler`` can create a variety of QUBO formulation by combining
        different objectives and constraints with penalty or preference parameters.

        Args:
            portfolio_data: A ``PortfolioData`` object containing the portfolio to
                optimize.
            k: The number of bits that are used to represent the outstanding amount for
                each asset. A fixed point representation is used to represent `$2^k$`
                different equidistant values in the range `$[LB_i, UB_i]$` for asset i.
        """
        self._qubo_factory = QuboFactory(portfolio_data, k)

        self._to_compile: list[Callable[[], QUBO]] = []
        self._compiled_qubos: list[QUBO] = []

    def add_minimize_hhi(
        self: QuboCompilerT,
    ) -> QuboCompilerT:
        r"""Adds the minimize HHI objective to the compile list.

        The HHI objective is given by

        $$HHI(x) = \sum_{i=1}^N\left(\frac{x_i}{\sum_{j=1}^N x_j}\right)^2,$$

        where

            - `$N$` is the total number of assets,
            - `$x_i$` is the future outstanding amount for asset `$i$`.

        For the QUBO formulation, see the docs of
        :py:meth:`~portfolio_optimization.components.qubos.QuboFactory.calc_minimize_hhi`.

        Returns:
            Self.
        """
        self._to_compile.append(self._qubo_factory.calc_minimize_hhi)
        return self

    def add_maximize_roc(
        self: QuboCompilerT,
        formulation: int,
        ancilla_variables: int = 0,
    ) -> QuboCompilerT:
        """Adds the maximize ROC objective and a stabilize c constraint.

        Args:
            formulation: Integer representing which formulation to pick. If formulation
                is ``1``, then one QUBO term will be added. If formulation is ``2``,
                then 2 QUBO terms will be added as well, but the argument
                `ancilla_variables` must be provided.
            ancilla_variables: Number of ancilla variables to use for formulation ``2``.

        Returns:
            Self.
        """
        if formulation == 1:
            self._to_compile.append(self._qubo_factory.calc_maximize_roc1)
        elif formulation == 2:
            self._qubo_factory.n_vars += ancilla_variables
            self._to_compile.append(self._qubo_factory.calc_maximize_roc2)
            self._to_compile.append(self._qubo_factory.calc_stabilize_c)

        return self

    def add_emission_constraint(
        self: QuboCompilerT,
        emission_now: str,
        emission_future: str | None = None,
        reduction_percentage_target: float = 0.7,
    ) -> QuboCompilerT:
        r"""Adds the emission constraint to the compile list.

        The constraint is given by

        .. math::

            \frac{\sum_{i=1}^Nf_i \cdot x_i}{\sum_{i=1}^N x_i}
            =
            g_e \frac{\sum_{i=1}^Ne_i \cdot y_i}{\sum_{i=1}^N y_i},

        where:

            - `$x_i$` is the future outstanding amount for asset `$i$`,
            - `$y_i$` is the current outstanding amount for asset `$i$`,
            - `$e_i$` is the current emission intensity for asset `$i$`,
            - `$f_i$` is the expected emission intensity at the future for asset `$i$`,
            - `$g_e$` is the target value for the relative emission reduction.

        For the QUBO formulation, see the docs of
        :py:meth:`~portfolio_optimization.components.qubos.QuboFactory.calc_emission_constraint`.

        Args:
            emission_now: Name of the column in the portfolio dataset corresponding to
                the variables at current time.
            emission_future: Name of the column in the portfolio dataset corresponding
                to the variables at future time. If no value is provided, it is assumed
                that the value is constant over time, i.e., the variable
                ``emission_now`` will be used.
            reduction_percentage_target: target value for reduction percentage amount.

        Returns:
            Self.
        """
        method = partial(
            self._qubo_factory.calc_emission_constraint,
            emission_now=emission_now,
            emission_future=emission_future,
            reduction_percentage_target=reduction_percentage_target,
        )
        self._to_compile.append(method)
        return self

    def add_growth_factor_constraint(
        self: QuboCompilerT, growth_target: float
    ) -> QuboCompilerT:
        r"""Adds the capital growth factor constraint to the compile list.

        The constraint is given by

        $$\frac{\sum_{i=1}^N x_i}{\sum_{i=1}^N y_i} = g_c,$$

        where

            - `$N$` is the total number of assets,
            - `$x_i$` is the future outstanding amount for asset `$i$`,
            - `$y_i$` is the current outstanding amount for asset `$i$`,
            - `$g_c$` is the target value for the total growth factor.

        For the QUBO formulation, see the docs of
        :py:meth:`~portfolio_optimization.components.qubos.QuboFactory.calc_growth_factor_constraint`.

        Args:
            growth_target: target value for growth factor total outstanding amount.

        Returns:
            Self.
        """
        method = partial(
            self._qubo_factory.calc_growth_factor_constraint,
            growth_target=growth_target,
        )
        self._to_compile.append(method)
        return self

    def compile(self: QuboCompilerT) -> QuboCompilerT:
        """Compiles all QUBOs in the compile list.

        Returns:
            Self.
        """
        self._compiled_qubos = []
        for constructor in self._to_compile:
            self._compiled_qubos.append(constructor())
        return self

    def make_qubo(self, *lambdas: float) -> QUBO:
        """Makes a QUBO of the entire problem with the given lambdas.

        Args:
            lambdas: Scaling parameters for each QUBO in the formulation.

        Returns:
            The combined QUBO matrix.
        """
        if len(lambdas) != len(self._compiled_qubos):
            error_msg = (
                "Number of lambdas does not correspond with the number of Hamiltonians."
            )
            raise ValueError(error_msg)

        combined_qubo = lambdas[0] * self._compiled_qubos[0]
        for lambda_i, qubo_i in zip(lambdas[1:], self._compiled_qubos[1:]):
            combined_qubo += lambda_i * qubo_i

        return combined_qubo
