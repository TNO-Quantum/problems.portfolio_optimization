"""This module contains the ``QuboFactory`` class.

The ``QuboFactory`` class provides a convenient interface for constructing intermediate
QUBO matrices for different objectives and constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tno.quantum.optimization.qubo.components import QUBO

if TYPE_CHECKING:
    from tno.quantum.problems.portfolio_optimization.components.io import PortfolioData


class QuboFactory:
    """QuboFactory - A factory class for creating QUBO instances.

    This class provides a convenient interface for constructing intermediate QUBO
    matrices for different objectives and constraints.

    Methods:
    - `calc_minimize_hhi`: Calculates the to minimize HHI QUBO
    - `calc_maximize_roc1`: Calculates the to maximize return on capital QUBO variant 1
    - `calc_maximize_roc2`: Calculates the to maximize return on capital QUBO variant 2
    - `calc_emission_constraint`: Calculates the emission constraint QUBO
    - `calc_growth_factor_constraint`: Calculates the growth factor constraint QUBO
    - `calc_stabilize_c`: Calculates the constraint QUBO that stabilizes growth factor.
    """

    def __init__(self, portfolio_data: PortfolioData, k: int) -> None:
        """Init of the ``QuboFactory``.

        Args:
            portfolio_data: A ``PortfolioData`` object containing the portfolio to
                optimize.
            k: The number of bits that are used to represent the outstanding amount for
                each asset. A fixed point representation is used to represent `$2^k$`
                different equidistant values in the range `$[LB_i, UB_i]$` for asset i.
        """
        self.portfolio_data = portfolio_data
        self.number_of_assets = len(portfolio_data)
        self.n_vars = self.number_of_assets * k
        self.outstanding_now = portfolio_data.get_outstanding_now()
        self.l_bound = portfolio_data.get_l_bound()
        self.u_bound = portfolio_data.get_u_bound()
        self.income = portfolio_data.get_income()
        self.returns = portfolio_data.get_returns()
        self.capital = portfolio_data.get_capital()
        self.k = k

    def calc_minimize_hhi(self) -> QUBO:
        r"""Calculates the to minimize HHI QUBO.

        The QUBO formulation is given by

        .. math::

            QUBO(x)
            =
            \sum_{i=1}^N\left(LB_i + \frac{UB_i-LB_i}{2^k-1}\sum_{j=0}^{k-1}2^j\cdot x_{i,j}\right)^2,

        where:

            - `$LB_i$` is the lower bound for asset `$i$`,
            - `$UB_i$` is the upper bound for asset `$i$`,
            - `$k$` is the number of bits,
            - and `$x_{i,j}$` are the $k$ binary variables for asset `$i$` with $j<k$.

        Returns:
            The QUBO.
        """  # noqa: E501
        qubo = np.zeros((self.n_vars, self.n_vars))
        offset = np.sum(self.l_bound**2)
        multiplier = (self.u_bound - self.l_bound) / (2**self.k - 1)
        for asset_i, (l_bound_i, multiplier_i) in enumerate(
            zip(self.l_bound, multiplier)
        ):
            # For asset i: (LB + multiplier * sum_j 2**j x_j) ** 2
            for bit_j in range(self.k):
                idx_1 = asset_i * self.k + bit_j

                # Diagonal elements: 2 * LB * multiplier * sum_j 2**j x_j
                qubo[idx_1, idx_1] += l_bound_i * multiplier_i * 2 ** (bit_j + 1)
                qubo[idx_1, idx_1] += multiplier_i**2 * 2 ** (2 * bit_j)

                # Elements: multiplier**2 * (sum_j 2**j x_j) * (sum_j' 2**j' x_j')
                for bit_j_prime in range(bit_j + 1, self.k):
                    idx_2 = asset_i * self.k + bit_j_prime
                    qubo[idx_1, idx_2] += multiplier_i**2 * 2 ** (
                        bit_j + bit_j_prime + 1
                    )

        return QUBO(qubo, offset)

    def calc_emission_constraint(
        self,
        emission_now: str,
        emission_future: str | None = None,
        reduction_percentage_target: float = 0.7,
    ) -> QUBO:
        r"""Calculate emission constraint QUBO for arbitrary reduction target.

        The QUBO formulation is given by

        .. math::

            QUBO(x)
            &=
            \left(
            \sum_{i=1}^N f_i x_i
            - g_e E
            \sum_{i=1}^N x_i
            \right)^2,

            x_i & = LB_i+\frac{UB_i-LB_i}{2^k-1}\sum_{j=0}^{k-1}2^j\cdot x_{i,j},

            E &= \frac{\sum_{i=1}^N e_i \cdot y_i}{\sum_{i=1}^N y_i},

        where:

            - `$LB_i$` is the lower bound for asset `$i$`,
            - `$UB_i$` is the upper bound for asset `$i$`,
            - `$k$` is the number of bits,
            - `$e_i$` is the current emission intensity for asset `$i$`,
            - `$f_i$` is the expected emission intensity at the future for asset `$i$`,
            - `$y_i$` is the current outstanding amount for asset `$i$`,
            - `$g_e$` is the target value for the relative emission reduction,
            - and `$x_{i,j}$` are the $k$ binary variables for asset `$i$` with $j<k$.

        Args:
            emission_now: Name of the column in the portfolio dataset corresponding to
                the variables at current time.
            emission_future: Name of the column in the portfolio dataset corresponding
                to the variables at future time. If no value is provided, it is assumed
                that the value is constant over time, i.e., the variable
                ``emission_now`` will be used.
            reduction_percentage_target: Target value for reduction percentage amount.

        Raises:
            KeyError: if the provided column names are not in the portfolio_data.

        Returns:
            The QUBO.
        """
        if emission_future is None:
            emission_future = emission_now

        if emission_now not in self.portfolio_data:
            error_msg = f"Column name {emission_now} not present in portfolio dataset."
            raise KeyError(error_msg)
        if emission_future not in self.portfolio_data:
            error_msg = (
                f"Column name {emission_future} not present in portfolio dataset."
            )
            raise KeyError(error_msg)

        emission_intensity_now = self.portfolio_data.get_column(emission_now)
        emission_intensity_future = self.portfolio_data.get_column(emission_future)

        total_emission_now = np.sum(emission_intensity_now * self.outstanding_now)
        relelative_total_emission_now = total_emission_now / np.sum(
            self.outstanding_now
        )

        alpha = np.sum(
            (
                emission_intensity_future
                - reduction_percentage_target * relelative_total_emission_now
            )
            * self.l_bound
        )

        mantisse = np.power(2, np.arange(self.k))
        multiplier = (
            (
                emission_intensity_future
                - reduction_percentage_target * relelative_total_emission_now
            )
            * (self.u_bound - self.l_bound)
            / (2**self.k - 1)
        )
        beta = np.kron(multiplier, mantisse)
        qubo = np.zeros((self.n_vars, self.n_vars))

        offset = alpha**2
        for idx1 in range(self.number_of_assets * self.k):
            qubo[idx1, idx1] += 2 * alpha * beta[idx1] + beta[idx1] ** 2
            for idx2 in range(idx1 + 1, self.number_of_assets * self.k):
                qubo[idx1, idx2] += 2 * beta[idx1] * beta[idx2]

        return QUBO(qubo, offset)

    def calc_growth_factor_constraint(self, growth_target: float) -> QUBO:
        r"""Calculates the growth factor constraint QUBO.

        The QUBO formulation is given by

        .. math::

            QUBO(x)
            =
            \left(
            \frac{\sum_{i=1}^N LB_i + \frac{UB_i-LB_i}{2^k-1}\sum_{j=0}^{k-1} 2^j\cdot x_{i,j}}{\sum_{i=1}^N y_i}
            - g_c
            \right)^2

        where:

            - `$LB_i$` is the lower bound for asset `$i$`,
            - `$UB_i$` is the upper bound for asset `$i$`,
            - `$k$` is the number of bits,
            - `$g_c$` is the target value for the total growth factor,
            - `$y_i$` is the current outstanding amount for asset `$i$`,
            - and `$x_{i,j}$` are the $k$ binary variables for asset `$i$` with $j<k$.

        Args:
            growth_target: target value for growth factor total outstanding amount.

        Returns:
            The QUBO.
        """  # noqa: E501
        total_outstanding_now = np.sum(self.outstanding_now)
        alpha = np.sum(self.l_bound) / total_outstanding_now - growth_target

        mantisse = np.power(2, np.arange(self.k))
        multiplier = (self.u_bound - self.l_bound) / (
            (2**self.k - 1) * total_outstanding_now
        )
        beta = np.kron(multiplier, mantisse)

        qubo = np.zeros((self.n_vars, self.n_vars))
        # We only fill the upper left part of the matrix, since we don't use ancilla
        # variables
        qubo_slice = qubo[
            : self.k * self.number_of_assets, : self.k * self.number_of_assets
        ]
        # Add the off diagonal elements
        qubo_slice += np.triu(2 * np.outer(beta, beta), k=1)
        # Add the diagonal elements
        np.fill_diagonal(qubo_slice, beta**2 + 2 * alpha * beta)
        offset = alpha**2
        return QUBO(qubo, offset)

    def calc_maximize_roc1(self) -> QUBO:
        r"""Calculates the to maximize ROC QUBO for variant 1.

        The QUBO formulation is given by

        .. math::

            QUBO(x)
            =
            -\sum_{i=1}^N\frac{r_i}{c_i\cdot y_i}
            \left(LB_i + \frac{UB_i-LB_i}{2^k-1}\sum_{j=0}^{k-1}2^j\cdot x_{i,j}\right),

        where

            - `$LB_i$` is the lower bound for asset `$i$`,
            - `$UB_i$` is the upper bound for asset `$i$`,
            - `$k$` is the number of bits,
            - `$y_i$` is the current outstanding amount for asset `$i$`,
            - `$r_i$` is the current return for asset `$i$`,
            - `$c_i$` is the regulatory capital for asset `$i$`,
            - and `$x_{i,j}$` are the $k$ binary variables for asset `$i$` with $j<k$.

        Returns:
            The QUBO.
        """
        theta = self.returns / (self.outstanding_now * self.capital)
        offset = np.sum(theta * self.l_bound)
        mantisse = np.power(2, np.arange(self.k))
        multiplier = theta * (self.u_bound - self.l_bound) / (2**self.k - 1)
        qubo_diag = np.kron(multiplier, mantisse)

        qubo = np.diag(qubo_diag)
        return QUBO(-qubo, -offset)

    def calc_maximize_roc2(self) -> QUBO:
        r"""Calculates the to maximize ROC QUBO for variant 2.

        The QUBO formulation is given by

        .. math::

            QUBO(x,g)
            &=
            -
            G_{inv}(g) \cdot
            \sum_{i=1}^N\frac{r_i}{y_i}
            \left(LB_i + \frac{UB_i-LB_i}{2^k-1}\sum_{j=0}^{k-1}2^j\cdot x_{i,j}\right),

            G_{inv}(g) &=
            1 + \sum_{j=0}^{k-1} 2^{-j-1}(2^{-j-1} - 1)\cdot g_{j},

        where

            - `$LB_i$` is the lower bound for asset `$i$`,
            - `$UB_i$` is the upper bound for asset `$i$`,
            - `$k$` is the number of bits,
            - `$a$` is the number of ancilla variables,
            - `$y_i$` is the current outstanding amount for asset `$i$`,
            - `$r_i$` is the return for asset `$i$`,
            - `$c_i$` is the regulatory capital for asset `$i$`,
            - `$x_{i,j}$` are the $k$ binary variables for asset `$i$` with $j<k$.
            - `$g_{j}$` are the $a$ binary ancilla variables with $j<a$.

        Returns:
            The QUBO.
        """
        ancilla_variables = self.n_vars - self.k * self.number_of_assets

        alpha = np.sum(self.l_bound * self.returns / self.outstanding_now)
        mantisse = np.power(2, np.arange(self.k))
        multiplier = (
            self.returns
            * (self.u_bound - self.l_bound)
            / (self.outstanding_now * (2**self.k - 1))
        )
        beta = np.kron(multiplier, mantisse)

        gamma = np.power(2.0, np.arange(-1, -ancilla_variables - 1, -1))
        gamma = gamma**2 - gamma

        qubo = np.zeros((self.n_vars, self.n_vars))
        np.fill_diagonal(
            qubo[: self.number_of_assets * self.k, : self.number_of_assets * self.k],
            beta,
        )
        qubo[: self.number_of_assets * self.k, self.number_of_assets * self.k :] += (
            np.outer(beta, gamma)
        )
        np.fill_diagonal(
            qubo[self.number_of_assets * self.k :, self.number_of_assets * self.k :],
            alpha * gamma,
        )
        offset = alpha
        return QUBO(-qubo, -offset)

    def calc_stabilize_c(self) -> QUBO:
        r"""Calculate QUBO that stabilizes the growth factor in second ROC formulation.

        The QUBO formulation is given by

        .. math::

            QUBO(x,g)
            &=
            \left(
            \sum_{i=1}^N\frac{c_i}{y_i}
            \left(LB_i + \frac{UB_i-LB_i}{2^k-1}\sum_{j=0}^{k-1}2^j\cdot x_{i,j}\right)
            - G_C(g)\sum_{i=1}^N c_i
            \right)^2,

            G_C &= 1 + \sum_{j=0}^{k-1} 2^{-j - 1} \cdot g_j,

        where

            - `$LB_i$` is the lower bound for asset `$i$`,
            - `$UB_i$` is the upper bound for asset `$i$`,
            - `$k$` is the number of bits,
            - `$a$` is the number of ancilla variables,
            - `$y_i$` is the current outstanding amount for asset `$i$`,
            - `$c_i$` is the regulatory capital for asset `$i$`,
            - `$x_{i,j}$` are the $k$ binary variables for asset `$i$` with $j<k$,
            - `$g_j$` are the $a$ ancillary binary variables with $j<a$.

        Returns:
            The QUBO.
        """
        ancilla_variables = self.n_vars - self.k * self.number_of_assets
        alpha = np.sum(self.capital * self.l_bound / self.outstanding_now) - np.sum(
            self.capital
        )

        mantisse = np.power(2, np.arange(self.k))
        multiplier = (
            self.capital
            * (self.u_bound - self.l_bound)
            / (self.outstanding_now * (2**self.k - 1))
        )
        beta = np.kron(multiplier, mantisse)

        gamma = -np.power(2.0, np.arange(-1, -ancilla_variables - 1, -1)) * np.sum(
            self.capital
        )

        qubo = np.zeros((self.n_vars, self.n_vars))
        qubo_upper_left = qubo[
            : self.number_of_assets * self.k, : self.number_of_assets * self.k
        ]
        qubo_upper_left += np.triu(2 * np.outer(beta, beta), k=1)
        np.fill_diagonal(qubo_upper_left, 2 * alpha * beta + beta**2)

        qubo_upper_right = qubo[
            : self.number_of_assets * self.k, self.number_of_assets * self.k :
        ]
        qubo_upper_right += 2 * np.outer(beta, gamma)

        qubo_lower_right = qubo[
            self.number_of_assets * self.k :, self.number_of_assets * self.k :
        ]
        qubo_lower_right += np.triu(2 * np.outer(gamma, gamma), k=1)
        np.fill_diagonal(qubo_lower_right, 2 * alpha * gamma + gamma**2)
        offset = alpha**2

        return QUBO(qubo, offset)
