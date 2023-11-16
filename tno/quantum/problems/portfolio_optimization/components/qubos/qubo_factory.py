"""This module contains the ``QuboFactory`` class."""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame


class QuboFactory:
    """
    QuboFactory - A factory class for creating QUBO instances.

    This class provides a convenient interface for constructing intermediate QUBO
    matrices for different objectives and constraints.

    Methods:

    - `calc_minimize_HHI`: Calculate the to minimize HHI QUBO
    - `calc_maximize_ROC1`: Calculate the to maximize return on capital QUBO variant 1
    - `calc_maximize_ROC2`: Calculate the to maximize return on capital QUBO variant 2
    - `calc_maximize_ROC3`: Calculate the to maximize return on capital QUBO variant 3
    - `calc_maximize_ROC4`: Calculate the to maximize return on capital QUBO variant 4
    - `calc_emission_constraint`:
    - `calc_growth_factor_constraint`: Calculate the growth factor constraint QUBO
    - `calc_stabilize_c1`:
    - `calc_stabilize_c2`:

    """

    def __init__(self, portfolio_data: DataFrame, k: int) -> None:
        """

        Args:
            portfolio_data: A ``pandas.Dataframe`` containing the portfolio to optimize.
            k: The number of bits that are used to represent the outstanding amount for
                each asset. A fixed point representation is used to represent `$2^k$`
                different equidistant values in the range `$[LB_i, UB_i]$` for asset i.

        """
        self.portfolio_data = portfolio_data
        self.number_of_assets = len(portfolio_data)
        self.n_vars = self.number_of_assets * k
        self.outstanding_now = portfolio_data["outstanding_now"].to_numpy()
        self.LB = portfolio_data["min_outstanding_future"].to_numpy()
        self.UB = portfolio_data["max_outstanding_future"].to_numpy()
        self.income = portfolio_data["income_now"].to_numpy()
        self.capital = portfolio_data["regcap_now"].to_numpy()
        self.k = k

    def calc_minimize_HHI(self) -> tuple[NDArray[np.float_], float]:
        r"""Calculate the to minimize HHI QUBO.

        The QUBO formulation is given by

        $$QUBO = \frac{\sum_i\left(LB_i + \frac{UB_i-LB_i}{2^k-1}\sum_j2^jx_{i,j}\right)^2}{\left(\frac{1}{2}\sum_iUB_i+LB_i\right)^2}$$

        where:

            - `$LB_i$` is the lower bound for asset `$i$`,
            - `$UB_i$` is the upper bound for asset `$i$`,
            - `$k$` is the number of bits,
            - and `$x_{i,j}$` are the $j$ binary variables for asset `$i$` with $j<k$.

        Returns:
            qubo matrix and its offset
        """
        expected_total_outstanding_future = np.sum((self.UB + self.LB)) / 2

        qubo = np.zeros((self.n_vars, self.n_vars))
        offset = np.sum(self.LB**2) / expected_total_outstanding_future**2
        for asset_i in range(self.number_of_assets):
            multiplier = (self.UB[asset_i] - self.LB[asset_i]) / (2**self.k - 1)

            # For asset i: (LB + multiplier * sum_j 2**j x_j) ** 2
            for bit_j in range(self.k):
                idx_1 = asset_i * self.k + bit_j

                # Diagonal elements: 2 * LB * multiplier * sum_j 2**j x_j
                qubo[idx_1, idx_1] += 2 * self.LB[asset_i] * multiplier * 2**bit_j

                # Elements: multiplier**2 * (sum_j 2**j x_j) * (sum_j' 2**j' x_j')
                for bit_j_prime in range(self.k):
                    idx_2 = asset_i * self.k + bit_j_prime
                    qubo[idx_1, idx_2] += (
                        multiplier**2 * (2**bit_j) * (2**bit_j_prime)
                    )

        qubo = qubo / expected_total_outstanding_future**2
        return qubo, offset

    def calc_emission_constraint(
        self,
        column_name_now: str,
        column_name_future: Optional[str] = None,
        reduction_percentage_target: float = 0.7,
    ) -> tuple[NDArray[np.float_], float]:
        r"""Calculate the emission constraint QUBO for arbitrary target reduction target

        The QUBO formulation is given by

        $$QUBO = \left(\frac{\sum_i f_i \left(LB_i + \frac{UB_i-LB_i}{2^k-1}\sum_j2^jx_{i,j}\right)}{{\frac{1}{2}\sum_iUB_i+LB_i}} - g \frac{\sum_i e_i \cdot out_i}{\sum_i out_i} \right)^2$$

        where:

            - `$LB_i$` is the lower bound for asset `$i$`,
            - `$UB_i$` is the upper bound for asset `$i$`,
            - `$k$` is the number of bits,
            - `$e_i$` is the current emission intensity for asset `$i$`,
            - `$f_i$` is the expected emission intensity at the future for asset `$i$`,
            - `$out_i$` is the current outstanding amount for asset `$i$`,
            - `$g$` is the target value for the relative emission reduction,
            - and `$x_{i,j}$` are the $j$ binary variables for asset `$i$` with $j<k$.

        Args:
            variable_now: Name of the column in the portfolio dataset corresponding to
                the variables at current time.
            variable_future: Name of the column in the portfolio dataset corresponding
                to the variables at future time. If no value is provided, it is assumed
                that the value is constant over time, i.e., the variable
                ``variable_now`` will be used.
            reduction_percentage_target: target value for reduction percentage amount.

        Raises:
            KeyError: if the provided column names are not in the portfolio_data.

        Returns:
            qubo matrix and its offset
        """
        if column_name_future is None:
            column_name_future = column_name_now

        if column_name_now not in self.portfolio_data:
            raise KeyError(
                f"Column name {column_name_now} not present in portfolio dataset."
            )
        if column_name_future not in self.portfolio_data:
            raise KeyError(
                f"Column name {column_name_future} not present in portfolio dataset."
            )

        emission_intensity_now = self.portfolio_data[column_name_now].to_numpy()
        emission_intensity_future = self.portfolio_data[column_name_future].to_numpy()

        total_emission_now = np.sum(emission_intensity_now * self.outstanding_now)
        rel_total_emission_now = total_emission_now / np.sum(self.outstanding_now)

        alpha = np.sum(
            (
                emission_intensity_future
                - reduction_percentage_target * rel_total_emission_now
            )
            * self.LB
        )

        mantisse = np.power(2, np.arange(self.k))
        multiplier = (
            (
                emission_intensity_future
                - reduction_percentage_target * rel_total_emission_now
            )
            * (self.UB - self.LB)
            / (2**self.k - 1)
        )
        beta = np.kron(multiplier, mantisse)

        qubo = np.zeros((self.n_vars, self.n_vars))

        offset = alpha**2 / emission_intensity_now**2
        for idx1 in range(self.number_of_assets * self.k):
            qubo[idx1, idx1] += 2 * alpha * beta[idx1] + beta[idx1] ** 2
            for idx2 in range(idx1 + 1, self.number_of_assets * self.k):
                qubo[idx1, idx2] += 2 * beta[idx1] * beta[idx2]

        qubo = qubo / emission_intensity_now**2

        return qubo, offset

    def calc_growth_factor_constraint(
        self, growth_target: float
    ) -> tuple[NDArray[np.float_], float]:
        r"""Calculate the growth factor constraint QUBO

        The QUBO formulation is given by

        $$QUBO = \left(\frac{\sum_i LB_i + \frac{UB_i-LB_i}{2^k-1}\sum_j 2^jx_{i,j}}{\sum_i out_i} - g\right)^2$$

        where:

            - `$LB_i$` is the lower bound for asset `$i$`,
            - `$UB_i$` is the upper bound for asset `$i$`,
            - `$k$` is the number of bits,
            - `$g$` is the target value for the total growth factor,
            - `$out_i$` is the current outstanding amount for asset `$i$`,
            - and `$x_{i,j}$` are the $j$ binary variables for asset `$i$` with $j<k$.

        Args:
            growth_target: target value for growth factor total outstanding amount.

        Returns:
            qubo matrix and its offset
        """
        total_outstanding_now = np.sum(self.outstanding_now)
        alpha = np.sum(self.LB) / total_outstanding_now - growth_target

        qubo = np.zeros((self.n_vars, self.n_vars))
        offset = alpha**2
        for asset_i in range(self.number_of_assets):
            multiplier = (self.UB[asset_i] - self.LB[asset_i]) / (
                2**self.k * total_outstanding_now
            )

            # For asset i: (alpha + multiplier * sum_j 2**j x_j) ** 2
            for bit_j in range(self.k):
                idx_1 = asset_i * self.k + bit_j

                # Diagonal elements: 2 * alpha * multiplier * sum_j 2**j x_j
                qubo[idx_1, idx_1] += 2 * alpha * multiplier * 2**bit_j

                # Elements: multiplier**2 * (sum_j 2**j x_j) * (sum_j' 2**j' x_j')
                for bit_j_prime in range(self.k):
                    idx_2 = asset_i * self.k + bit_j_prime
                    qubo[idx_1, idx_2] += (
                        multiplier**2 * (2**bit_j) * (2**bit_j_prime)
                    )

        return qubo, offset

    def calc_maximize_ROC1(self) -> tuple[NDArray[np.float_], float]:
        r"""$-\left(\sum_i\frac{2*out_now_i}{UB_i+LB_i}\right)\sum_i\frac{income_i}{capital_i*out_now_i}\left(\sum_i LB_i + (UB_i-LB_i)\sum_k2^kx_{ik}\right)$"""
        expected_average_growth_factor = 0.5 * np.sum(
            (self.UB + self.LB) / self.outstanding_now
        )
        returns = self.income / self.outstanding_now
        offset = np.sum(
            self.LB
            / (self.capital * self.outstanding_now * expected_average_growth_factor)
        )
        mantisse = np.power(2, np.arange(self.k))
        multiplier = (
            (self.UB - self.LB)
            * returns
            / ((2**self.k - 1) * self.capital * expected_average_growth_factor)
        )
        qubo_diag = -np.kron(multiplier, mantisse)

        qubo = np.diag(qubo_diag)
        return qubo, -offset

    def calc_maximize_ROC2(
        self, capital_growth_factor: float
    ) -> tuple[NDArray[np.float_], float]:
        r"""$\sum_i\frac{income_i}{out_now_i}\left(LB_i+\frac{UB_i-LB_i}{2**k - 1}\sum_k2^kx_{ik}\right)$"""
        capital_target = capital_growth_factor * np.sum(self.capital)

        mantisse = np.power(2, np.arange(self.k))
        multiplier = (
            (self.UB - self.LB)
            * self.income
            / (self.outstanding_now * (2**self.k - 1))
        )
        qubo_diag = -np.kron(multiplier, mantisse) / capital_target
        qubo = np.diag(qubo_diag)
        offset = np.sum(self.LB * self.income / self.outstanding_now) / capital_target
        return qubo, -offset

    def calc_maximize_ROC3(self) -> tuple[NDArray[np.float_], float]:
        ancilla_qubits = self.n_vars - self.k * self.number_of_assets
        capital_now = np.sum(self.capital)

        alpha = np.sum(self.LB * self.income / self.outstanding_now)
        mantisse = mantisse = np.power(2, np.arange(self.k))
        multiplier = (
            self.income
            * (self.UB - self.LB)
            / (self.outstanding_now * (2**self.k - 1))
        )
        beta = np.kron(multiplier, mantisse)

        gamma = np.power(2.0, np.arange(-1, -ancilla_qubits - 1, -1))
        gamma = gamma**2 - gamma

        qubo = np.zeros((self.n_vars, self.n_vars))
        np.fill_diagonal(
            qubo[: self.number_of_assets * self.k, : self.number_of_assets * self.k],
            beta,
        )
        qubo[
            : self.number_of_assets * self.k, self.number_of_assets * self.k :
        ] += np.outer(beta, gamma)
        np.fill_diagonal(
            qubo[self.number_of_assets * self.k :, self.number_of_assets * self.k :],
            alpha * gamma,
        )
        offset = alpha / capital_now

        qubo = qubo / capital_now

        return -qubo, -offset

    def calc_maximize_ROC4(self) -> tuple[NDArray[np.float_], float]:
        r"""$\frac{\sum_i\frac{income_i}{out_now_i}\left(LB_i+\frac{UB_i-LB_i}{2**k - 1}\sum_k 2^k x_{ik}\right)}{\sum_i \frac{LB_i+UB_i}{2out_now_i}*capital_i}$"""
        returns = self.income / self.outstanding_now
        mantisse = np.power(2, np.arange(self.k))
        multiplier = returns * (self.UB - self.LB) / (2**self.k - 1)
        beta = np.kron(multiplier, mantisse)
        scaling = -2 / (
            np.sum((self.LB + self.UB) * self.capital / self.outstanding_now)
        )

        qubo = np.diag(beta) * scaling
        offset = np.sum(returns * self.LB) * scaling

        return qubo, offset

    def calc_stabilize_c1(
        self, capital_growth_factor: float
    ) -> tuple[NDArray[np.float_], float]:
        capital_target = capital_growth_factor * np.sum(self.capital)
        alpha = np.sum(self.capital * self.LB / self.outstanding_now) - capital_target

        mantisse = np.power(2, np.arange(self.k))
        multiplier = (
            self.capital
            * (self.UB - self.LB)
            / (self.outstanding_now * (2**self.k - 1))
        )
        beta = np.kron(multiplier, mantisse)

        qubo = np.triu(2 * np.outer(beta, beta), k=1)
        np.fill_diagonal(qubo, beta**2 + 2 * alpha * beta)
        offset = alpha**2
        return qubo, offset

    def calc_stabilize_c2(self) -> tuple[NDArray[np.float_], float]:
        ancilla_qubits = self.n_vars - self.k * self.number_of_assets
        alpha = np.sum(self.capital * self.LB / self.outstanding_now) - np.sum(
            self.capital
        )

        mantisse = mantisse = np.power(2, np.arange(self.k))
        multiplier = (
            self.capital
            * (self.UB - self.LB)
            / (self.outstanding_now * (2**self.k - 1))
        )
        beta = np.kron(multiplier, mantisse)

        gamma = -np.power(2.0, np.arange(-1, -ancilla_qubits - 1, -1)) * np.sum(
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

        return qubo, offset
