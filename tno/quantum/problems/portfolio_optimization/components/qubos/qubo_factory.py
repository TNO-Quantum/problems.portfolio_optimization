"""This module contains the ``QuboFactory`` class."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame


class QuboFactory:
    def __init__(self, portfolio_data: DataFrame, kmin: int, kmax: int) -> None:
        self.N = len(portfolio_data)
        self.n_vars = self.N * kmax
        self.out2021 = portfolio_data["out_2021"].to_numpy()
        self.LB = portfolio_data["out_2030_min"].to_numpy()
        self.UB = portfolio_data["out_2030_max"].to_numpy()
        self.e = (portfolio_data["emis_intens_2021"].to_numpy() / 100).astype(float)
        self.income = portfolio_data["income_2021"].to_numpy()
        self.capital = portfolio_data["regcap_2021"].to_numpy()
        self.kmin = kmin
        self.kmax = kmax
        self.maxk = 2 ** (kmax + kmin) - 1 + (2 ** (-kmin) - 1) / (2 ** (-kmin))

    def calc_minimize_HHI(self) -> tuple[NDArray[np.float_], float]:
        Exp_total_out2030 = np.sum((self.UB + self.LB)) / 2

        qubo = np.zeros((self.n_vars, self.n_vars))
        offset = np.sum(self.LB**2) / Exp_total_out2030**2
        for i in range(self.N):
            multiplier = (self.UB[i] - self.LB[i]) / self.maxk
            for k in range(self.kmax):
                idx_1 = i * self.kmax + k
                value_1 = multiplier * 2 ** (k + self.kmin)
                qubo[idx_1, idx_1] = value_1 * (value_1 + 2 * self.LB[i])
                for l in range(k + 1, self.kmax):
                    idx_2 = i * self.kmax + l
                    value_2 = multiplier * 2 ** (l + self.kmin)
                    qubo[idx_1, idx_2] = 2 * value_1 * value_2

        qubo = qubo / Exp_total_out2030**2
        return qubo, offset

    def calc_emission_constraint(self) -> tuple[NDArray[np.float_], float]:
        emis2021 = np.sum(self.e * self.out2021)
        bigE = emis2021 / np.sum(self.out2021)

        alpha = (0.76 * self.e - 0.7 * bigE) * self.LB
        tmp = (0.76 * self.e - 0.7 * bigE) * (self.UB - self.LB) / self.maxk
        beta = np.array([tmp * 2 ** (k + self.kmin) for k in range(self.kmax)]).T

        qubo = np.zeros((self.n_vars, self.n_vars))

        alpha_tot = np.sum(alpha)
        offset = alpha_tot**2 / emis2021**2
        for idx1 in range(self.N * self.kmax):
            i, k = divmod(idx1, self.kmax)
            qubo[idx1, idx1] += 2 * alpha_tot * beta[i, k] + beta[i, k] ** 2
            for idx2 in range(idx1 + 1, self.N * self.kmax):
                j, l = divmod(idx2, self.kmax)
                qubo[idx1, idx2] += 2 * beta[i, k] * beta[j, l]

        qubo = qubo / emis2021**2

        return qubo, offset

    def calc_growth_factor_constraint(
        self, growth_target: float
    ) -> tuple[NDArray[np.float_], float]:
        out2021_tot = np.sum(self.out2021)
        alpha = np.sum(self.LB) / out2021_tot - growth_target

        mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = (self.UB - self.LB) / (self.maxk * out2021_tot)
        beta = np.kron(multiplier, mantisse)

        qubo = np.zeros((self.n_vars, self.n_vars))
        qubo[: self.kmax * self.N, : self.kmax * self.N] += np.triu(
            2 * np.outer(beta, beta), k=1
        )
        np.fill_diagonal(
            qubo[: self.kmax * self.N, : self.kmax * self.N],
            beta**2 + 2 * alpha * beta,
        )
        offset = alpha**2
        return qubo, offset

    def calc_maximize_ROC1(self) -> tuple[NDArray[np.float_], float]:
        Exp_avr_growth_fac = 0.5 * np.sum((self.UB + self.LB) / self.out2021)
        returns = self.income / self.out2021
        offset = np.sum(self.LB / (self.capital * self.out2021 * Exp_avr_growth_fac))
        mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = (
            (self.UB - self.LB)
            * returns
            / (self.maxk * self.capital * Exp_avr_growth_fac)
        )
        qubo_diag = -np.kron(multiplier, mantisse)

        qubo = np.diag(qubo_diag)
        return qubo, -offset

    def calc_maximize_ROC2(
        self, capital_growth_factor: float
    ) -> tuple[NDArray[np.float_], float]:
        capital_target = capital_growth_factor * np.sum(self.capital)

        mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = (self.UB - self.LB) * self.income / (self.out2021 * self.maxk)
        qubo_diag = -np.kron(multiplier, mantisse) / capital_target
        qubo = np.diag(qubo_diag)
        offset = np.sum(self.LB * self.income / self.out2021) / capital_target
        return qubo, -offset

    def calc_maximize_ROC3(self) -> tuple[NDArray[np.float_], float]:
        ancilla_qubits = self.n_vars - self.kmax * self.N
        capital2021 = np.sum(self.capital)

        alpha = np.sum(self.LB * self.income / self.out2021)
        mantisse = mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = self.income * (self.UB - self.LB) / (self.out2021 * self.maxk)
        beta = np.kron(multiplier, mantisse)

        gamma = np.power(2.0, np.arange(-1, -ancilla_qubits - 1, -1))
        gamma = gamma**2 - gamma

        qubo = np.zeros((self.n_vars, self.n_vars))
        np.fill_diagonal(qubo[: self.N * self.kmax, : self.N * self.kmax], beta)
        qubo[: self.N * self.kmax, self.N * self.kmax :] += np.outer(beta, gamma)
        np.fill_diagonal(
            qubo[self.N * self.kmax :, self.N * self.kmax :], alpha * gamma
        )
        offset = alpha / capital2021

        qubo = qubo / capital2021

        return -qubo, -offset

    def calc_maximize_ROC4(self) -> tuple[NDArray[np.float_], float]:
        returns = self.income / self.out2021
        mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = returns * (self.UB - self.LB) / self.maxk
        beta = np.kron(multiplier, mantisse)
        scaling = -2 / (np.sum((self.LB + self.UB) * self.capital / self.out2021))

        qubo = np.diag(beta) * scaling
        offset = np.sum(returns * self.LB) * scaling

        return qubo, offset

    def calc_stabilize_c1(
        self, capital_growth_factor: float
    ) -> tuple[NDArray[np.float_], float]:
        capital_target = capital_growth_factor * np.sum(self.capital)
        alpha = np.sum(self.capital * self.LB / self.out2021) - capital_target

        mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = self.capital * (self.UB - self.LB) / (self.out2021 * self.maxk)
        beta = np.kron(multiplier, mantisse)

        qubo = np.triu(2 * np.outer(beta, beta), k=1)
        np.fill_diagonal(qubo, beta**2 + 2 * alpha * beta)
        offset = alpha**2
        return qubo, offset

    def calc_stabilize_c2(self) -> tuple[NDArray[np.float_], float]:
        ancilla_qubits = self.n_vars - self.kmax * self.N
        alpha = np.sum(self.capital * self.LB / self.out2021) - np.sum(self.capital)

        mantisse = mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = self.capital * (self.UB - self.LB) / (self.out2021 * self.maxk)
        beta = np.kron(multiplier, mantisse)

        gamma = -np.power(2.0, np.arange(-1, -ancilla_qubits - 1, -1)) * np.sum(
            self.capital
        )

        qubo = np.zeros((self.n_vars, self.n_vars))
        qubo_upper_left = qubo[: self.N * self.kmax, : self.N * self.kmax]
        qubo_upper_left += np.triu(2 * np.outer(beta, beta), k=1)
        np.fill_diagonal(qubo_upper_left, 2 * alpha * beta + beta**2)

        qubo_upper_right = qubo[: self.N * self.kmax, self.N * self.kmax :]
        qubo_upper_right += 2 * np.outer(beta, gamma)

        qubo_lower_right = qubo[self.N * self.kmax :, self.N * self.kmax :]
        qubo_lower_right += np.triu(2 * np.outer(gamma, gamma), k=1)
        np.fill_diagonal(qubo_lower_right, 2 * alpha * gamma + gamma**2)
        offset = alpha**2

        return qubo, offset
