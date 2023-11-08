"""This module contains the ``QuboFactory`` class."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame


class QuboFactory:
    def __init__(self, portfolio_data: DataFrame, kmin: int, kmax: int) -> None:
        self.portfolio_data = portfolio_data
        self.N = len(portfolio_data)
        self.n_vars = self.N * kmax
        self.out_now = portfolio_data["out_now"].to_numpy()
        self.LB = portfolio_data["out_future_min"].to_numpy()
        self.UB = portfolio_data["out_future_max"].to_numpy()
        self.income = portfolio_data["income_now"].to_numpy()
        self.capital = portfolio_data["regcap_now"].to_numpy()
        self.kmin = kmin
        self.kmax = kmax
        self.maxk = 2 ** (kmax + kmin) - 1 + (2 ** (-kmin) - 1) / (2 ** (-kmin))

    def calc_minimize_HHI(self) -> tuple[NDArray[np.float_], float]:
        r"""$\frac{\sum_i\left(LB_i + \frac{UB_i-LB_i}{maxk}\sum_k2^kx_{ik}\right)^2}{\left(\frac{1}{2}\sum_iUB_i-LB_i\right)^2}$"""
        Exp_total_out_future = np.sum((self.UB + self.LB)) / 2

        qubo = np.zeros((self.n_vars, self.n_vars))
        offset = np.sum(self.LB**2) / Exp_total_out_future**2
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

        qubo = qubo / Exp_total_out_future**2
        return qubo, offset

    def calc_emission_constraint(self) -> tuple[NDArray[np.float_], float]:
        r"""$\frac{\left(\sum_i(efuture_i-0.7\frac{enow_j*out_now_j}{\sum_j out_now_j}(LB_i+\frac{UB_i-LB_i}{maxk}\sum_k 2^kx_{ik}))\right)^2}{\left(\sum_i enow_i*out_now_i\right)^2}$"""
        e_intens_now = self.portfolio_data["emis_intens_now"].to_numpy()
        e_intens_future = self.portfolio_data["emis_intens_future"].to_numpy()

        emisnow = np.sum(e_intens_now * self.out_now)
        bigE = emisnow / np.sum(self.out_now)

        alpha = np.sum((e_intens_future - 0.7 * bigE) * self.LB)

        mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = (e_intens_future - 0.7 * bigE) * (self.UB - self.LB) / self.maxk
        beta = np.kron(multiplier, mantisse)

        qubo = np.zeros((self.n_vars, self.n_vars))

        offset = alpha**2 / emisnow**2
        for idx1 in range(self.N * self.kmax):
            qubo[idx1, idx1] += 2 * alpha * beta[idx1] + beta[idx1] ** 2
            for idx2 in range(idx1 + 1, self.N * self.kmax):
                qubo[idx1, idx2] += 2 * beta[idx1] * beta[idx2]

        qubo = qubo / emisnow**2

        return qubo, offset

    def calc_growth_factor_constraint(
        self, growth_target: float
    ) -> tuple[NDArray[np.float_], float]:
        r"""$\left(\frac{\sum_i LB_i + \frac{UB_i-LB_i}{maxk}\sum_k x_{ik}}{\sum_i out_now_i} - growth\_factor\right)^2$"""
        out_now_tot = np.sum(self.out_now)
        alpha = np.sum(self.LB) / out_now_tot - growth_target

        mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = (self.UB - self.LB) / (self.maxk * out_now_tot)
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
        r"""$-\left(\sum_i\frac{2*out_now_i}{UB_i+LB_i}\right)\sum_i\frac{income_i}{capital_i*out_now_i}\left(\sum_i LB_i + (UB_i-LB_i)\sum_k2^kx_{ik}\right)$"""
        Exp_avr_growth_fac = 0.5 * np.sum((self.UB + self.LB) / self.out_now)
        returns = self.income / self.out_now
        offset = np.sum(self.LB / (self.capital * self.out_now * Exp_avr_growth_fac))
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
        r"""$\sum_i\frac{income_i}{out_now_i}\left(LB_i+\frac{UB_i-LB_i}{maxk}\sum_k2^kx_{ik}\right)$"""
        capital_target = capital_growth_factor * np.sum(self.capital)

        mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = (self.UB - self.LB) * self.income / (self.out_now * self.maxk)
        qubo_diag = -np.kron(multiplier, mantisse) / capital_target
        qubo = np.diag(qubo_diag)
        offset = np.sum(self.LB * self.income / self.out_now) / capital_target
        return qubo, -offset

    def calc_maximize_ROC3(self) -> tuple[NDArray[np.float_], float]:
        ancilla_qubits = self.n_vars - self.kmax * self.N
        capitalnow = np.sum(self.capital)

        alpha = np.sum(self.LB * self.income / self.out_now)
        mantisse = mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = self.income * (self.UB - self.LB) / (self.out_now * self.maxk)
        beta = np.kron(multiplier, mantisse)

        gamma = np.power(2.0, np.arange(-1, -ancilla_qubits - 1, -1))
        gamma = gamma**2 - gamma

        qubo = np.zeros((self.n_vars, self.n_vars))
        np.fill_diagonal(qubo[: self.N * self.kmax, : self.N * self.kmax], beta)
        qubo[: self.N * self.kmax, self.N * self.kmax :] += np.outer(beta, gamma)
        np.fill_diagonal(
            qubo[self.N * self.kmax :, self.N * self.kmax :], alpha * gamma
        )
        offset = alpha / capitalnow

        qubo = qubo / capitalnow

        return -qubo, -offset

    def calc_maximize_ROC4(self) -> tuple[NDArray[np.float_], float]:
        r"""$\frac{\sum_i\frac{income_i}{out_now_i}\left(LB_i+\frac{UB_i-LB_i}{maxk}\sum_k 2^k x_{ik}\right)}{\sum_i \frac{LB_i+UB_i}{2out_now_i}*capital_i}$"""
        returns = self.income / self.out_now
        mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = returns * (self.UB - self.LB) / self.maxk
        beta = np.kron(multiplier, mantisse)
        scaling = -2 / (np.sum((self.LB + self.UB) * self.capital / self.out_now))

        qubo = np.diag(beta) * scaling
        offset = np.sum(returns * self.LB) * scaling

        return qubo, offset

    def calc_stabilize_c1(
        self, capital_growth_factor: float
    ) -> tuple[NDArray[np.float_], float]:
        capital_target = capital_growth_factor * np.sum(self.capital)
        alpha = np.sum(self.capital * self.LB / self.out_now) - capital_target

        mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = self.capital * (self.UB - self.LB) / (self.out_now * self.maxk)
        beta = np.kron(multiplier, mantisse)

        qubo = np.triu(2 * np.outer(beta, beta), k=1)
        np.fill_diagonal(qubo, beta**2 + 2 * alpha * beta)
        offset = alpha**2
        return qubo, offset

    def calc_stabilize_c2(self) -> tuple[NDArray[np.float_], float]:
        ancilla_qubits = self.n_vars - self.kmax * self.N
        alpha = np.sum(self.capital * self.LB / self.out_now) - np.sum(self.capital)

        mantisse = mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = self.capital * (self.UB - self.LB) / (self.out_now * self.maxk)
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
