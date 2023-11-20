"""This module contains a container for Results object."""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tno.quantum.problems.portfolio_optimization.components.io import PortfolioData


class Results:
    """Results container"""

    def __init__(
        self,
        portfolio_data: PortfolioData,
        provided_emission_constraints: list[tuple(str, str, float)] = [],
        provided_growth_target: Optional[float] = None,
    ) -> None:
        """Init of Results container.

        Args:
            portfolio_data: the portfolio data
            provided_emission_constraints: list of all the emission constraints that are
                provided. Each list element contains the ``variable_now``,
                ``variable_future`` and ``reduction_percentage_target`` input.
            provided_growth_target: target outstanding amount growth factor if the
                growth factor constraint is set, otherwise None.
        """
        self.portfolio_data = portfolio_data
        self.provided_emission_constraints = provided_emission_constraints
        self.provided_growth_target = provided_growth_target

        self._outstanding_now = portfolio_data.get_outstanding_now()
        self._total_outstanding_now = np.sum(self._outstanding_now)

        self._returns = portfolio_data.get_income() / self._outstanding_now
        self._capital = portfolio_data.get_capital()
        self._roc_now = np.sum(portfolio_data.get_income()) / np.sum(self._capital)
        self._hhi_now = (
            np.sum(self._outstanding_now**2) / np.sum(self._outstanding_now) ** 2
        )

        self.columns = ["Outstanding amount", "ROC", "HHI", "Outstanding growth",] + [
            constraint[0] + " growth"
            for constraint in self.provided_emission_constraints
        ]
        self.results_df = pd.DataFrame(columns=self.columns)

        self._x: deque[NDArray[np.float_]] = deque()
        self._y: deque[NDArray[np.float_]] = deque()
        self._outstanding_future: deque[NDArray[np.float_]] = deque()

    def __len__(self) -> int:
        """Return the number of samples stored in the ``Results`` object."""
        return len(self._x)

    def add_result(self, outstanding_future: NDArray[np.float_]) -> None:
        """Add a new outstanding_future data point to results container.

        Args:
            outstanding_future: the in future outstanding amounts.
        """
        total_outstanding_future = np.sum(outstanding_future, axis=1)

        # Compute the ROC growth
        roc = np.sum(outstanding_future * self._returns, axis=1) / np.sum(
            outstanding_future * self._capital / self._outstanding_now, axis=1
        )
        roc_growth = 100 * (roc / self._roc_now - 1)

        # Compute the diversification.
        hhi = np.sum(outstanding_future**2, axis=1) / total_outstanding_future**2
        diversification = 100 * (1 - (hhi / self._hhi_now))

        # Compute the growth outstanding in outstanding amount
        growth_outstanding = total_outstanding_future / self._total_outstanding_now

        new_data = [
            outstanding_future,
            roc,
            hhi,
            growth_outstanding,
        ]

        # Compute the emission constraint growths
        for (
            column_name_now,
            column_name_future,
            _,
        ) in self.provided_emission_constraints:
            total_emission_now = np.sum(
                self._outstanding_now * self.portfolio_data.get_column(column_name_now)
            )
            total_emission_future = np.sum(
                outstanding_future * self.portfolio_data.get_column(column_name_future)
            )

            new_data.append(total_emission_future / total_emission_now)
        
        # Write results
        self.results_df.loc[len(self.results_df)] = new_data
        self._x.extend(diversification)
        self._y.extend(roc_growth)
        self._outstanding_future.extend(outstanding_future)

    def head(self, n=5):
        """Return first n rows of self.results_df DataFrame

        Args:
            n: number of results to return
        """
        return self.results_df.head(n)

    def aggregate(self) -> None:
        """Aggregate unique results."""
        diversification = np.asarray(self._x)
        roc_growth = np.asarray(self._y)
        outstanding_future = np.asarray(self._outstanding_future)

        data = np.vstack((np.asarray(self._x), np.asarray(self._y)))
        _, indices = np.unique(data, axis=1, return_index=True)

        self._x = deque(diversification[indices])
        self._y = deque(roc_growth[indices])
        self._outstanding_future = deque(outstanding_future[indices])

    def slice_results(
        self, growth_target: Optional[float] = None
    ) -> tuple[
        tuple[NDArray[np.float_], NDArray[np.float_]],
        tuple[NDArray[np.float_], NDArray[np.float_]],
        tuple[NDArray[np.float_], NDArray[np.float_]],
    ]:
        """Slice the results in three groups, growth targets met, almost met, not met or
        not.

            - Realized growth > growth target
            - 98% of the growth target < Realized growth < growth target
            - Realized growth < 98% of the growth target

        Args:
            growth_target: the target to

        #TODO: Handle growth_target is None docs, is quite specific/hardcoded
        """
        diversification = np.asarray(self._x)
        roc_growth = np.asarray(self._y)
        outstanding_future = np.array(self._outstanding_future)
        total_outstanding_future = np.sum(outstanding_future, axis=1)

        if growth_target is None:
            res_emis = 0.76 * np.sum(self._e * self._outstanding_future, axis=1)
            # norm1 = (
            #     self._relelative_total_emission_now * 0.70 * total_outstanding_future
            # )
            norm1 = 10 # TODO TEMP BREAKING 
            norm2 = 1.020 * norm1
            discriminator1 = res_emis < norm1
            discriminator2 = res_emis < norm2
        else:
            realized_growth = total_outstanding_future / self._total_outstanding_now
            discriminator1 = realized_growth > growth_target
            discriminator2 = realized_growth > 0.98 * growth_target

        mask1 = discriminator1
        mask2 = ~mask1 & (discriminator2)
        mask3 = ~(mask1 | mask2)

        x_met = diversification[mask1]
        y_met = roc_growth[mask1]
        x_reduced = diversification[mask2]
        y_reduced = roc_growth[mask2]
        x_violated = diversification[mask3]
        y_violated = roc_growth[mask3]

        return (x_met, y_met), (x_reduced, y_reduced), (x_violated, y_violated)
