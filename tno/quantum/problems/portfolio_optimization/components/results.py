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

        self.columns = [
            "outstanding amount",
            "diff ROC",
            "diff diversification",
            "diff outstanding",
        ] + [
            "diff " + constraint[0] for constraint in self.provided_emission_constraints
        ]
        self.results_df = pd.DataFrame(columns=self.columns)

    def __len__(self) -> int:
        """Return the number of samples stored in the ``Results`` object."""
        return self.results_df.shape[0]

    def add_result(self, outstanding_future: NDArray[np.float_]) -> None:
        """Add a new outstanding_future data point to results container.

        Args:
            outstanding_future: the in future outstanding amounts.
        """
        # Skip if outstanding future already in dataset.
        if any([np.array_equal(outstanding_future, a) for a in self.results_df['outstanding amount'].to_numpy()]):
            return 

        total_outstanding_future = np.sum(outstanding_future, axis=1)
        # Compute the ROC growth
        roc = np.sum(outstanding_future * self._returns, axis=1) / np.sum(
            outstanding_future * self._capital / self._outstanding_now, axis=1
        )
        roc_growth = 100 * (roc / self._roc_now - 1)

        # Compute the diversification.
        hhi = np.sum(outstanding_future**2, axis=1) / total_outstanding_future**2
        diff_diversification = 100 * (1 - (hhi / self._hhi_now))

        # Compute the growth outstanding in outstanding amount
        growth_outstanding = total_outstanding_future / self._total_outstanding_now

        new_data = [
            outstanding_future,
            roc_growth,
            diff_diversification,
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

    def head(self, n=5):
        """Return first n rows of self.results_df DataFrame

        Args:
            selected_columns: By default all columns
            n: number of results to return
        """
        selected_columns = [
            column for column in self.columns if column != "Outstanding amount"
        ]
        return self.results_df[selected_columns].head(n)

    def slice_results(
        self, tolerance: float = 0.0
    ) -> tuple[
        tuple[NDArray[np.float_], NDArray[np.float_]],
        tuple[NDArray[np.float_], NDArray[np.float_]],
    ]:
        """Slice the results in two groups, those that satisfy and those that violate
        the growth factor and emission constraints.

        Args:
            tolerance: tolerance on how strict the constraints need to be satisfied.

        Returns:
            Relative difference (diversification, roc) coordinates for solutions that
            satisfy the constraints, and for those that do not satisfy the constraints.

        Raises:
            ValueError: when there are no emission or growth factor constraints set.
        """
        if (
            self.provided_growth_target is None
            and len(self.provided_emission_constraints) == 0
        ):
            raise ValueError("There are no emission or growth constraints set.")

        mask_growth_target = (
            True
            if self.provided_growth_target is None
            else (
                self.results_df["diff outstanding"]
                >= self.provided_growth_target * (1 + tolerance)
            )
        )

        mask_emission_constraint = True
        for name, _, value in self.provided_emission_constraints:
            mask_emission_constraint &= self.results_df["diff " + name] <= value * (
                1 + tolerance
            )

        combined_mask = mask_growth_target & mask_emission_constraint

        # Filter data based on masks
        filtered_data_met = self.results_df[combined_mask]
        filtered_data_violated = self.results_df[~combined_mask]

        x_met = filtered_data_met["diff diversification"].to_numpy()
        y_met = filtered_data_met["diff ROC"].to_numpy()
        x_violated = filtered_data_violated["diff diversification"].to_numpy()
        y_violated = filtered_data_violated["diff ROC"].to_numpy()
        return (x_met, y_met), (x_violated, y_violated)
