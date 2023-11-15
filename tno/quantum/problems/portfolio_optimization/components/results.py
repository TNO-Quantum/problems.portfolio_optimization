"""This module contains a container for Results object."""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame


class Results:
    """Results container"""

    def __init__(self, portfolio_data: DataFrame) -> None:
        """Init of Results container.

        Args:
            portfolio_data: the portfolio data
        """
        self._out_now = portfolio_data["out_now"].to_numpy()
        self._e = portfolio_data["emis_intens_now"].to_numpy()
        income = portfolio_data["income_now"].to_numpy()
        self._capital = portfolio_data["regcap_now"].to_numpy()
        self._returns = income / self._out_now
        self._ROC_now = np.sum(income) / np.sum(self._capital)
        self._HHI_now = np.sum(self._out_now**2) / np.sum(self._out_now) ** 2
        self._bigE = np.sum(self._e * self._out_now) / np.sum(self._out_now)
        self._Out_now = np.sum(self._out_now)

        self._x: deque[NDArray[np.float_]] = deque()
        self._y: deque[NDArray[np.float_]] = deque()
        self._out_future: deque[NDArray[np.float_]] = deque()

    def __len__(self) -> int:
        """Return the number of samples stored in the ``Results`` object."""
        return len(self._x)

    def add_result(self, out_future: NDArray[np.float_]) -> None:
        """Add a new out_future data point to results container.

        Args:
            out_future: ...
        """
        Out_future = np.sum(out_future, axis=1)
        # Compute the future HHI.
        HHI_future = np.sum(out_future**2, axis=1) / Out_future**2
        # Compute the future ROC
        ROC = np.sum(out_future * self._returns, axis=1) / np.sum(
            out_future * self._capital / self._out_now, axis=1
        )
        # Compute the emissions from the resulting future portfolio.
        x = 100 * (1 - (HHI_future / self._HHI_now))
        y = 100 * (ROC / self._ROC_now - 1)

        self._x.extend(x)
        self._y.extend(y)
        self._out_future.extend(out_future)

    def aggregate(self) -> None:
        """Aggregate unique results."""
        x = np.asarray(self._x)
        y = np.asarray(self._y)
        out_future = np.asarray(self._out_future)

        data = np.vstack((np.asarray(self._x), np.asarray(self._y)))
        _, indices = np.unique(data, axis=1, return_index=True)

        self._x = deque(x[indices])
        self._y = deque(y[indices])
        self._out_future = deque(out_future[indices])

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
        x = np.asarray(self._x)
        y = np.asarray(self._y)
        out_future = np.array(self._out_future)
        Out_future = np.sum(out_future, axis=1)

        if growth_target is None:
            res_emis = 0.76 * np.sum(self._e * self._out_future, axis=1)
            norm1 = self._bigE * 0.70 * Out_future
            norm2 = 1.020 * norm1
            discriminator1 = res_emis < norm1
            discriminator2 = res_emis < norm2
        else:
            Realized_growth = Out_future / self._Out_now
            discriminator1 = Realized_growth > growth_target
            discriminator2 = Realized_growth > 0.98 * growth_target

        mask1 = discriminator1
        mask2 = ~mask1 & (discriminator2)
        mask3 = ~(mask1 | mask2)

        x1 = x[mask1]
        y1 = y[mask1]
        x2 = x[mask2]
        y2 = y[mask2]
        x3 = x[mask3]
        y3 = y[mask3]

        return (x1, y1), (x2, y2), (x3, y3)
