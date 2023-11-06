from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame


class Results:
    def __init__(self, portfolio_data: DataFrame) -> None:
        self._out2021 = portfolio_data["out_2021"].to_numpy()
        self._e = (portfolio_data["emis_intens_2021"].to_numpy() / 100).astype(float)
        income = portfolio_data["income_2021"].to_numpy()
        self._capital = portfolio_data["regcap_2021"].to_numpy()
        self._returns = income / self._out2021
        self._ROC2021 = np.sum(income) / np.sum(self._capital)
        self._HHI2021 = np.sum(self._out2021**2) / np.sum(self._out2021) ** 2
        self._bigE = np.sum(self._e * self._out2021) / np.sum(self._out2021)
        self._Out2021 = np.sum(self._out2021)

        # These are the variables to store 3 kinds of results.
        self._x = deque()
        self._y = deque()
        self._out2030 = deque()
        self.x1 = deque()
        self.y1 = deque()  # Emission target met
        self.x2 = deque()
        self.y2 = deque()  # Reduced emission
        self.x3 = deque()
        self.y3 = deque()  # Targets not met

    def add_result(self, out2030: NDArray[np.float_]) -> None:
        Out2030 = np.sum(out2030, axis=1)
        # Compute the 2030 HHI.
        HHI2030 = np.sum(out2030**2, axis=1) / Out2030**2
        # Compute the 2030 ROC
        ROC = np.sum(out2030 * self._returns, axis=1) / np.sum(
            out2030 * self._capital / self._out2021, axis=1
        )
        # Compute the emissions from the resulting 2030 portfolio.
        x = 100 * (1 - (HHI2030 / self._HHI2021))
        y = 100 * (ROC / self._ROC2021 - 1)

        self._x.extend(x)
        self._y.extend(y)
        self._out2030.extend(out2030)

    def aggregate(self) -> None:
        x = np.asarray(self._x)
        y = np.asarray(self._y)
        out_2030 = np.asarray(self._out2030)

        data = np.vstack((np.asarray(self._x), np.asarray(self._y)))
        _, indices = np.unique(data, axis=1, return_index=True)

        self._x = x[indices]
        self._y = y[indices]
        self._out2030 = out_2030[indices]

    def slice_results(self, growth_target: Optional[float] = None) -> None:
        x = np.asarray(self._x)
        y = np.asarray(self._y)
        out2030 = np.array(self._out2030)
        Out2030 = np.sum(out2030, axis=1)

        if growth_target is None:
            res_emis = 0.76 * np.sum(self._e * self._out2030, axis=1)
            norm1 = self._bigE * 0.70 * Out2030
            norm2 = 1.020 * norm1
            discriminator1 = res_emis < norm1
            discriminator2 = res_emis < norm2
        else:
            Realized_growth = Out2030 / self._Out2021
            discriminator1 = Realized_growth > self._Growth_target
            discriminator2 = Realized_growth > 0.98 * self._Growth_target

        mask1 = discriminator1
        mask2 = ~mask1 & (discriminator2)
        mask3 = ~(mask1 | mask2)

        self.x1 = x[mask1]
        self.y1 = y[mask1]
        self.x2 = x[mask2]
        self.y2 = y[mask2]
        self.x3 = x[mask3]
        self.y3 = y[mask3]
