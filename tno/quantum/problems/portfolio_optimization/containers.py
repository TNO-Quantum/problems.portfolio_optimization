from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame


class Results:
    def __init__(
        self, portfolio_data: DataFrame, Growth_target: Optional[float] = None
    ) -> None:
        self._out2021 = portfolio_data["out_2021"].to_numpy()
        self._e = (portfolio_data["emis_intens_2021"].to_numpy() / 100).astype(float)
        income = portfolio_data["income_2021"].to_numpy()
        self._capital = portfolio_data["regcap_2021"].to_numpy()
        self._returns = income / self._out2021
        self._ROC2021 = np.sum(income) / np.sum(self._capital)
        self._HHI2021 = np.sum(self._out2021**2) / np.sum(self._out2021) ** 2
        self._Growth_target = Growth_target
        self._bigE = np.sum(self._e * self._out2021) / np.sum(self._out2021)
        self._Out2021 = np.sum(self._out2021)

        # These are the variables to store 3 kinds of results.
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

        if self._Growth_target is None:
            res_emis = 0.76 * np.sum(self._e * out2030, axis=1)
            norm1 = self._bigE * 0.70 * Out2030
            norm2 = 1.020 * norm1
            discriminator1 = res_emis < norm1
            discriminator2 = res_emis < norm2
        else:
            Realized_growth = Out2030 / self._Out2021
            discriminator1 = Realized_growth > self._Growth_target
            discriminator2 = Realized_growth > 0.98 * self._Growth_target

        slice1 = discriminator1
        slice2 = ~slice1 & (discriminator2)
        slice3 = ~(slice1 | slice2)
        self.x1.extend(x[slice1])
        self.y1.extend(y[slice1])
        self.x2.extend(x[slice2])
        self.y2.extend(y[slice2])
        self.x3.extend(x[slice3])
        self.y3.extend(y[slice3])
