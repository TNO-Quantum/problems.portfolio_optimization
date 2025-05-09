"""This module implements I/O."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

DEFAULT_REQUIRED_COLUMN_NAMES = [
    "asset",
    "outstanding_now",
    "min_outstanding_future",
    "max_outstanding_future",
    "income_now",
    "regcap_now",
]

PortfolioDataT = TypeVar("PortfolioDataT", bound="PortfolioData")


class PortfolioData:
    """The :class:`~PortfolioData` stores the data used for portfolio optimization.

    The example below shows how to load and print info for a benchmark dataset.

    >>> from tno.quantum.problems.portfolio_optimization import PortfolioData
    >>> portfolio_data = PortfolioData.from_file("benchmark_dataset")
    >>> portfolio_data.print_portfolio_info()
    PortfolioData: --------- Portfolio information -----------
    PortfolioData: Total outstanding now: 21252.70
    PortfolioData: ROC now: 1.0642
    PortfolioData: HHI now: 1.0000
    PortfolioData: Total Emission now: 43355508.00
    PortfolioData: Relative emission intensity now: 2040.00
    PortfolioData: Expected total outstanding future: 31368.00
    PortfolioData: Std dev: 886.39
    PortfolioData: Expected average growth factor: 1.4760
    PortfolioData: Std dev: 0.0417
    PortfolioData: --------- --------------------- -----------
    """

    def __init__(
        self,
        portfolio_dataframe: DataFrame,
        columns_rename: dict[str, str] | None = None,
    ) -> None:
        """Creates a :class:`~PortfolioData` object from a pandas :class:`~pandas.DataFrame`.

        The portfolio data is expected to contain at least the following columns names:

            - ``"assets"``: The name of the asset.
            - ``"outstanding_now_now"``: Current outstanding amount per asset.
            - ``"min_outstanding_future"``: Lower bound outstanding amount in the future per asset.
            - ``"max_outstanding_future"``: Upper bound outstanding amount in the future per asset.
            - ``"income_now"``: Current income per asset, corresponds to return multiplied by the current outstanding amount.
            - ``"regcap_now"``: Current regulatory capital per asset.

        Different column names in the dataset can be used, but in that case they need to
        be provided as a renaming dictionary to the ``columns_rename`` argument.

        Args:
            portfolio_dataframe: :class:`~pandas.DataFrame` containing the portfolio data.
            columns_rename: to rename columns provided as dict with new column names as keys
                and to replace column name as value. Example
                ``{"outstanding_2021": "outstanding_now"}``.

        Raises:
            ValueError if required columns are not present in dataset.
        """  # noqa: E501
        if columns_rename is not None:
            portfolio_dataframe.rename(columns=columns_rename, inplace=True)  # noqa: PD002

        # Validate dataset to contain required column names
        for required_column_name in DEFAULT_REQUIRED_COLUMN_NAMES:
            if required_column_name not in portfolio_dataframe.columns:
                error_msg = (
                    f"Required column name {required_column_name} is not in dataset."
                )
                raise ValueError(error_msg)

        self.portfolio_df = portfolio_dataframe

        # Create logger
        self._logger = logging.getLogger("PortfolioData")
        self._logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        self._logger.addHandler(handler)

    @classmethod
    def from_file(
        cls: type[PortfolioDataT],
        filename: str | Path,
        columns_rename: dict[str, str] | None = None,
    ) -> PortfolioDataT:
        """Reads portfolio data object into :class:`~PortfolioData`.

        The portfolio data is expected to contain at least the following columns names:

            - ``"assets"``: The name of the asset.
            - ``"outstanding_now_now"``: Current outstanding amount per asset.
            - ``"min_outstanding_future"``: Lower bound outstanding amount in the future per asset.
            - ``"max_outstanding_future"``: Upper bound outstanding amount in the future per asset.
            - ``"income_now"``: Current income per asset, corresponds to return multiplied by the current outstanding amount.
            - ``"regcap_now"``: Current regulatory capital per asset.

        Different column names in the dataset can be used, but in that case they need to
        be provided as a renaming dictionary to the ``columns_rename`` argument.

        Args:
            filename: path to portfolio data. If instead ``benchmark_dataset`` is
                provided, a default benchmark dataset containing 52 assets will be used.
            columns_rename: to rename columns provided as dict with new column names as keys
                and to replace column name as value. Example
                ``{"outstanding_2021": "outstanding_now"}``.

        Raises:
            ValueError if required columns are not present in dataset.
        """  # noqa: E501
        if str(filename) == "benchmark_dataset":
            filename = Path(__file__).parents[1] / "datasets" / "benchmark_dataset.xlsx"

        filename = str(filename)
        if filename.endswith(".xlsx"):
            portfolio_data = pd.read_excel(filename)
        elif filename.endswith(".csv"):
            portfolio_data = pd.read_csv(filename)
        elif filename.endswith(".json"):
            portfolio_data = pd.read_json(filename)
        else:
            error_msg = "Datatype not supported."
            raise ValueError(error_msg)
        return cls(portfolio_data, columns_rename)

    def get_outstanding_now(self) -> NDArray[np.float64]:
        """Gets the `outstanding_now` data from the dataset.

        Returns:
            The `outstanding_now` column from the dataset as a numpy array.
        """
        return cast(
            "NDArray[np.float64]", self.portfolio_df["outstanding_now"].to_numpy()
        )

    def get_l_bound(self) -> NDArray[np.float64]:
        """Gets the `l_bound` data from the dataset.

        Returns:
            The `min_outstanding_future` column from the dataset as a numpy array.
        """
        return cast(
            "NDArray[np.float64]",
            self.portfolio_df["min_outstanding_future"].to_numpy(),
        )

    def get_u_bound(self) -> NDArray[np.float64]:
        """Gets the `u_bound` data from the dataset.

        Returns:
            The `max_outstanding_future` column from the dataset as a numpy array.
        """
        return cast(
            "NDArray[np.float64]",
            self.portfolio_df["max_outstanding_future"].to_numpy(),
        )

    def get_income(self) -> NDArray[np.float64]:
        """Gets the `income` data from the dataset.

        Returns:
            The `income_now` column from the dataset as a numpy array.
        """
        return cast("NDArray[np.float64]", self.portfolio_df["income_now"].to_numpy())

    def get_capital(self) -> NDArray[np.float64]:
        """Gets the `capital` data from the dataset.

        Returns:
            The `regcap_now` column from the dataset as a numpy array.
        """
        return cast("NDArray[np.float64]", self.portfolio_df["regcap_now"].to_numpy())

    def get_returns(self) -> NDArray[np.float64]:
        """Gets the `returns` data from the dataset.

        Returns:
            Returns is defined as income / outstanding_now
        """
        income = self.get_income()
        outstanding_now = self.get_outstanding_now()
        return cast("NDArray[np.float64]", income / outstanding_now)

    def get_column(self, column_name: str) -> NDArray[np.float64]:
        """Gets the specified column from the dataset.

        Args:
            column_name: Name of the column to get.

        Returns:
            The `regcap_now` columns from the dataset as a numpy array.
        """
        return cast("NDArray[np.float64]", self.portfolio_df[column_name].to_numpy())

    def print_portfolio_info(self) -> None:
        """Prints information about portfolio data to terminal."""
        outstanding_now = self.get_outstanding_now()
        l_bound = self.get_l_bound()
        u_bound = self.get_u_bound()
        income = self.get_income()
        capital = self.get_capital()

        # Calculate the total outstanding amount in now
        total_outstanding_now = np.sum(outstanding_now)
        self._logger.info("--------- Portfolio information -----------")
        self._logger.info("Total outstanding now: %.2f", total_outstanding_now)

        # Calculate the ROC for now
        roc_now = np.sum(income) / np.sum(capital)
        self._logger.info("ROC now: %.4f", roc_now)

        # Calculate the HHI diversification for now
        hhi_now = np.sum(total_outstanding_now**2) / np.sum(total_outstanding_now) ** 2
        self._logger.info("HHI now: %.4f", hhi_now)

        if "emis_intens_now" in self.portfolio_df:
            # Calculate the total emissions for now
            total_emission_now = np.sum(
                self.get_column("emis_intens_now") * total_outstanding_now
            )
            self._logger.info("Total Emission now: %.2f", total_emission_now)

            # Calculate the average emission intensity now
            relative_total_emission = total_emission_now / total_outstanding_now
            self._logger.info(
                "Relative emission intensity now: %.2f", relative_total_emission
            )

        # Estimate the total outstanding amount and its standard deviation for future.
        # This follows from the assumption of a symmetric probability distribution on
        # the interval [l_bound, u_bound] and the central limit theorem.
        expected_total_outstanding_future = np.sum(u_bound + l_bound) / 2
        expected_stddev_total_outstanding_future = np.linalg.norm(
            (u_bound - l_bound) / 2
        )

        self._logger.info(
            "Expected total outstanding future: %.2f", expected_total_outstanding_future
        )
        self._logger.info("Std dev: %.2f", expected_stddev_total_outstanding_future)

        # Estimate average growth factor and its standard deviation for now-future. This
        # consists of the (averaged) amount per asset in future, which is the outcome of
        # the optimization, divided by the amount for now.
        expected_average_growth_fac = np.sum(
            (u_bound + l_bound) / (2 * total_outstanding_now)
        )
        expected_stddev_average_growth_fac = np.linalg.norm(
            (u_bound - l_bound) / (2 * total_outstanding_now)
        )
        self._logger.info(
            "Expected average growth factor: %.4f", expected_average_growth_fac
        )
        self._logger.info("Std dev: %.4f", expected_stddev_average_growth_fac)
        self._logger.info("--------- --------------------- -----------")

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.portfolio_df)

    def __repr__(self) -> str:
        """Representation for debugging."""
        txt = f"{self.__class__.__name__} object containing the following data:\n"
        txt += repr(self.portfolio_df)
        return txt

    def __str__(self) -> str:
        """String representation of the :class:`~PortfolioData` object."""
        txt = f"{self.__class__.__name__} object containing the following data:\n"
        txt += str(self.portfolio_df)
        return txt

    def __contains__(self, other: object) -> bool:
        """Check if ``other`` is part of the dataset."""
        return other in self.portfolio_df
