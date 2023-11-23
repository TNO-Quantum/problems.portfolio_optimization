"""This module implements I/O"""
from __future__ import annotations

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
    """The ``PortfolioData`` stores the data used for portfolio optimization."""

    def __init__(
        self,
        portfolio_dataframe: DataFrame,
        columns_rename: dict[str, str] | None = None,
    ):
        """Create a ``PortfolioData`` object from a pandas ``DataFrame``.

        The portfolio data is expected to contain at least the following columns names:

            - ``"assets"``
            - ``"outstanding_now_now"``
            - ``"min_outstanding_future"``
            - ``"max_outstanding_future"``
            - ``"income_now"``
            - ``"regcap_now"``

        Different column names in the dataset can be used but need to be provided as a
        renaming dictionary to the ``columns_rename`` argument.

        Args:
            portfolio_dataframe: Pandas ``DataFrame`` containing the portfolio data.
            column_rename: to rename columns provided as dict with new column names as keys
                and to replace column name as value. Example
                ``{"outstanding_2021": "outstanding_now"}``.

        Raises:
            ValueError if required columns are not present in dataset.
        """
        if columns_rename is not None:
            portfolio_dataframe.rename(columns=columns_rename, inplace=True)

        # Validate dataset to contain required column names
        for required_column_name in DEFAULT_REQUIRED_COLUMN_NAMES:
            if required_column_name not in portfolio_dataframe.columns:
                raise ValueError(
                    f"Required column name {required_column_name} is not in dataset."
                )

        self.portfolio_df = portfolio_dataframe

    @classmethod
    def from_file(
        cls: type[PortfolioDataT],
        filename: str | Path,
        columns_rename: dict[str, str] | None = None,
    ) -> PortfolioDataT:
        """Read portfolio data object into ``PortfolioData``.

        The portfolio data is expected to contain at least the following columns names:

            - ``"assets"``
            - ``"outstanding_now_now"``
            - ``"min_outstanding_future"``
            - ``"max_outstanding_future"``
            - ``"income_now"``
            - ``"regcap_now"``

        Different column names in the dataset can be used but need to be provided as a
        renaming dictionary to the ``columns_rename`` argument.

        Args:
            filename: path to portfolio data. If instead ``benchmark_dataset`` is
                provided, a default benchmark dataset containing 52 assets will be used.
            column_rename: to rename columns provided as dict with new column names as keys
                and to replace column name as value. Example
                ``{"outstanding_2021": "outstanding_now"}``.

        Raises:
            ValueError if required columns are not present in dataset.
        """
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
            raise ValueError("Datatype not supported.")
        return cls(portfolio_data, columns_rename)

    def get_outstanding_now(self) -> NDArray[np.float_]:
        """Get the `outstanding_now` data from the dataset.

        Returns:
            The `outstanding_now` column from the dataset as a numpy array.
        """
        return cast(NDArray[np.float_], self.portfolio_df["outstanding_now"].to_numpy())

    def get_l_bound(self) -> NDArray[np.float_]:
        """Get the `l_bound` data from the dataset.

        Returns:
            The `min_outstanding_future` column from the dataset as a numpy array.
        """
        return cast(
            NDArray[np.float_], self.portfolio_df["min_outstanding_future"].to_numpy()
        )

    def get_u_bound(self) -> NDArray[np.float_]:
        """Get the `u_bound` data from the dataset.

        Returns:
            The `max_outstanding_future` column from the dataset as a numpy array.
        """
        return cast(
            NDArray[np.float_], self.portfolio_df["max_outstanding_future"].to_numpy()
        )

    def get_income(self) -> NDArray[np.float_]:
        """Get the `income` data from the dataset.

        Returns:
            The `income_now` column from the dataset as a numpy array.
        """
        return cast(NDArray[np.float_], self.portfolio_df["income_now"].to_numpy())

    def get_capital(self) -> NDArray[np.float_]:
        """Get the `capital` data from the dataset.

        Returns:
            The `regcap_now` column from the dataset as a numpy array.
        """
        return cast(NDArray[np.float_], self.portfolio_df["regcap_now"].to_numpy())

    def get_returns(self) -> NDArray[np.float_]:
        """Get the `returns` data from the dataset.

        Returns:
            Returns is defined as income / outstanding_now
        """
        income = self.get_income()
        outstanding_now = self.get_outstanding_now()
        return cast(NDArray[np.float_], income / outstanding_now)

    def get_column(self, column_name: str) -> NDArray[np.float_]:
        """Get the specified column from the dataset.

        Args:
            column_name: Name of the column to get.

        Returns:
            The `regcap_now` columns from the dataset as a numpy array.
        """
        return cast(NDArray[np.float_], self.portfolio_df[column_name].to_numpy())

    def print_portfolio_info(self) -> None:
        """Print information about portfolio data to terminal."""
        outstanding_now = self.get_outstanding_now()
        l_bound = self.get_l_bound()
        u_bound = self.get_u_bound()
        income = self.get_income()
        capital = self.get_capital()

        # Calculate the total outstanding amount in now
        total_outstanding_now = np.sum(outstanding_now)
        print(f"Total outstanding now: {total_outstanding_now:.2f}")

        # Calculate the ROC for now
        roc_now = np.sum(income) / np.sum(capital)
        print(f"ROC now: {roc_now:.6f}")

        # Calculate the HHI diversification for now
        hhi_now = (
            np.sum(total_outstanding_now**2) / np.sum(total_outstanding_now) ** 2
        )
        print(f"HHI now: {hhi_now:.4f}")

        if "emis_intens_now" in self.portfolio_df:
            # Calculate the total emissions for now
            total_emission_now = np.sum(
                self.get_column("emis_intens_now") * total_outstanding_now
            )
            print(f"Total Emission now: {total_emission_now:.2f}")

            # Calculate the average emission intensity now
            relative_total_emission = total_emission_now / total_outstanding_now
            print(f"Relative emission intensity now: {relative_total_emission:.4f}")

        # Estimate the total outstanding amount and its standard deviation for future. This
        # follows from the assumption of a symmetric probability distribution on the
        # interval [l_bound, u_bound] and the central limit theorem.
        expected_total_outstanding_future = np.sum(u_bound + l_bound) / 2
        expected_stddev_total_outstanding_future = np.linalg.norm(
            (u_bound - l_bound) / 2
        )
        print(
            f"Expected total outstanding future: {expected_total_outstanding_future:.2f}",
            f"Std dev: {expected_stddev_total_outstanding_future:.2f}",
        )

        # Estimate a average growth factor and its standard deviation for now-future. This
        # consists of the (averaged) amount per asset in future, which is the outcome of the
        # optimization, divided by the amount for now.
        expected_average_growth_fac = np.sum(
            (u_bound + l_bound) / (2 * total_outstanding_now)
        )
        expected_stddev_average_growth_fac = np.linalg.norm(
            (u_bound - l_bound) / (2 * total_outstanding_now)
        )
        print(
            f"Expected average growth factor: {expected_average_growth_fac:.4f}",
            f"Std dev: {expected_stddev_average_growth_fac:.4f}",
        )

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.portfolio_df)

    def __repr__(self) -> str:
        """Representation for debugging."""
        txt = f"{self.__class__.__name__} object containing the following data:\n"
        txt += repr(self.portfolio_df)
        return txt

    def __str__(self) -> str:
        """String representation of the ``PortfolioData`` object."""
        txt = f"{self.__class__.__name__} object containing the following data:\n"
        txt += str(self.portfolio_df)
        return txt

    def __contains__(self, other: object) -> bool:
        """Check if ``other`` is part of the dataset."""
        return other in self.portfolio_df
