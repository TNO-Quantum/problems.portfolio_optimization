"""This module implements I/O"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

DEFAULT_COLUMN_NAMES = [
    "asset",
    "outstanding_now",
    "min_outstanding_future",
    "max_outstanding_future",
    "emis_intens_now",
    "emis_intens_future",
    "income_now",
    "regcap_now",
]


def read_portfolio_data(
    filename: str | Path, columns_rename: Optional[dict[str, str]] = None
) -> pd.DataFrame:
    """
    Read portfolio data into DataFrame.

    The portfolio data is expected to contain the following columns names:

        - ``"assets"``
        - ``"outstanding_now_now"``
        - ``"min_outstanding_future"``
        - ``"max_outstanding_future"``
        - ``"emis_intens_now"``
        - ``"emis_intens_future"``
        - ``"income_now"``
        - ``"regcap_now"``

    Different column names in the dataset can be used but need to be provided as a
    renaming dictionary to the ``columns_rename`` argument.

    Args:
        filename: path to portfolio data
        column_rename: to rename columns provided as dict with new column names as keys
            and to replace column name as value. Example ``{"outstanding_2021": "outstanding_now"}``.

    Raises:
        ValueError if required columns are not present in dataset.
    """
    if str(filename) == "rabobank":
        filename = Path(__file__).parents[1] / "datasets" / "rabodata.xlsx"

    df = pd.read_excel(str(filename))
    if columns_rename is not None:
        df.rename(columns=columns_rename, inplace=True)

    # Validate dataset to contain required column names
    for required_column_name in DEFAULT_COLUMN_NAMES:
        if required_column_name not in df.columns:
            raise ValueError(
                f"Required column name {required_column_name} is not in dataset."
            )

    return df


def print_portfolio_info(
    portfolio_data: DataFrame, columns_rename: Optional[dict[str, str]] = None
) -> None:
    """Print information about portfolio data to terminal.

    Args:
        portfolio_data: DataFrame with portfolio data.
        column_rename: to rename columns provided as dict with new column names as keys
            and to replace column name as value. Example ``{"outstanding_2021": "outstanding_now"}``.

    """
    portfolio_data_ = portfolio_data.copy()
    if columns_rename is not None:
        portfolio_data_.rename(columns=columns_rename, inplace=True)

    outstanding_now = portfolio_data_["outstanding_now"].to_numpy()
    LB = portfolio_data_["min_outstanding_future"].to_numpy()
    UB = portfolio_data_["max_outstanding_future"].to_numpy()
    e = portfolio_data_["emis_intens_now"].to_numpy()
    income = portfolio_data_["income_now"].to_numpy()
    capital = portfolio_data_["regcap_now"].to_numpy()

    # Calculate the total outstanding amount in now
    total_outstanding_now = np.sum(outstanding_now)
    print(f"Total outstanding now: {total_outstanding_now}")

    # Calculate the ROC for now
    ROC_now = np.sum(income) / np.sum(capital)
    print(f"ROC now: {ROC_now}")

    # Calculate the HHI diversification for now
    HHI_now = np.sum(total_outstanding_now**2) / np.sum(total_outstanding_now) ** 2
    print("HHI now: ", HHI_now)

    # Calculate the total emissions for now
    total_emission_now = np.sum(e * total_outstanding_now)
    print("Emission now: ", total_emission_now)

    # Calculate the average emission intensity now
    relative_total_emission = total_emission_now / total_outstanding_now
    print("Emission intensity now:", relative_total_emission)

    # Estimate the total outstanding amount and its standard deviation for future. This
    # follows from the assumption of a symmetric probability distribution on the
    # interval [LB, UB] and the central limit theorem.
    expected_total_outstanding_future = np.sum(UB + LB) / 2
    expected_stddev_total_outstanding_future = np.linalg.norm(((UB - LB) / 2))
    print(
        f"Expected total outstanding future: {expected_total_outstanding_future}",
        f"Std dev: {expected_stddev_total_outstanding_future}",
    )

    # Estimate a average growth factor and its standard deviation for now-future. This
    # consists of the (averaged) amount per asset in future, which is the outcome of the
    # optimization, divided by the amount for now.
    expected_average_growth_fac = np.sum((UB + LB) / (2 * total_outstanding_now))
    expected_stddev_average_growth_fac = np.linalg.norm(
        (UB - LB) / (2 * total_outstanding_now)
    )
    print(
        f"Expected average growth factor: {expected_average_growth_fac}",
        f"Std dev: {expected_stddev_average_growth_fac}",
    )
