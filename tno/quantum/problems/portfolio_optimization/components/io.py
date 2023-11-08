"""This module implements I/O"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

DEFAULT_COLUMN_NAMES = [
    "asset",
    "out_now",
    "out_future_min",
    "out_future_max",
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
        - ``"out_now"``
        - ``"out_future_min"``
        - ``"out_future_max"``
        - ``"emis_intens_now"``
        - ``"emis_intens_future"``
        - ``"income_now"``
        - ``"regcap_now"``

    Different column names in the dataset can be used but need to be provided as a
    renaming dictionary to the ``columns_rename`` argument.

    Args:
        filename: path to portfolio data
        column_rename: to rename columns provided as dict with new column names as keys
            and to replace column name as value. Example ``{"out_2021": "out_now"}``.

    Raises:
        ValueError if required columns are not present in dataset.
    """
    print("Status: reading data")
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


def print_portfolio_info(portfolio_data: DataFrame) -> None:
    """Print information about portfolio data to terminal.
    
    Args:
        portfolio_data: DataFrame with portfolio data.
    """
    out_now = portfolio_data["out_now"].to_numpy()
    LB = portfolio_data["out_future_min"].to_numpy()
    UB = portfolio_data["out_future_max"].to_numpy()
    e = portfolio_data["emis_intens_now"].to_numpy()
    income = portfolio_data["income_now"].to_numpy()
    capital = portfolio_data["regcap_now"].to_numpy()

    # Calculate the total outstanding amount in now
    out_now = np.sum(out_now)
    print(f"Total outstanding now: {out_now}")

    # Calculate the ROC for now
    ROC_now = np.sum(income) / np.sum(capital)
    print(f"ROC now: {ROC_now}")

    # Calculate the HHI diversification for now
    HHI_now = np.sum(out_now**2) / np.sum(out_now) ** 2
    print("HHI now: ", HHI_now)

    # Calculate the total emissions for now
    emis_now = np.sum(e * out_now)
    print("Emission now: ", emis_now)

    # Calculate the average emission intensity now
    bigE = emis_now / out_now
    print("Emission intensity now:", bigE)

    # Estimate the total outstanding amount and its standard deviation for future. This
    # follows from the assumption of a symmetric probability distribution on the
    # interval [LB,UB] and the central limit theorem.
    Exp_total_out_future = np.sum(UB + LB) * 0.5
    Exp_stddev_total_out_future = np.linalg.norm(((UB - LB) / 2))
    print(
        f"Expected total outstanding future: {Exp_total_out_future}",
        f"Std dev: {Exp_stddev_total_out_future}",
    )

    # Estimate a average growth factor and its standard deviation for now-future. This
    # consists of the (averaged) amount per asset in future, which is the outcome of the
    # optimization, divided by the amount for now.
    Exp_avr_growth_fac = np.sum((UB + LB) / (2 * out_now))
    Exp_stddev_avr_growth_fac = np.linalg.norm((UB - LB) / (2 * out_now))
    print(
        f"Expected average growth factor: {Exp_avr_growth_fac}",
        f"Std dev: {Exp_stddev_avr_growth_fac}",
    )


def get_rabo_fronts() -> tuple[list[float], list[float], list[float], list[float]]:
    """Get hardcoded pareto fronts found using classical solver."""
    # x/y_rabo1 corresponds to a front optimized including the emission target.
    x_rabo1 = [
        3.44,
        3.66,
        3.84,
        3.96,
        4.01,
        3.99,
        3.94,
        3.83,
        3.62,
        3.31,
        2.9,
        0.4,
        -3.53,
        0.9,
        2.02,
        2.73,
        3.14,
    ]
    y_rabo1 = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, -2, -1.5, -1, -0.5]

    # x/y_rabo1 corresponds to a front optimized without the emission target.
    x_rabo2 = [
        3.275,
        3.634,
        3.89,
        4.293,
        4.447,
        4.753,
        4.897,
        5.034,
        5.148,
        5.149,
        5.198,
        5.179,
    ]
    y_rabo2 = [9, 7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5]
    return x_rabo1, y_rabo1, x_rabo2, y_rabo2
