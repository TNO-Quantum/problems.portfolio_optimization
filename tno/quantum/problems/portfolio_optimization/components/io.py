from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_COLUMN_NAMES = [
    "asset",
    "out_now",
    "out_future_min", "out_future_max",
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


def get_rabo_fronts() -> tuple[list[float], list[float], list[float], list[float]]:
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
