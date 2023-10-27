from __future__ import annotations

import pandas as pd


def read_portfolio_data(filename: str) -> pd.DataFrame:
    print("Status: reading data")
    portfolio_data = pd.read_excel(filename)
    # TODO remove hardcoded 52, this is a budge for the corrupt input data. Data should
    # be fixend and the hardcoed 52 should be removed
    portfolio_data = portfolio_data[:52]
    return portfolio_data


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
