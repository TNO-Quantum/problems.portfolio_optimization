"""This module implements required preprocessing"""
import numpy as np
from pandas import DataFrame


def print_info(portfolio_data: DataFrame) -> None:
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
