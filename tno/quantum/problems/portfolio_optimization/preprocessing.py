import numpy as np
from numpy.typing import NDArray


def print_info(
    out2021: NDArray[np.float_],
    LB: NDArray[np.float_],
    UB: NDArray[np.float_],
    e: NDArray[np.float_],
    income: NDArray[np.float_],
    capital: NDArray[np.float_],
) -> None:
    # Calculate the total outstanding amount in 2021
    Out2021 = np.sum(out2021)
    print(f"Total outstanding 2021: {Out2021}")

    # Calculate the ROC for 2021
    ROC2021 = np.sum(income) / np.sum(capital)
    print(f"ROC 2021: {ROC2021}")

    # Calculate the HHI diversification for 2021
    HHI2021 = np.sum(out2021**2) / np.sum(out2021) ** 2
    print("HHI 2021: ", HHI2021)

    # Calculate the total emissions for 2021
    emis2021 = np.sum(e * out2021)
    print("Emission 2021: ", emis2021)

    # Calculate the average emission intensity 2021
    bigE = emis2021 / Out2021
    print("Emission intensity 2021:", bigE)

    # Estimate the total outstanding amount and its standard deviation for 2030. This
    # follows from the assumption of a symmetric probability distribution on the
    # interval [LB,UB] and the central limit theorem.
    Exp_total_out2030 = np.sum(UB + LB) * 0.5
    Exp_stddev_total_out2030 = np.linalg.norm(((UB - LB) / 2))
    print(
        f"Expected total outstanding 2030: {Exp_total_out2030}",
        f"Std dev: {Exp_stddev_total_out2030}",
    )

    # Estimate a average growth factor and its standard deviation for 2021-2030. This
    # consists of the (averaged) amount per asset in 2030, which is the outcome of the
    # optimization, divided by the amount for 2021.
    Exp_avr_growth_fac = np.sum((UB + LB) / (2 * out2021))
    Exp_stddev_avr_growth_fac = np.linalg.norm((UB - LB) / (2 * out2021))
    print(
        f"Expected average growth factor: {Exp_avr_growth_fac}",
        f"Std dev: {Exp_stddev_avr_growth_fac}",
    )
