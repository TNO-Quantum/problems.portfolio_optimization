import pandas as pd


def read_portfolio_data(filename: str):
    print("Status: reading data")
    r = pd.read_excel(filename)
    # Outstanding amounts of the portfolio in 2021
    out2021 = r["out_2021"]
    # Lower and Upper bound for the 2030 portfolio
    LB = r["out_2030_min"]
    UB = r["out_2030_max"]
    # Emission intensity, income and regulatory capital of the portfolio in 2021
    e = r["emis_intens_2021"]
    income = r["income_2021"]
    capital = r["regcap_2021"]
    # Convert these to numpy arrays.
    out2021 = out2021.to_numpy()
    LB = LB.to_numpy()
    UB = UB.to_numpy()
    e = e.to_numpy()
    e = (e / 100).astype(float)
    income = income.to_numpy()
    capital = capital.to_numpy()
    return out2021, LB, UB, e, income, capital
