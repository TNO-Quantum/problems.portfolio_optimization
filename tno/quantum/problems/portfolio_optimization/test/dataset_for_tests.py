"""Dataset creation for testing."""
from pandas import DataFrame

from tno.quantum.problems.portfolio_optimization.components.io import PortfolioData


def make_test_dataset() -> PortfolioData:
    """Creates a small portfolio datasets for testing."""
    data = [
        ["asset 1", 1.0, 10.0, 19.0, 100.0, 76.0, 1.0, 1.0],
        ["asset 2", 2.0, 30.0, 39.0, 200.0, 152.0, 1.0, 1.0],
    ]
    column_names = [
        "asset",
        "outstanding_now",
        "min_outstanding_future",
        "max_outstanding_future",
        "emis_intens_now",
        "emis_intens_future",
        "income_now",
        "regcap_now",
    ]
    data_frame = DataFrame(data=data, columns=column_names)
    return PortfolioData(data_frame)
