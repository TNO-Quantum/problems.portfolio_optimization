"""Dataset creation for testing."""
from pandas import DataFrame


def make_test_dataset() -> DataFrame:
    """Create a small portfolio datasets for testing."""
    index = ["asset 1", "asset 2"]
    data = [
        [1.0, 10.0, 19.0, 100.0, 76.0, 1.0, 1.0],
        [2.0, 30.0, 39.0, 200.0, 152.0, 1.0, 1.0],
    ]
    column_names = [
        "outstanding_now",
        "min_outstanding_future",
        "max_outstanding_future",
        "emis_intens_now",
        "emis_intens_future",
        "income_now",
        "regcap_now",
    ]
    return DataFrame(data=data, columns=column_names, index=index)
