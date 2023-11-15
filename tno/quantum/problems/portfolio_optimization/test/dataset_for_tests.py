"""Dataset creation for testing."""
from pandas import DataFrame

from tno.quantum.problems.portfolio_optimization.components.io import (
    DEFAULT_COLUMN_NAMES,
)


def make_test_dataset() -> DataFrame:
    """Create a small portfolio datasets for testing."""
    index = ["asset 1", "asset 2"]
    data = [
        [1.0, 10.0, 19.0, 100.0, 76.0, 1.0, 1.0],
        [2.0, 30.0, 39.0, 200.0, 152.0, 1.0, 1.0],
    ]
    return DataFrame(data=data, columns=DEFAULT_COLUMN_NAMES[1:], index=index)
