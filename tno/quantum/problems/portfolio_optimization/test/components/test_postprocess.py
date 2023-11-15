"""This module contains tests for post-processing of results."""
from typing import Mapping

import numpy as np
import pytest
from dimod import SampleSet
from numpy.typing import NDArray
from pandas import DataFrame

from tno.quantum.problems.portfolio_optimization.components import Decoder


@pytest.fixture(name="decoder")
def decoder_fixture() -> Decoder:
    columns = [
        "out_now",
        "out_future_min",
        "out_future_max",
        "emis_intens_now",
        "emis_intens_future",
        "income_now",
        "regcap_now",
    ]
    index = ["asset 1", "asset 2"]
    data = [
        [1.0, 10.0, 19.0, 100.0, 76.0, 1.0, 1.0],
        [2.0, 30.0, 39.0, 200.0, 152.0, 1.0, 1.0],
    ]
    portfolio_data = DataFrame(data=data, columns=columns, index=index)
    return Decoder(portfolio_data, 0, 2)


@pytest.mark.parametrize(
    "sample,expected_results",
    [
        ({0: 0, 1: 0, 2: 0, 3: 0}, np.array([10, 30])),
        ({0: 0, 1: 0, 2: 0, 3: 1}, np.array([10, 36])),
        ({0: 0, 1: 0, 2: 1, 3: 0}, np.array([10, 33])),
        ({0: 0, 1: 0, 2: 1, 3: 1}, np.array([10, 39])),
        ({0: 0, 1: 1, 2: 0, 3: 0}, np.array([16, 30])),
        ({0: 0, 1: 1, 2: 0, 3: 1}, np.array([16, 36])),
        ({0: 0, 1: 1, 2: 1, 3: 0}, np.array([16, 33])),
        ({0: 0, 1: 1, 2: 1, 3: 1}, np.array([16, 39])),
        ({0: 1, 1: 0, 2: 0, 3: 0}, np.array([13, 30])),
        ({0: 1, 1: 0, 2: 0, 3: 1}, np.array([13, 36])),
        ({0: 1, 1: 0, 2: 1, 3: 0}, np.array([13, 33])),
        ({0: 1, 1: 0, 2: 1, 3: 1}, np.array([13, 39])),
        ({0: 1, 1: 1, 2: 0, 3: 0}, np.array([19, 30])),
        ({0: 1, 1: 1, 2: 0, 3: 1}, np.array([19, 36])),
        ({0: 1, 1: 1, 2: 1, 3: 0}, np.array([19, 33])),
        ({0: 1, 1: 1, 2: 1, 3: 1}, np.array([19, 39])),
    ],
)
def test_decode_sample(
    decoder: Decoder, sample: Mapping[int, int], expected_results: NDArray[np.float_]
) -> None:
    np.testing.assert_array_equal(decoder.decode_sample(sample), expected_results)


def test_decode_sampleset(decoder: Decoder) -> None:
    samples = [
        {0: 0, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 0, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: 1},
        {0: 0, 1: 1, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 1, 3: 0},
        {0: 1, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 1, 1: 1, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 1, 3: 0},
        {0: 1, 1: 1, 2: 1, 3: 1},
    ]
    sampleset = SampleSet.from_samples(samples, "BINARY", [0] * 16)
    expected_result = np.array(
        [
            [10, 30],
            [10, 36],
            [10, 33],
            [10, 39],
            [16, 30],
            [16, 36],
            [16, 33],
            [16, 39],
            [13, 30],
            [13, 36],
            [13, 33],
            [13, 39],
            [19, 30],
            [19, 36],
            [19, 33],
            [19, 39],
        ]
    )

    np.testing.assert_array_equal(decoder.decode_sampleset(sampleset), expected_result)
