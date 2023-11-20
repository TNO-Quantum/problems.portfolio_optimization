"""This module contains tests for the io module."""
from pathlib import Path

from tno.quantum.problems.portfolio_optimization.components import PortfolioData
from tno.quantum.problems.portfolio_optimization.test.dataset_for_tests import (
    make_test_dataset,
)

# pylint: disable=missing-function-docstring


def test_default_dataset() -> None:
    PortfolioData.from_file("rabobank")


def test_from_file() -> None:
    rabobank_file = Path(__file__).parents[1] / "datasets" / "rabobank.xlsx"
    PortfolioData.from_file(rabobank_file)
    PortfolioData.from_file(str(rabobank_file))


def test_init() -> None:
    data_frame = make_test_dataset()
    PortfolioData(data_frame)


def test_len() -> None:
    data_frame = make_test_dataset()
    assert len(PortfolioData(data_frame)) == 2


def test_print() -> None:
    data_frame = make_test_dataset()
    portfolio_data = PortfolioData(data_frame)
    portfolio_data.print_portfolio_info()
