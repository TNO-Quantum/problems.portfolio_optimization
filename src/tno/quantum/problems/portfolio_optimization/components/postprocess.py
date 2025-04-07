"""This module implements required post processing steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import ConvexHull

if TYPE_CHECKING:
    from tno.quantum.optimization.qubo.components import ResultInterface
    from tno.quantum.problems.portfolio_optimization.components.io import PortfolioData


class Decoder:
    """``Decoder`` class for decoding samples and samplesets."""

    def __init__(
        self,
        portfolio_data: PortfolioData,
        k: int,
    ) -> None:
        """Init for the ``Decoder`` Class.

        Args:
            portfolio_data: A ``PortfolioData`` object containing the portfolio to
                optimize.
            k: The number of bits that are used to represent the outstanding amount for
                each asset. A fixed point representation is used to represent `$2^k$`
                different equidistant values in the range `$[LB_i, UB_i]$` for asset i.
        """
        self.number_of_assets = len(portfolio_data)
        self.k = k
        self.mantissa = np.power(2, np.arange(self.k))
        self.l_bound = portfolio_data.get_l_bound()
        self.u_bound = portfolio_data.get_u_bound()
        self.multiplier = (self.u_bound - self.l_bound) / (2**self.k - 1)

    def decode_result(self, result: ResultInterface) -> NDArray[np.float64]:
        """Decodes `ResultInterface` to a matrix of `outstanding_future` values.

        Each row in the matrix corresponds to a different solution in the result.

        Args:
            result: ``ResultInterface`` from a QUBO solver.

        Returns:
            Matrix containing all `outstanding future` values.
        """
        bv_matrix = np.array(
            [np.array(bv)[: self.number_of_assets * self.k] for bv, _, _ in result.freq]
        )

        bv_reshaped = bv_matrix.reshape(
            (bv_matrix.shape[0], self.number_of_assets, self.k)
        )

        ints = np.sum(bv_reshaped * self.mantissa, axis=2)
        outstanding_future = self.l_bound + self.multiplier * ints

        if (self.l_bound > outstanding_future).any() or (
            self.u_bound < outstanding_future
        ).any():
            error_msg = "Bounds are not obeyed."
            raise ValueError(error_msg)

        return np.asarray(outstanding_future, dtype=np.float64)


def pareto_front(
    xvals: ArrayLike,
    yvals: ArrayLike,
    min_points: int = 50,
    *,
    upper_right_quadrant: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate the pareto front using convex hull around data points.

    A pareto front is created by repeatedly creating a convex hull around data points.
    The final pareto front has at least `min_points` data points.

    Args:
        xvals: x-values of data points.
        yvals: y-values of data points.
        min_points: minimum number of points to be selected.
        upper_right_quadrant: If ``True``, only show the upper right quadrant of the
            pareto front.

    Returns:
        x, y values of the points that are on the pareto front.
    """
    points = np.vstack((xvals, yvals)).T
    points = np.unique(points, axis=0)
    if len(points) < 3:
        return points.T[0], points.T[1]

    hull = ConvexHull(points)
    pareto_points = points[hull.vertices]

    # Expand the pareto front so that it contains at least min_points.
    for _ in range(min_points):
        if len(pareto_points) >= min_points:
            break
        # Remove current hull vertices from data and create a new hull
        points = np.delete(points, hull.vertices, axis=0)
        if len(points) < 3:
            break
        hull = ConvexHull(points)
        # Add the new hull vertices to the pareto front
        new_points = points[hull.vertices]
        pareto_points = np.vstack((pareto_points, new_points))
        if upper_right_quadrant:
            pareto_points = _get_upper_quadrant(pareto_points)

    if upper_right_quadrant:
        pareto_points = _get_upper_quadrant(pareto_points)

    return pareto_points.T[0], pareto_points.T[1]


def _get_upper_quadrant(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Removes values that are not in the upper right quadrant of the pareto front."""
    x_values = points.T[0]
    y_values = points.T[1]

    x_bound = x_values[np.argmax(y_values)]
    y_bound = y_values[np.argmax(x_values)]
    mask = (x_values >= x_bound) & (y_values >= y_bound)

    x_values = x_values[mask]
    y_values = y_values[mask]

    return np.vstack((x_values, y_values)).T
