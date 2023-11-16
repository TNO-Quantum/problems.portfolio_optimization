"""This module implements required post processing steps."""
from __future__ import annotations

from typing import Mapping

import numpy as np
from dimod import SampleSet
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame
from scipy.spatial import ConvexHull


class Decoder:
    """Decoder class ..."""

    def __init__(
        self,
        portfolio_data: DataFrame,
        k: int,
    ) -> None:
        """
        Decoder ...

        Args:
            portfolio_data: A ``pandas.Dataframe`` containing the portfolio to optimize.
            k: The number of bits that are used to represent the outstanding amount for
                each asset. A fixed point representation is used to represent `$2^k$`
                different equidistant values in the range `$[LB_i, UB_i]$` for asset i.
        """
        self.number_of_assets = len(portfolio_data)
        self.k = k
        self.mantissa = np.power(2, np.arange(self.k))
        self.LB = portfolio_data["min_outstanding_future"].to_numpy()
        self.UB = portfolio_data["max_outstanding_future"].to_numpy()
        self.multiplier = (self.UB - self.LB) / (2**self.k - 1)

    def decode_sample(self, sample: Mapping[int, int]) -> NDArray[np.float_]:
        # Compute the future portfolio
        sample_array = np.array(
            [sample[i] for i in range(self.number_of_assets * self.k)], dtype=np.uint8
        )
        sample_reshaped = sample_array.reshape((self.number_of_assets, self.k))
        ints = np.sum(sample_reshaped * self.mantissa, axis=1)
        outstanding_future = self.LB + self.multiplier * ints
        if (self.LB > outstanding_future).any() or (self.UB < outstanding_future).any():
            raise ValueError("Bounds not obeyed.")

        return np.asarray(outstanding_future, dtype=np.float_)

    def decode_sampleset(self, sampleset: SampleSet) -> NDArray[np.float_]:
        # Compute the future portfolio
        samples_matrix = sampleset.record.sample[:, : self.number_of_assets * self.k]
        samples_reshaped = samples_matrix.reshape(
            (len(sampleset), self.number_of_assets, self.k)
        )

        ints = np.sum(samples_reshaped * self.mantissa, axis=2)
        outstanding_future = self.LB + self.multiplier * ints

        if (self.LB > outstanding_future).any() or (self.UB < outstanding_future).any():
            raise ValueError("Bounds not obeyed.")

        return np.asarray(outstanding_future, dtype=np.float_)


def pareto_front(
    x: ArrayLike, y: ArrayLike, min_points: int = 50, upper_right_quadrant: bool = True
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Calculate the pareto front with at least min_points data points by repeatedly
    creating a convex hull around data points.

    Args:
        x: x-values of data points
        y: y-values of data points
        min_points: minimum number of points to be selected
        upper_right_quadrant: If ``True``, only show the upper right quadrant of the
            pareto front.

    Returns:
        x, y values of the points that are on the pareto front
    """
    points = np.vstack((x, y)).T
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


def _get_upper_quadrant(points: NDArray[np.float_]) -> NDArray[np.float_]:
    """Remove all values that are not in the upper right quadrant of the pareto front."""
    x_values = points.T[0]
    y_values = points.T[1]

    x_bound = x_values[np.argmax(y_values)]
    y_bound = y_values[np.argmax(x_values)]
    mask = (x_values >= x_bound) & (y_values >= y_bound)

    x_values = x_values[mask]
    y_values = y_values[mask]

    return np.vstack((x_values, y_values)).T
