"""This module implements pareto front"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import ConvexHull


def pareto_front(
    x: ArrayLike, y: ArrayLike, min_points: int = 100
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Calculate the pareto front with at least min_points data points by repeatedly
    creating a convex hull around data points.

    Args:
        x: x-values of data points
        y: y-values of data points
        min_points: minimum number of points to be selected

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
        if len(pareto_points) > min_points:
            break
        # Remove current hull vertices from data and create a new hull
        points = np.delete(points, hull.vertices, axis=0)
        if len(points) < 3:
            break
        hull = ConvexHull(points)
        # Add the new hull vertices to the pareto front
        pareto_points = np.vstack((pareto_points, points[hull.vertices]))

    return pareto_points.T[0], pareto_points.T[1]
