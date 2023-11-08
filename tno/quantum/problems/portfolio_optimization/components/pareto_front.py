"""This module implements pareto front"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import ConvexHull


def pareto_front(
    x: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    min_points = 100
    points = np.vstack((x, y)).T
    points = np.unique(points, axis=0)
    if len(points) < 3:
        return points.T[0], points.T[1]

    hull = ConvexHull(points)
    pareto_points = points[hull.vertices]

    for _ in range(min_points):
        if len(pareto_points) > min_points:
            break
        points = np.delete(points, hull.vertices, axis=0)
        if len(points) < 3:
            break
        hull = ConvexHull(points)
        pareto_points = np.vstack((pareto_points, points[hull.vertices]))

    return pareto_points.T[0], pareto_points.T[1]
