"""This module contains visualization tools."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from tno.quantum.problems.portfolio_optimization.components.postprocess import (
    pareto_front,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from matplotlib.axes import Axes
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Colormap
    from matplotlib.typing import ColorType
    from numpy.typing import ArrayLike


def plot_points(
    diversification_values: ArrayLike,
    roc_values: ArrayLike,
    color: str | None = None,
    label: str | None = None,
    c: ArrayLike | Sequence[ColorType] | ColorType | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float | None = None,
    cmap: str | Colormap | None = None,
    ax: Axes | None = None,  # pylint: disable=invalid-name
) -> PatchCollection | Any:
    """Plots the given data-points in a Diversification-ROC plot.

    Args:
        diversification_values: 1-D ``ArrayLike`` containing the x values of the plot.
        roc_values: 1-D ``ArrayLike`` containing the y values of the plot.
        color: Optional color to use for the points. For an overview of allowed colors
            see the `Matplotlib Documentation`_. If ``None`` is given, a default color
            will be assigned by ``matplotlib``. Default is ``None``.
        label: Label to use in the legend. If ``None`` is given, no label will be used.
            Default is ``None``.
        c: The marker colors as used by ``matplotlib``.
        vmin: min value of data range that colormap covers as used by ``matplotlib``.
        vmax: max value of data range that colormap covers as used by ``matplotlib``.
        alpha: The alpha blending value as used by ``matplotlib``.
        cmap: The Colormap instance or registered colormap name as used by ``matplotlib``.
        ax:  ``Axes`` to plot on. If ``None``, a new figure with one ``Axes`` will be
            created.

    Returns:
        The ``matplotlib`` PathCollection object created by scatter.

    .. _Matplotlib Documentation: https://matplotlib.org/stable/gallery/color/named_colors.html
    """
    if ax is None:
        _, ax = plt.subplots()
    collection = ax.scatter(
        diversification_values,
        roc_values,
        color=color,
        c=c,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        label=label,
    )
    ax.set_xlabel("Diversification Change (%)")
    ax.set_ylabel("ROC Change (%)")
    ax.grid()
    if label is not None:
        ax.legend()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlim(*xlim, auto=True)
    ax.set_ylim(*ylim, auto=True)
    return collection


def plot_front(
    diversification_values: ArrayLike,
    roc_values: ArrayLike,
    color: str | None = None,
    label: str | None = None,
    c: ArrayLike | Sequence[ColorType] | ColorType | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float | None = None,
    cmap: str | Colormap | None = None,
    ax: Axes | None = None,  # pylint: disable=invalid-name
) -> PatchCollection:
    """Plots a pareto front of the given data-points in a Diversification-ROC plot.

    Args:
        diversification_values: 1-D ``ArrayLike`` containing the x values of the plot.
        roc_values: 1-D ``ArrayLike`` containing the y values of the plot.
        color: Optional color to use for the points. For an overview of allowed colors
            see the `Matplotlib Documentation`_. If ``None`` is given, a default color
            will be assigned by ``matplotlib``. Default is ``None``.
        label: Label to use in the legend. If ``None`` is given, no label will be used.
            Default is ``None``.
        c: The marker colors as used by ``matplotlib``.
        vmin: min value of data range that colormap covers as used by ``matplotlib``.
        vmax: max value of data range that colormap covers as used by ``matplotlib``.
        alpha: The alpha blending value as used by ``matplotlib``.
        cmap: The Colormap instance or registered colormap name as used by ``matplotlib``.
        ax:  ``Axes`` to plot on. If ``None``, a new figure with one ``Axes`` will be
            created.

    Returns:
        The ``matplotlib`` PathCollection object created by scatter.

    .. _Matplotlib Documentation: https://matplotlib.org/stable/gallery/color/named_colors.html
    """
    x_values, y_values = pareto_front(diversification_values, roc_values)
    return plot_points(
        x_values,
        y_values,
        color=color,
        c=c,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        label=label,
        ax=ax,
    )
