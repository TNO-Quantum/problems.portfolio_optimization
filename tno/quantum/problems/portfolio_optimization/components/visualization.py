"""This module contains visualization tools."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from tno.quantum.problems.portfolio_optimization.components.postprocess import (
    pareto_front,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.typing import ColorType
    from numpy.typing import ArrayLike


def plot_points(
    diversification_values: ArrayLike,
    roc_values: ArrayLike,
    color: str | None = None,
    label: str | None = None,
    c: ArrayLike | Sequence[ColorType] | ColorType | None = None,
    alpha: float | None = None,
    cmap: str | Colormap | None = None,
    ax: Axes | None = None,  # pylint: disable=invalid-name
) -> None:
    """Plots the given data-points in a Diversification-ROC plot.

    Args:
        diversification_values: 1-D ``ArrayLike`` containing the x values of the plot.
        roc_values: 1-D ``ArrayLike`` containing the y values of the plot.
        color: Optional color to use for the points. For an overview of allowed colors
            see the `Matplotlib Documentation`_. If ``None`` is given, a default color
            will be assigned by ``matplotlib``. Default is ``None``.
        label: Label to use in the legend. If ``None`` is given, no label will be used.
            Default is ``None``.
        c: The marker colors as to be used by ``matplotlib``.
        alpha: The alpha blending value as to be used by ``matplotlib``.
        cmap: The Colormap instance or registered colormap name used by ``matplotlib``.
        ax:  ``Axes`` to plot on. If ``None``, a new figure with one ``Axes`` will be
            created.

    .. _Matplotlib Documentation: https://matplotlib.org/stable/gallery/color/named_colors.html
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(
        diversification_values,
        roc_values,
        color=color,
        c=c,
        alpha=alpha,
        cmap=cmap,
        label=label,
    )
    ax.set_xlabel("Diversification Change (%)")
    ax.set_ylabel("ROC Change (%)")
    ax.grid()
    ax.legend()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.hlines(0, xlim[0], xlim[1], colors=["black"], lw=1)
    ax.vlines(0, ylim[0], ylim[1], colors=["black"], lw=1)
    ax.set_xlim(*xlim, auto=True)
    ax.set_ylim(*ylim, auto=True)


def plot_front(
    diversification_values: ArrayLike,
    roc_values: ArrayLike,
    color: str | None = None,
    label: str | None = None,
    c: ArrayLike | Sequence[ColorType] | ColorType | None = None,
    alpha: float | None = None,
    cmap: str | Colormap | None = None,
    ax: Axes | None = None,  # pylint: disable=invalid-name
) -> None:
    """Plots a pareto front of the given data-points in a Diversification-ROC plot.

    Args:
        diversification_values: 1-D ``ArrayLike`` containing the x values of the plot.
        roc_values: 1-D ``ArrayLike`` containing the y values of the plot.
        color: Optional color to use for the points. For an overview of allowed colors
            see the `Matplotlib Documentation`_. If ``None`` is given, a default color
            will be assigned by ``matplotlib``. Default is ``None``.
        label: Label to use in the legend. If ``None`` is given, no label will be used.
            Default is ``None``.
        c: The marker colors as to be used by ``matplotlib``.
        alpha: The alpha blending value as to be used by ``matplotlib``.
        cmap: The Colormap instance or registered colormap name used by ``matplotlib``.
        ax:  ``Axes`` to plot on. If ``None``, a new figure with one ``Axes`` will be
            created.

    .. _Matplotlib Documentation: https://matplotlib.org/stable/gallery/color/named_colors.html
    """
    x_values, y_values = pareto_front(diversification_values, roc_values)
    plot_points(
        x_values, y_values, color=color, c=c, alpha=alpha, cmap=cmap, label=label, ax=ax
    )
