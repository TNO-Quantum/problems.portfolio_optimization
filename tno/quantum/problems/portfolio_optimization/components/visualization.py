from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tno.quantum.problems.portfolio_optimization.components.containers import Results
from tno.quantum.problems.portfolio_optimization.components.io import get_rabo_fronts
from tno.quantum.problems.portfolio_optimization.components.pareto_front import (
    pareto_front,
    pareto_front_scipy,
)


def plot_points(results: Results, color1, color2, color3) -> Figure:
    # Comparing with Rabobank's fronts.
    # x/y_rabo1 corresponds to a front optimized including the emission target.
    # x/y_rabo2 corresponds to a front optimized without the emission target.
    x_rabo1, y_rabo1, x_rabo2, y_rabo2 = get_rabo_fronts()
    ax: Axes
    fig, ax = plt.subplots()
    ax.scatter(results.x3, results.y3, color=color3)
    ax.scatter(results.x2, results.y2, color=color2)
    ax.scatter(results.x1, results.y1, color=color1)
    ax.scatter(x_rabo1, y_rabo1, color="blue")
    ax.scatter(x_rabo2, y_rabo2, color="gray")
    ax.legend(
        [
            "QUBO constraint not met",
            "QUBO reduced",
            "QUBO constraint met",
            "classical constrained",
            "classical unconstrained",
        ]
    )
    ax.scatter(0, 0)
    ax.set_xlabel("Diversification")
    ax.set_ylabel("ROC")
    ax.grid()
    return fig


def plot_front(results: Results, color1, color2, color3) -> Figure:
    # Comparing with Rabobank's fronts.
    # x/y_rabo1 corresponds to a front optimized including the emission target.
    # x/y_rabo2 corresponds to a front optimized without the emission target.
    x_rabo1, y_rabo1, x_rabo2, y_rabo2 = get_rabo_fronts()
    starttime = datetime.now()

    x1, y1 = pareto_front_scipy(results.x1, results.y1)
    x2, y2 = pareto_front_scipy(results.x2, results.y2)
    x3, y3 = pareto_front_scipy(results.x3, results.y3)

    print(len(x1), len(x2), len(x3))

    print("Time consumed:", datetime.now() - starttime)

    # Make a plot of the results.
    ax: Axes
    fig, ax = plt.subplots()
    ax.scatter(x3, y3, color=color3)
    ax.scatter(x2, y2, color=color2)
    ax.scatter(x1, y1, color=color1)
    ax.scatter(x_rabo1, y_rabo1, color="blue")
    ax.scatter(x_rabo2, y_rabo2, color="gray")
    ax.legend(
        [
            "QUBO constraint not met",
            "QUBO reduced",
            "QUBO constraint met",
            "classical constrained",
            "classical unconstrained",
        ]
    )
    ax.scatter(0, 0)
    ax.set_xlabel("Diversification")
    ax.set_ylabel("ROC")
    ax.grid()
    return fig
