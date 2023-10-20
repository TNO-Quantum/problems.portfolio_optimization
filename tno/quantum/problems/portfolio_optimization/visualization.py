from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tno.quantum.problems.portfolio_optimization.pareto_front import pareto_front


def plot_points(
    x1, y1, color1, x2, y2, color2, x3, y3, color3, x_rabo1, y_rabo1, x_rabo2, y_rabo2
) -> Figure:
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


def plot_front(
    x1, y1, color1, x2, y2, color2, x3, y3, color3, x_rabo1, y_rabo1, x_rabo2, y_rabo2
) -> Figure:
    starttime = datetime.now()

    x1, y1 = pareto_front(x1, y1)
    x2, y2 = pareto_front(x2, y2)
    x3, y3 = pareto_front(x3, y3)

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
