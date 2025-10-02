import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

DEFAULT_COLOUR = "#356285"


def make_radar_chart(
    fig: Figure,
    pos: int,
    data: pd.DataFrame,
    *,
    title: str | None = None,
    r_ticks: dict[float | int, str] | None = None,
) -> Figure:
    """
    r is rows, theta is columns
    data is assumed to be in percent
    """

    labels = data.columns

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    ax = fig.add_subplot(pos, polar=True)

    for _, row in data.iterrows():
        stats = np.concatenate((row.to_numpy(), [row.to_numpy()[0]]))
        ax.fill(angles, stats, c=DEFAULT_COLOUR, alpha=0.05)

    ax.set_thetagrids(angles * 180 / np.pi, [*labels, labels[0]])
    ax.set_ylim(-100, None)  # centre must be -100%

    if r_ticks:
        plt.yticks(list(r_ticks.keys()), list(r_ticks.values()))
    if title:
        ax.set_title(title)
    ax.grid(True)

    return fig


def make_radar_chart2(
    fig: Figure, pos: int, data: pd.DataFrame, *, title: str | None = None, r_ticks: list[float | int] | None = None
) -> Axes:
    """
    r is rows, theta is columns
    data is assumed to be in percent
    """

    labels = data.columns

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    ax = fig.add_subplot(pos, polar=True)

    for name, row in data.iterrows():
        stats = np.concatenate((row.to_numpy(), [row.to_numpy()[0]]))
        ax.plot(angles, stats, label=name)

    ax.set_ylim(-100, None)
    ax.set_thetagrids(angles * 180 / np.pi, [*labels, labels[0]])
    if r_ticks:
        plt.yticks(r_ticks)
    if title:
        ax.set_title(title)
    ax.grid(True)

    return ax
