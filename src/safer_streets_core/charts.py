import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_COLOUR = "#356285"


def make_radar_chart(data: pd.DataFrame, *, title: str | None = None, r_ticks: list[float | int] | None = None):
    """r is rows, theta is columns"""

    labels = data.columns

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    for _, row in data.iterrows():
        stats = np.concatenate((row.to_numpy(), [row.to_numpy()[0]]))
        ax.fill(angles, stats, c=DEFAULT_COLOUR, alpha=0.05)

    ax.set_thetagrids(angles * 180 / np.pi, [*labels, labels[0]])
    if r_ticks:
        plt.yticks(r_ticks)
    if title:
        ax.set_title(title)
    ax.grid(True)

    return fig
