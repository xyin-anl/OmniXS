from typing import Literal, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def plot_line_heatmap(
    data: np.ndarray,
    ax=None,
    height: Union[Literal["same"], int] = "same",
    cmap="jet",
    # norm: matplotlib.colors.Normalize = LogNorm(),
    norm: matplotlib.colors.Normalize = LogNorm(
        vmin=1, vmax=100
    ),  # Adjust normalization
    aspect=0.618,  # golden ratio
    x_ticks=None,
    y_ticks=None,
    interpolate: Union[None, int] = None,
):
    """
    Generate a heatmap from multiple lines of data.
    """

    if ax is None:
        ax = plt.gca()

    if interpolate is not None:
        from scipy.interpolate import interp1d

        x = np.arange(data.shape[1])
        x_new = np.linspace(0, data.shape[1] - 1, interpolate)
        f_x = interp1d(x, data)
        data = f_x(x_new)

    # initialize heatmap to zeros
    width = data.shape[1]
    height = width if height == "same" else height

    heatmap = np.zeros((width, height))
    max_val = data.max()
    max_val *= 1.1  # add some padding
    for line in data:
        for x_idx, y_val in enumerate(line):
            y_idx = y_val / max_val * height
            y_idx = y_idx.astype(int)
            y_idx = np.clip(y_idx, 0, height - 1)
            heatmap[y_idx, x_idx] += 1

    colorbar = ax.imshow(
        heatmap,
        origin="lower",
        cmap=cmap,
        norm=norm,
        aspect=aspect,
        interpolation="nearest",
    )

    if x_ticks is not None:
        x_ticks_pos = np.linspace(0, width, len(x_ticks))
        colorbar.axes.xaxis.set_ticks(x_ticks_pos, x_ticks)
    if y_ticks is not None:
        y_ticks_pos = np.linspace(0, height, len(y_ticks))
        colorbar.axes.yaxis.set_ticks(y_ticks_pos, y_ticks)

    return colorbar
