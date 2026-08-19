
"""Plotting helpers for the Jumper notebook supporting both PCA and ConvVAE."""

from typing import Sequence, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_simulation_snapshots(simus: Sequence, names: Sequence[str]):
    """
    Plot the first time step for each simulation.

    Parameters
    ----------
    simus : sequence
        Simulation objects.
    names : sequence of str
        Labels for each simulation.

    Returns
    -------
    tuple
        Matplotlib ``(fig, axes)``.
    """
    fig, axes = plt.subplots(1, len(simus), figsize=(20, 4), squeeze=False)
    axes = axes.flatten()

    for ax, simu, name in zip(axes, simus, names, strict=True):
        data = simu.simulation[0]
        # Handle 3D spatial variables vs 2D surface variables
        if data.ndim == 3:
            im = ax.pcolormesh(data[0])
            ax.set_title(f"Surface {name}")
        else:
            im = ax.pcolormesh(data)
            ax.set_title(f"{name}")
            
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x (grid index)")
        ax.set_ylabel("y (grid index)")

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_dr_diagnostics(simus: Sequence, names: Sequence[str], colors: Sequence[str]):
    """
    Plot dimensionality reduction diagnostics (PCA variance / CVAE loss curves,
    latent series, and leading reconstructed mode) for each simulation.

    Parameters
    ----------
    simus : sequence
        Simulation objects.
    names : sequence of str
        Labels for each simulation.
    colors : sequence of str
        Colors to use for plotting latents.

    Returns
    -------
    tuple
        Matplotlib ``(fig, axes)``.
    """
    fig, axes = plt.subplots(3, len(simus), figsize=(20, 10), squeeze=False)

    for i, (simu, name, color) in enumerate(zip(simus, names, colors, strict=True)):
        dr_model = getattr(simu, "dimensionality_reduction", getattr(simu, "dr", None))

        # --- Row 0: Variance Ratio (PCA) or Loss Curve (ConvVAE) ---
        if hasattr(dr_model, "explained_variance_ratio_"):
            axes[0, i].plot(dr_model.explained_variance_ratio_ * 100, "ko-", markersize=4)
            axes[0, i].set_title(f"Explained Variance Ratio - {name}")
            axes[0, i].set_xlabel("Component")
            axes[0, i].set_ylabel("Variance (%)")
        elif hasattr(dr_model, "train_losses") and len(dr_model.train_losses) > 0:
            axes[0, i].plot(dr_model.train_losses, label="Train Loss", color=color)
            if hasattr(dr_model, "val_losses") and len(dr_model.val_losses) > 0:
                axes[0, i].plot(dr_model.val_losses, label="Val Loss", linestyle="--", color="black")
            axes[0, i].set_title(f"CVAE Training Loss - {name}")
            axes[0, i].set_xlabel("Epoch")
            axes[0, i].set_ylabel("Loss")
            axes[0, i].legend()
        else:
            axes[0, i].text(0.5, 0.5, "No Diagnostic Stats Available", ha="center", va="center")
            axes[0, i].set_title(f"Diagnostics - {name}")

        # --- Row 1: Latent Space Time Series (1st & 2nd components/dimensions) ---
        latents = getattr(simu, "components", getattr(simu, "latents", None))
        if latents is not None and latents.shape[1] > 0:
            axes[1, i].plot(latents[:, 0], color=color, alpha=0.9, label="Latent 1 ($z_0$)")
            if latents.shape[1] > 1:
                axes[1, i].plot(latents[:, 1], color=color, alpha=0.4, label="Latent 2 ($z_1$)")
            axes[1, i].set_title(f"Latent Vectors - {name}")
            axes[1, i].set_xlabel("Time step")
            axes[1, i].set_ylabel("Latent Value")
            axes[1, i].legend()

        # --- Row 2: Spatial Feature / Mode 0 ---
        try:
            comp_mode = simu.get_component(0)
            if comp_mode.ndim == 3:
                im = axes[2, i].pcolormesh(comp_mode[0])
                axes[2, i].set_title(f"1st Mode Surface - {name}")
            else:
                im = axes[2, i].pcolormesh(comp_mode)
                axes[2, i].set_title(f"1st Mode - {name}")
            plt.colorbar(im, ax=axes[2, i])
        except Exception:
            axes[2, i].text(0.5, 0.5, "Spatial Mode View N/A", ha="center", va="center")

        axes[2, i].set_xlabel("x (grid index)")
        axes[2, i].set_ylabel("y (grid index)")

    fig.suptitle("DIMENSIONALITY REDUCTION DIAGNOSTICS")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return fig, axes


def plot_rmse_depth_profile(
    values: Sequence[np.ndarray],
    depth: Any,
    names: Sequence[str],
    colors: Sequence[str],
    title: str,
):
    """
    Plot RMSE error bars versus depth.

    Expects variables with depth dimensions. Each entry in ``values``
    must have shape ``(depth, time)``.
    """
    fig, ax = plt.subplots(figsize=(6, 8))
    for val, name, color in zip(values, names, colors, strict=True):
        ax.errorbar(
            np.mean(val, axis=1),
            depth,
            xerr=np.std(val, axis=1),
            fmt=".",
            label=name,
            color=color,
            ecolor="grey",
        )

    ax.set_title(title)
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("RMSE")
    ax.legend()
    ax.invert_yaxis()
    plt.show()
    return fig, ax


def plot_rmse_maps(maps: Sequence[np.ndarray], names: Sequence[str]):
    """Plot mean RMSE spatial maps for each variable."""
    fig, axes = plt.subplots(1, len(maps), figsize=(20, 5), squeeze=False)
    axes = axes.flatten()

    for ax, rmse_map, name in zip(axes, maps, names, strict=True):
        # Collapse time / depth dimensions down to a 2D (Y, X) map
        data = rmse_map
        while data.ndim > 2:
            data = np.nanmean(data, axis=0)

        im = ax.pcolormesh(data)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Mean RMSE map - {name}")
        ax.set_xlabel("x (grid index)")
        ax.set_ylabel("y (grid index)")

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_reconstructions(maps: Sequence[np.ndarray], names: Sequence[str]):
    """Plot reconstructed spatial fields (first time step) for each variable."""
    fig, axes = plt.subplots(1, len(maps), figsize=(20, 4), squeeze=False)
    axes = axes.flatten()

    for ax, simu, name in zip(axes, maps, names, strict=True):
        if simu.ndim > 3:  # (time, depth, y, x)
            im = ax.pcolormesh(simu[0, 0])
            ax.set_title(f"Surface {name}")
        else:  # (time, y, x)
            im = ax.pcolormesh(simu[0])
            ax.set_title(f"{name}")

        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x (grid index)")
        ax.set_ylabel("y (grid index)")

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_bar_with_errors(
    categories: Sequence[str],
    means: Sequence[float],
    errors: Sequence[float],
    title: str,
    ylabel: str,
    colors: Sequence[str] | None = None,
    xlabel: str = "Categories",
):
    """Plot a bar chart with error bars."""
    if colors is None:
        colors = [f"C{i}" for i in range(len(categories))]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(categories, means, yerr=errors, capsize=5, color=colors)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.show()
    return fig, ax


def plot_depth_error_profiles(
    depth: Any,
    mean_ref: Sequence[np.ndarray],
    std_ref: Sequence[np.ndarray],
    mean_pred: Sequence[np.ndarray],
    std_pred: Sequence[np.ndarray],
    labels: Sequence[str],
    colors: Sequence[str],
    title: str,
):
    """Plot absolute error profiles over depth for multiple variables."""
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 6), squeeze=False)
    axes = axes.flatten()

    for i, label in enumerate(labels):
        axes[i].plot(
            depth,
            mean_ref[i],
            color="black",
            label=f"{label} ref",
            linestyle="dashed",
            alpha=0.6,
        )
        axes[i].fill_between(
            depth,
            mean_ref[i] + std_ref[i],
            mean_ref[i] - std_ref[i],
            color="black",
            alpha=0.1,
        )

        axes[i].plot(depth, mean_pred[i], color=colors[i], label=f"{label} pred")
        axes[i].fill_between(
            depth,
            mean_pred[i] + std_pred[i],
            mean_pred[i] - std_pred[i],
            color=colors[i],
            alpha=0.2,
        )
        axes[i].set_xlabel("Depth (m)")
        axes[i].set_ylabel("Mean Error")
        axes[i].legend()

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return fig, axes


def plot_depth_prediction_reference(
    depth: Any,
    mean_pred: Sequence[np.ndarray],
    mean_ref: Sequence[np.ndarray],
    titles: Sequence[str],
    ylabel: str = "",
):
    """Plot prediction vs reference profiles over depth for multiple variables."""
    fig, axes = plt.subplots(1, len(titles), figsize=(15, 4), squeeze=False)
    axes = axes.flatten()

    for ax, title, pred, ref in zip(axes, titles, mean_pred, mean_ref, strict=True):
        ax.plot(depth, pred, label="predictions")
        ax.plot(depth, ref, label="reference")
        ax.set_title(title)
        ax.set_xlabel("Depth (m)")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.legend()

    fig.suptitle("Average over depth")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return fig, axes


def plot_component_timeseries(
    ref: Sequence,
    pred: Sequence[pd.DataFrame],
    names: Sequence[str],
    colors: Sequence[str],
    comp: int,
    total_len: int,
    train_len: int,
    steps_per_year: int = 1,
):
    """
    Plot latent component time series ($z_{comp}$) for reference and predicted data.
    """
    fig, axes = plt.subplots(len(ref), 1, figsize=(10, 8), squeeze=False)
    axes = axes.flatten()

    for ax, simu, pred_item, name, color in zip(
        axes, ref, pred, names, colors, strict=True
    ):
        latents = getattr(simu, "components", getattr(simu, "latents", None))

        if latents is not None:
            ax.plot(
                latents[:, comp],
                color="grey",
                linestyle="dashed",
                label="ref (separate DR fit)",
            )

        # Training segment (dashed)
        ax.plot(
            np.arange(0, train_len),
            pred_item.iloc[:train_len, comp],
            color=color,
            alpha=0.9,
            linestyle="dashed",
            label=f"{name} (training)",
        )

        # Forecast segment (solid)
        ax.plot(
            np.arange(train_len - 1, total_len),
            pred_item.iloc[train_len - 1 :, comp],
            color=color,
            alpha=0.9,
            label=f"{name} (forecast)",
        )

        ax.set_title(f"Latent Component {comp} - {name}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Component value")
        ax.set_xticks(range(0, total_len, steps_per_year))
        ax.legend()

    fig.suptitle("LATENT TIME SERIES FORECAST")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return fig, axes
