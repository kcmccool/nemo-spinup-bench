"""Helper utilities used by the Jumper notebook."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nemo_spinup_forecast.forecast import Predictions, Simulation, load_ts


@dataclass(frozen=True)
class TermDef:
    """Minimal specification for one ocean term.

    Attributes
    ----------
    key : str
        Short key used as the dictionary identifier for this term.
    term : str
        Ocean variable name consumed by loaders and forecasting classes.
    filename : str
        File name pattern used to select matching simulation files.
    """

    key: str
    term: str
    filename: str


@dataclass(frozen=True)
class TermSpec(TermDef):
    """Specification for one ocean term handled in the notebook pipeline.

    Extends :class:`TermDef` with axes used for notebook analysis.

    Attributes
    ----------
    mean_axes : tuple[int, ...]
        Axes used in notebook analysis to compute mean prediction/reference
        profiles.
    err_axes : tuple[int, ...]
        Axes used in notebook analysis to reduce absolute-error arrays into
        summary statistics.
    """

    mean_axes: tuple[int, ...]
    err_axes: tuple[int, ...]


def build_simulations(
    specs: Sequence[TermDef],
    *,
    data_path: str,
    start: int,
    end: int,
    comp: Any,
    ye: bool,
    dr_method: Any,
    stand: bool = True,
) -> dict[str, Simulation]:
    """Initialize and prepare simulations for all configured terms.

    Parameters
    ----------
    specs : Sequence[TermDef]
        Specifications defining each term to load and prepare.
    data_path : str
        Root path containing simulation files.
    start : int
        Start index used when slicing simulation data.
    end : int
        End index used when slicing simulation data.
    comp : Any
        Dimensionality-reduction component configuration forwarded to
        :class:`~nemo_spinup_forecast.forecast.Simulation`.
    ye : bool
        Whether yearly processing is enabled for each simulation.
    dr_method : Any
        Dimensionality-reduction class or factory passed to
        :class:`~nemo_spinup_forecast.forecast.Simulation`.
    stand : bool, default=True
        Whether simulation data should be standardized during preparation.

    Returns
    -------
    dict[str, Simulation]
        Prepared simulation instances keyed by :attr:`TermDef.key`.

    Notes
    -----
    This function calls :meth:`Simulation.get_simulation_data` for each term
    and prints one progress message per entry.
    """
    sims: dict[str, Simulation] = {}
    for spec in specs:
        s = Simulation(
            path=data_path,
            start=start,
            end=end,
            comp=comp,
            ye=ye,
            term=spec.term,
            filename=spec.filename,
            dimensionality_reduction=dr_method,
        )
        s.get_simulation_data(stand=stand)
        sims[spec.key] = s
        print(f"{spec.key} loaded & prepared (stand={stand})")
    return sims


def decompose_all(sims: Mapping[str, Simulation]) -> None:
    """Run dimensionality reduction for each simulation.

    Parameters
    ----------
    sims : Mapping[str, Simulation]
        Simulation objects keyed by term identifier.

    Notes
    -----
    Updates simulations in place.

    Calls :meth:`~nemo_spinup_forecast.forecast.Simulation.decompose` and
    prints one progress message per key.
    """
    for k, s in sims.items():
        s.decompose()
        print(f"Decomposition applied on {k}")


def compute_rmse_for_terms(
    specs: Sequence[TermDef],
    sims: Mapping[str, Simulation],
    n_components: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    
    recs: dict[str, Any] = {}
    rmseVs: dict[str, Any] = {}
    rmseMs: dict[str, Any] = {}
    
    for spec in specs:
        s = sims[spec.key]
        dr = s.dimensionality_reduction
        n = n_components if n_components is not None else len(s.pca.components_)
        
        # 1. Get raw standardized reconstruction and truth
        rec_std = dr.reconstruct_components(n=n)
        truth_std = s.simulation
        desc = getattr(s, "desc", None)
        
        # 2. Un-standardize both to physical units
        if desc and "std" in desc:
            rec = rec_std * 2 * desc["std"] + desc["mean"]
            truth = truth_std * 2 * desc["std"] + desc["mean"]
        else:
            rec = rec_std
            truth = truth_std
            
        # 3. Compute RMSE
        valid_count = np.count_nonzero(~np.isnan(truth[0]))
        rmseV = np.sqrt(np.nansum((truth - rec) ** 2, axis=(1, 2)) / valid_count)
        rmseM = np.sqrt(np.nansum((truth - rec) ** 2, axis=0) / len(truth))
        
        recs[spec.key] = rec
        rmseVs[spec.key] = rmseV
        rmseMs[spec.key] = rmseM
        print(f"RMSE computed for {spec.key}")
        
    return recs, rmseVs, rmseMs


def make_dicos(sims: Mapping[str, Simulation]) -> dict[str, dict[str, Any]]:
    """Create serialized simulation dictionaries for all terms.

    Parameters
    ----------
    sims : Mapping[str, Simulation]
        Simulation objects keyed by term identifier.

    Returns
    -------
    dict[str, dict[str, Any]]
        Serialized simulation payloads produced by
        :meth:`~nemo_spinup_forecast.forecast.Simulation.make_dico`,
        keyed by term.

    Notes
    -----
    Prints one progress message per processed key.
    """
    d: dict[str, dict[str, Any]] = {}
    for k, s in sims.items():
        d[k] = s.make_dico()
        print(f"{k} to dictionary")
    return d


def load_ts_all(
    prepared_path: str,
    specs: Sequence[TermDef],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Load all prepared time-series DataFrames and metadata dictionaries.

    Parameters
    ----------
    prepared_path : str
        Directory containing prepared ``.npz`` and PCA files.
    specs : Sequence[TermDef]
        Term specifications defining which prepared terms to load.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]
        Tuple ``(dfs, infos)`` keyed by :attr:`TermDef.key`.

    Notes
    -----
    Each term is loaded via :func:`~nemo_spinup_forecast.forecast.load_ts` using
    ``(prepared_path, spec.term)``.
    """
    dfs: dict[str, pd.DataFrame] = {}
    infos: dict[str, dict[str, Any]] = {}
    for spec in specs:
        df, info = load_ts(f"{prepared_path}/{spec.term}", spec.term)
        dfs[spec.key] = df
        infos[spec.key] = info
    return dfs, infos


def build_predictions(
    specs: Sequence[TermDef],
    dfs: Mapping[str, pd.DataFrame],
    infos: Mapping[str, dict[str, Any]],
    forecast_method: Any,
    dr_method: Any,
) -> dict[str, Predictions]:
    """Construct prediction objects for each configured term."""
    preds: dict[str, Predictions] = {}
    for spec in specs:
        # --- FIX: If dr_method is passed as an uninstantiated class type, instantiate it dynamically ---
        if isinstance(dr_method, type):
            # Extract latent component count safely from the columns of the decomposed DataFrame
            latent_components = len(dfs[spec.key].columns)
            dr_instance = dr_method(comp=latent_components)
        else:
            dr_instance = dr_method

        preds[spec.key] = Predictions(
            spec.term, dfs[spec.key], infos[spec.key], forecast_method, dr_instance
        )
    return preds

def forecast_all(
    specs: Sequence[TermDef],
    preds: Mapping[str, Predictions],
    *,
    train_len: int,
    steps: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    """Run parallel forecasts for all terms and return raw outputs by key.

    Parameters
    ----------
    specs : Sequence[TermDef]
        Term specifications defining processing order and output keys.
    preds : Mapping[str, Predictions]
        Prediction objects keyed by :attr:`TermDef.key`.
    train_len : int
        Number of initial rows used as the training window.
    steps : int
        Forecast horizon in time steps.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]
        Tuple ``(hats, hat_stds, metrics)`` keyed by :attr:`TermDef.key`.
        ``hats`` contains the raw forecast output (forecast period only).
    """
    hats: dict[str, pd.DataFrame] = {}
    hat_stds: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for spec in specs:
        hat, hat_std, m = preds[spec.key].parallel_forecast(train_len, steps)
        hats[spec.key] = hat
        hat_stds[spec.key] = hat_std
        metrics[spec.key] = m
    return hats, hat_stds, metrics


def abs_error_stats(
    err: np.ndarray,
    *,
    steps: int,
    axes: tuple[int, ...],
) -> dict[str, Any]:
    """Compute absolute-error summary statistics for prediction and reference windows.

    Parameters
    ----------
    err : np.ndarray
        Absolute-error array, typically ``abs(reference - prediction)``.
    steps : int
        Number of time steps at the end of ``err`` used as the forecast window;
        the rest of ``err``is the reference.
    axes : tuple[int, ...]
        Axes reduced with ``nanmean`` and ``nanstd``.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``pred_mean``, ``pred_std``, ``ref_mean``,
        and ``ref_std``.
    """
    n = len(err)
    pred = err[n - steps :]
    ref = err[: n - steps]
    print("pred shape:", pred.shape)
    print("ref shape:", ref.shape)
    return {
        "pred_mean": np.nanmean(pred, axis=axes),
        "pred_std": np.nanstd(pred, axis=axes),
        "ref_mean": np.nanmean(ref, axis=axes),
        "ref_std": np.nanstd(ref, axis=axes),
    }


def normalise_time_series(sim: Simulation) -> None:
    """Normalise a simulation time series in place.

    Parameters
    ----------
    sim : Simulation
        Simulation object whose ``simulation`` array will be normalised.

    Notes
    -----
    Modifies ``sim.simulation`` in place.

    Uses ``sim.desc["mean"]`` and ``sim.desc["std"]`` for scaling.
    """
    sim.simulation = (sim.simulation - sim.desc["mean"]) / (2 * sim.desc["std"])
