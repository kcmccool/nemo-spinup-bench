"""End-to-end pipeline for preparing simulations and producing forecasts."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nemo_spinup_forecast.dimensionality_reduction import DimensionalityReduction
from nemo_spinup_forecast.forecast import Predictions, Simulation
from nemo_spinup_forecast.forecast_method import BaseForecaster
from nemo_spinup_forecast.pipeline_utils import (
    TermDef,
    build_predictions,
    build_simulations,
    decompose_all,
    forecast_all,
    load_ts_all,
)


def run_pipeline(
    specs: Sequence[TermDef],
    *,
    data_path: str,
    out_dir: Path,
    start: int,
    end: int,
    steps: int,
    ye: bool,
    comp: int | float | None,
    dr_technique: DimensionalityReduction,
    forecast_technique: BaseForecaster,
) -> None:
    """Run the full spinup forecast pipeline for all configured terms.

    Prepares simulations, decomposes them, saves prepared data, loads it
    back, runs forecasts, reconstructs spatial fields, and saves predictions.

    Parameters
    ----------
    specs : Sequence[TermDef]
        Terms to process.
    data_path : str
        Root path containing raw simulation files.
    out_dir : Path
        Output directory for prepared and predicted data.
    start : int
        Start index used when slicing simulation data.
    end : int
        End index used when slicing simulation data.
    steps : int
        Forecast horizon in time steps.
    ye : bool
        Whether yearly processing is enabled.
    comp : int or float or None
        Explained variance ratio (or number of components) for dimensionality
        reduction.
    dr_technique : DimensionalityReduction
        Dimensionality reduction instance.
    forecast_technique : BaseForecaster
        Forecasting method instance.
    """
    prepared_path: str = str(out_dir / "simu_prepared")
    predicted_path: Path = out_dir / "simu_predicted"

    # --- Prepare ---
    sims: dict[str, Simulation] = build_simulations(
        specs,
        data_path=data_path,
        start=start,
        end=end,
        comp=comp,
        ye=ye,
        dr_method=dr_technique,
    )
    decompose_all(sims)

    for spec in specs:
        os.makedirs(f"{prepared_path}/{spec.term}", exist_ok=True)
        sims[spec.key].save(prepared_path, spec.term)
        print(f"{spec.term} saved to {prepared_path}/{spec.term}")

    # --- Forecast ---
    dfs: dict[str, pd.DataFrame]
    infos: dict[str, dict[str, Any]]
    dfs, infos = load_ts_all(prepared_path, specs)

    preds: dict[str, Predictions] = build_predictions(
        specs, dfs, infos, forecast_technique, dr_technique
    )
    train_len = len(preds[specs[0].key])

    hats: dict[str, pd.DataFrame]
    hats, _hat_stds, _metrics = forecast_all(
        specs, preds, train_len=train_len, steps=steps
    )

    # --- Reconstruct and save ---
    os.makedirs(predicted_path, exist_ok=True)
    for spec in specs:
        n = sims[spec.key].get_num_components()
        predictions = sims[spec.key].reconstruct(
            hats[spec.key], n, infos[spec.key], begin=0
        )
        np.save(predicted_path / f"{spec.term}.npy", predictions)
        print(f"{spec.term} predictions saved to {predicted_path / spec.term}.npy")
