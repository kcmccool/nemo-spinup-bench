from importlib.resources import files

import numpy as np
import pytest

from nemo_spinup_forecast.dimensionality_reduction import (
    dimensionality_reduction_techniques,
)
from nemo_spinup_forecast.forecast_method import forecast_techniques
from nemo_spinup_forecast.pipeline import run_pipeline
from nemo_spinup_forecast.pipeline_utils import TermDef
from nemo_spinup_forecast.utils import get_dr_technique, get_forecast_technique

tech_cfg = files("nemo_spinup_forecast.configs").joinpath("techniques_config.yaml")
dr_technique = get_dr_technique(tech_cfg, dimensionality_reduction_techniques)
forecast_technique = get_forecast_technique(tech_cfg, forecast_techniques)

SPECS = [
    TermDef(key="ssh", term="ssh", filename="DINO_1m_To_1y_grid_T.nc"),
    TermDef(key="toce", term="toce", filename="DINO_1y_grid_T.nc"),
    TermDef(key="soce", term="soce", filename="DINO_1y_grid_T.nc"),
]

STEPS = 3


def test_run_pipeline(tmp_path):
    """run_pipeline completes without error and writes the expected output files."""
    run_pipeline(
        SPECS,
        data_path="tests/data/nemo_data_e3",
        out_dir=tmp_path,
        start=20,
        end=50,
        steps=STEPS,
        ye=True,
        comp=0.9,
        dr_technique=dr_technique,
        forecast_technique=forecast_technique,
    )

    predicted_path = tmp_path / "simu_predicted"
    for spec in SPECS:
        npy_file = predicted_path / f"{spec.term}.npy"
        assert npy_file.exists(), f"missing output: {npy_file}"
        arr = np.load(npy_file)
        assert arr.shape[0] == STEPS, (
            f"{spec.term}: expected {STEPS} time steps, got {arr.shape[0]}"
        )
        assert not np.all(np.isnan(arr)), f"{spec.term}: output is all NaN"
