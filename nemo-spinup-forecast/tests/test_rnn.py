import os
import sys
import shutil
from pathlib import Path
import numpy as np
import xarray as xr
import yaml
import pytest
import torch
import glob
from nemo_spinup_forecast.forecast import Simulation

# Ensure 'src' directory is in Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Local path to actual dataset file
REAL_DATA_PATH = PROJECT_ROOT / "200" / "DINO_1800_combined_grid_T.nc"

from nemo_spinup_forecast.cli import main as cli_main
from nemo_spinup_forecast.dimensionality_reduction import ConvVAE, DimensionalityReductionCVAE
from nemo_spinup_forecast.forecast_method import RNNForecaster


# ============================================================================
# PyTest Fixtures (Enforcing Real Data Usage)
# ============================================================================

@pytest.fixture(scope="session")
def nemo_sliced_dataset_real(tmp_path_factory):
    """
    Provides a fast, CPU-friendly NetCDF dataset sliced directly from the 
    ACTUAL NEMO dataset (DINO_1y_grid_T.nc), incorporating all 4 physical variables 
    (toce, soce, e3t, rhop). Fails if real data is missing.
    """
    if not REAL_DATA_PATH.exists():
        pytest.fail(f"Real dataset not found at expected path: {REAL_DATA_PATH}. Aborting real-data test.")

    tmp_dir = tmp_path_factory.mktemp("real_data")
    base_file = tmp_dir / "200" / "DINO_1800_combined_grid_T.nc"
    base_file.parent.mkdir(parents=True, exist_ok=True)

    # Slice actual dataset to keep tests lightweight and CPU-friendly
    with xr.open_dataset(REAL_DATA_PATH) as ds:
        ds_sliced = ds.isel(
            time_counter=slice(0, 20),  # Enough frames for sequence and forecasting
            deptht=slice(0, 2),
            y=slice(0, 30),
            x=slice(0, 30),
        )
        ds_sliced.to_netcdf(base_file)

    # Expanded alias list ensuring file lookup passes cleanly across directory structures
    alias_filenames = [
        "DINO_1y_grid_T.nc",
        "DINO_1y_grid_T",
        "grid_T.nc",
        "grid_T",
        "200_DINO_1y_grid_T.nc",
        "0_DINO_1y_grid_T.nc",
        "1_DINO_1y_grid_T.nc",
    ]

    def populate_directory(target_dir: Path):
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname in alias_filenames:
            dest = target_dir / fname
            if not dest.exists():
                shutil.copy(base_file, dest)

    populate_directory(tmp_dir)
    populate_directory(tmp_dir / "200")

    for year in range(0, 7):
        populate_directory(tmp_dir / str(year))
        populate_directory(tmp_dir / f"{year:04d}")
        populate_directory(tmp_dir / f"{year:02d}")

    return tmp_dir


@pytest.fixture
def rnn_pipeline_configs(tmp_path):
    """
    Generates configuration files mapping the 4 physical variables 
    and setting RNNForecaster as the active forecasting technique.
    """
    # 1. Terms configuration file
    terms_path = tmp_path / "ocean_terms.yaml"
    terms_content = {
        "Terms": {
            "Temperature": "toce",
            "Salinity": "soce",
            "VerticalThickness": "e3t",
            "PotentialDensity": "rhop"
        },
        "terms": {
            "Temperature": "toce",
            "Salinity": "soce",
            "VerticalThickness": "e3t",
            "PotentialDensity": "rhop"
        }
    }
    with open(terms_path, "w") as f:
        yaml.dump(terms_content, f)

    # 2. Techniques configuration file configured for RNN
    config_path = tmp_path / "techniques_config.yaml"
    config_content = {
        "DR_technique": {
            "name": "cvae",
            "epochs": 2,
            "batch_size": 4
        },
        "Forecast_technique": {
            "name": "RNNForecaster",
            "epochs": 5,
            "hidden_dim": 16,
            "seq_len": 4,
            "cell_type": "lstm"
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
        
    return config_path


# ============================================================================
# Unit Tests for RNN Component
# ============================================================================

def test_rnn_forecaster_multivariate_fit():
    """Verifies RNNForecaster training and autoregressive multi-step prediction on CPU."""
    train_len, pred_len, num_components = 12, 3, 4
    x_train = np.arange(train_len)
    x_pred = np.arange(train_len, train_len + pred_len)

    np.random.seed(42)
    y_train = np.random.randn(train_len, num_components).astype(np.float32)

    forecaster = RNNForecaster(hidden_dim=16, seq_len=4, epochs=5, lr=1e-2, device="cpu")
    y_hat, y_hat_std = forecaster.apply_forecast(y_train, x_train, x_pred)

    assert y_hat.shape == (pred_len, num_components)
    assert not np.isnan(y_hat).any()
    assert y_hat_std.shape == (pred_len, num_components)


# ============================================================================
# End-to-End Integration Test with Real Data
# ============================================================================

def test_cli_rnn_end_to_end_real_data(tmp_path, nemo_sliced_dataset_real, rnn_pipeline_configs, monkeypatch):
    """
    Executes the entire pipeline end-to-end using the actual sliced NEMO dataset, 
    ConvVAE dimensionality reduction, and the RNNForecaster.
    """
    def mock_get_data(path, term, filename):
        files = sorted(glob.glob(os.path.join(path, "*.nc")))
        if not files:
            files = sorted(glob.glob(os.path.join(path, "**", "*.nc"), recursive=True))
        return files

    monkeypatch.setattr(Simulation, "get_data", staticmethod(mock_get_data))
    
    # Run test inside tmp_path so relative configuration lookups succeed
    monkeypatch.chdir(tmp_path)

    output_dir = tmp_path / "run_output_rnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_steps = 2
    args = [
        "cli.py",
        "--data-path", str(nemo_sliced_dataset_real),
        "--start", "0",
        "--end", "6",
        "--steps", str(forecast_steps),
        "--output-path", str(output_dir),
        "--comp", "4",
        "--techniques-config", str(rnn_pipeline_configs),
        "--ye", "True",
    ]

    monkeypatch.setattr(sys, "argv", args)

    try:
        cli_main()
    except SystemExit as e:
        assert e.code == 0, f"CLI exited with error code {e.code}"

    # Verify that prediction outputs were successfully written to disk
    output_npy_files = list(output_dir.glob("**/*.npy"))
    assert len(output_npy_files) > 0, "No forecast files generated for RNN pipeline"

    # Inspect the prediction shapes and values
    for npy_file in output_npy_files:
        pred_data = np.load(npy_file)
        assert pred_data.shape[0] == forecast_steps
        assert not np.isnan(pred_data).all()