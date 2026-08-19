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
from nemo_spinup_forecast.forecast_method import FNOForecaster


# ============================================================================
# PyTest Fixtures (Enforcing Strict Real Data Auditing with Lazy Dask Chunks)
# ============================================================================

@pytest.fixture(scope="session")
def nemo_sliced_dataset_real(tmp_path_factory):
    """
    Provides a fast, CPU-friendly NetCDF dataset sliced directly from the 
    ACTUAL NEMO combined dataset (DINO_1800_combined_grid_T.nc).
    
    Uses Dask chunking to audit large arrays safely without memory overflow.
    """
    if not REAL_DATA_PATH.exists():
        pytest.fail(f"CRITICAL: Real dataset not found at expected path: {REAL_DATA_PATH}. Aborting test.")

    print(f"\n[AUDIT START] Opening real dataset with Dask chunking at: {REAL_DATA_PATH}")
    
    try:
        # Enable chunks so large variables don't crash RAM during audits
        ds = xr.open_dataset(REAL_DATA_PATH, decode_times=False, chunks={"time_counter": 100})
    except Exception as e:
        pytest.fail(f"CRITICAL: Failed to parse real NetCDF file '{REAL_DATA_PATH}'. Error: {e}")

    # 1. Print DataFrame Head Preview for variables (using sliced point for safety)
    print("\n" + "="*60)
    print("DATASET DATAFRAME PREVIEW (First 5 time entries at y=0, x=0)")
    print("="*60)
    for var in ds.data_vars:
        try:
            if "deptht" in ds[var].dims:
                df_sample = ds[var].isel(deptht=0, y=0, x=0).to_dataframe()
            elif "y" in ds[var].dims and "x" in ds[var].dims:
                df_sample = ds[var].isel(y=0, x=0).to_dataframe()
            else:
                df_sample = ds[var].to_dataframe()
            print(f"\nVariable: '{var}'")
            print(df_sample.head(5))
        except Exception as ex:
            print(f"Could not preview variable '{var}' as DataFrame: {ex}")

    # 2. NaN and Infinite Value Audit (Memory-Safe via Dask)
    print("\n" + "="*60)
    print("AUDIT: NaN & MISSING VALUE CHECK")
    print("="*60)
    for var in ds.data_vars:
        var_data = ds[var]
        nan_count = var_data.isnull().sum().compute().item()
        total_elements = var_data.size
        nan_pct = (nan_count / total_elements) * 100 if total_elements > 0 else 0
        print(f"Variable '{var}': {nan_count} NaNs out of {total_elements} elements ({nan_pct:.2f}%)")
        if nan_pct > 50.0:
            pytest.fail(f"CRITICAL: Variable '{var}' has {nan_pct:.2f}% missing values, exceeding allowable limits.")

    # 3. Temporal Monotonicity Check
    print("\n" + "="*60)
    print("AUDIT: TEMPORAL MONOTONICITY CHECK")
    print("="*60)
    time_dim = "time_counter" if "time_counter" in ds.dims else next((d for d in ds.dims if "time" in d.lower()), None)
    if time_dim:
        time_vals = ds[time_dim].values
        diffs = np.diff(time_vals)
        is_monotonic = np.all(diffs > 0)
        print(f"Time dimension '{time_dim}' length: {len(time_vals)}")
        print(f"Strictly monotonic increase: {is_monotonic}")
        if not is_monotonic:
            pytest.fail("CRITICAL: Time coordinate entries are non-monotonic or contain duplicate indices.")
    else:
        print("WARNING: Time dimension could not be verified.")

    # 4. Physical Bounds & Plausibility Check (Memory-Safe via Dask)
    print("\n" + "="*60)
    print("AUDIT: PHYSICAL BOUNDS CHECK")
    print("="*60)
    for var in ds.data_vars:
        try:
            vmin = float(ds[var].min().compute())
            vmax = float(ds[var].max().compute())
            vmean = float(ds[var].mean().compute())
            print(f"[{var}] Min: {vmin:.4f} | Max: {vmax:.4f} | Mean: {vmean:.4f}")
        except Exception as e:
            print(f"Could not compute statistics for '{var}': {e}")

    # Slice actual dataset to keep tests lightweight and CPU-friendly
    tmp_dir = tmp_path_factory.mktemp("real_data_fno")
    base_file = tmp_dir / "200" / "DINO_1y_grid_T.nc"
    base_file.parent.mkdir(parents=True, exist_ok=True)

    ds_sliced = ds.isel(
        time_counter=slice(0, 20),  # Enough frames for sequence and forecasting
        deptht=slice(0, 2),
        y=slice(0, 30),
        x=slice(0, 30),
    )
    ds_sliced.to_netcdf(base_file)
    ds.close()

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
def fno_pipeline_configs(tmp_path):
    """
    Generates configuration files mapping physical variables 
    and setting FNOForecaster as the active forecasting technique.
    """
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

    config_path = tmp_path / "techniques_config.yaml"
    config_content = {
        "DR_technique": {
            "name": "cvae",
            "epochs": 2,
            "batch_size": 4
        },
        "Forecast_technique": {
            "name": "FNOForecaster",
            "epochs": 5,
            "hidden_dim": 16,
            "num_modes": 4,
            "num_layers": 2,
            "seq_len": 6
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
        
    return config_path


# ============================================================================
# Unit Tests for FNO Component
# ============================================================================

def test_fno_forecaster_multivariate_fit():
    """Verifies FNOForecaster training and spectral multi-step prediction on CPU."""
    train_len, pred_len, num_components = 14, 3, 4
    x_train = np.arange(train_len)
    x_pred = np.arange(train_len, train_len + pred_len)

    np.random.seed(42)
    y_train = np.random.randn(train_len, num_components).astype(np.float32)

    forecaster = FNOForecaster(hidden_dim=16, num_modes=4, num_layers=2, seq_len=6, epochs=5, lr=1e-2, device="cpu")
    y_hat, y_hat_std = forecaster.apply_forecast(y_train, x_train, x_pred)

    assert y_hat.shape == (pred_len, num_components)
    assert not np.isnan(y_hat).any()
    assert y_hat_std.shape == (pred_len, num_components)


# ============================================================================
# End-to-End Integration Test with Real Data
# ============================================================================

def test_cli_fno_end_to_end_real_data(tmp_path, nemo_sliced_dataset_real, fno_pipeline_configs, monkeypatch):
    """
    Executes the entire pipeline end-to-end using the actual NEMO dataset, 
    ConvVAE dimensionality reduction, and the FNOForecaster. Fails if real data fails.
    """
    def mock_get_data(path, term, filename):
        files = sorted(glob.glob(os.path.join(path, "*.nc")))
        if not files:
            files = sorted(glob.glob(os.path.join(path, "**", "*.nc"), recursive=True))
        return files

    monkeypatch.setattr(Simulation, "get_data", staticmethod(mock_get_data))
    
    monkeypatch.chdir(tmp_path)

    output_dir = tmp_path / "run_output_fno"
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
        "--techniques-config", str(fno_pipeline_configs),
        "--ye", "True",
    ]

    monkeypatch.setattr(sys, "argv", args)

    try:
        cli_main()
    except SystemExit as e:
        assert e.code == 0, f"CLI exited with error code {e.code}"

    output_npy_files = list(output_dir.glob("**/*.npy"))
    assert len(output_npy_files) > 0, "No forecast files generated for FNO pipeline"

    for npy_file in output_npy_files:
        pred_data = np.load(npy_file)
        assert pred_data.shape[0] == forecast_steps
        assert not np.isnan(pred_data).all()