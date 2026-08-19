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
from nemo_spinup_forecast.forecast_method import NeuralODEForecaster


# ============================================================================
# PyTest Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def nemo_sliced_dataset(tmp_path_factory):
    """
    Provides a fast, CPU-friendly NetCDF dataset across NEMO spin-up directories 
    (0 through 6, plus 200) containing all 4 physical variables (toce, soce, e3t, rhop).
    """
    tmp_dir = tmp_path_factory.mktemp("data")
    base_file = tmp_dir / "200" / "DINO_1800_combined_grid_T.nc"
    base_file.parent.mkdir(parents=True, exist_ok=True)

    if REAL_DATA_PATH.exists():
        # Slice actual dataset (10 time steps, 2 depths, 30x30 spatial grid)
        with xr.open_dataset(REAL_DATA_PATH) as ds:
            ds_sliced = ds.isel(
                time_counter=slice(0, 10),
                deptht=slice(0, 2),
                y=slice(0, 30),
                x=slice(0, 30),
            )
            ds_sliced.to_netcdf(base_file)
    else:
        # Synthetic fallback matching real schema with all 4 variables
        time_len, depth_len, y_len, x_len = 10, 2, 30, 30
        shape = (time_len, depth_len, y_len, x_len)
        np.random.seed(42)
        dummy_data = np.random.randn(*shape).astype(np.float32)

        ds = xr.Dataset(
            {
                "toce": (["time_counter", "deptht", "y", "x"], dummy_data + 15.0),
                "soce": (["time_counter", "deptht", "y", "x"], dummy_data + 35.0),
                "e3t":  (["time_counter", "deptht", "y", "x"], dummy_data + 100.0),
                "rhop": (["time_counter", "deptht", "y", "x"], dummy_data + 1025.0),
            },
            coords={
                "time_counter": np.arange(time_len),
                "deptht": np.array([5.0, 15.0]),
                "y": np.arange(y_len),
                "x": np.arange(x_len),
            },
        )
        ds.to_netcdf(base_file)

    # Expanded alias list ensuring file lookup passes cleanly
    alias_filenames = [
        "DINO_1y_grid_T.nc",
        "DINO_1y_grid_T",
        "grid_T.nc",
        "grid_T",
        "DINO_1y_grid_U.nc",
        "grid_U.nc",
        "grid_U",
        "DINO_1y_grid_V.nc",
        "grid_V.nc",
        "grid_V",
        "DINO_1y_grid_W.nc",
        "grid_W.nc",
        "grid_W",
        "toce.nc",
        "soce.nc",
        "e3t.nc",
        "rhop.nc",
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

    # Populate root, 200, and yearly subdirectories
    populate_directory(tmp_dir)
    populate_directory(tmp_dir / "200")

    for year in range(0, 7):
        populate_directory(tmp_dir / str(year))
        populate_directory(tmp_dir / f"{year:04d}")
        populate_directory(tmp_dir / f"{year:02d}")

    return tmp_dir


@pytest.fixture
def pipeline_configs(tmp_path):
    """
    Generates both 'ocean_terms.yaml' and 'techniques_config.yaml' in the test directory
    to ensure the pipeline correctly maps all 4 physical variables without referencing 'SSH'.
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

    # 2. Techniques configuration file
    config_path = tmp_path / "techniques_config.yaml"
    config_content = {
        "DR_technique": {
            "name": "cvae",
            "epochs": 2,
            "batch_size": 4
        },
        "Forecast_technique": {
            "name": "NeuralODEForecaster",
            "epochs": 5,
            "hidden_dim": 16
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
        
    return config_path


# ============================================================================
# Unit Tests
# ============================================================================

def test_convvae_architecture_spatial_alignment():
    """Verifies ConvVAE input/output tensor spatial dimension alignment on CPU."""
    batch_size = 2
    in_channels = 4  # Matches our 4 variables (toce, soce, e3t, rhop)
    spatial_shape = (30, 30)
    latent_dim = 4

    model = ConvVAE(latent_dim=latent_dim, in_channels=in_channels, in_shape=spatial_shape)
    model.eval()

    x_dummy = torch.randn(batch_size, in_channels, *spatial_shape)
    with torch.no_grad():
        recon, mu, logvar = model(x_dummy)

    assert recon.shape == x_dummy.shape
    assert mu.shape == (batch_size, latent_dim)


def test_neural_ode_forecaster_multivariate_fit():
    """Verifies Neural ODE fitting on small latent space on CPU."""
    train_len, pred_len, num_components = 8, 2, 4
    x_train = np.arange(train_len)
    x_pred = np.arange(train_len, train_len + pred_len)

    np.random.seed(42)
    y_train = np.random.randn(train_len, num_components).astype(np.float32)

    forecaster = NeuralODEForecaster(hidden_dim=8, epochs=5, lr=1e-2, device="cpu")
    y_hat, y_hat_std = forecaster.apply_forecast(y_train, x_train, x_pred)

    assert y_hat.shape == (pred_len, num_components)
    assert not np.isnan(y_hat).any()


def test_cvae_decomposition_and_weights_io(tmp_path):
    """Tests CVAE decomposition and checkpointing on CPU with 4 input channels."""
    time_steps, depth_channels, height, width = 8, 2, 30, 30
    sim_data = np.random.randn(time_steps, depth_channels, height, width).astype(np.float32)

    cvae = DimensionalityReductionCVAE(comp=4, epochs=2, batch_size=4, lr=1e-3)
    components, model, mask = cvae.decompose(sim_data, length=time_steps)

    assert components.shape == (time_steps, 4)

    save_dir = tmp_path / "model_weights"
    save_dir.mkdir()
    cvae.save_weights(str(save_dir))

    new_cvae = DimensionalityReductionCVAE(comp=4)
    new_cvae.shape = sim_data.shape[1:]
    new_cvae.load_weights(str(save_dir))
    assert new_cvae.model is not None


# ============================================================================
# End-to-End Integration Test
# ============================================================================

def test_cli_end_to_end_pipeline_cpu(tmp_path, nemo_sliced_dataset, pipeline_configs, monkeypatch):
    """
    Executes the pipeline end-to-end against sliced NEMO data containing all variables.
    """
    def mock_get_data(path, term, filename):
        files = sorted(glob.glob(os.path.join(path, "*.nc")))
        if not files:
            files = sorted(glob.glob(os.path.join(path, "**", "*.nc"), recursive=True))
        return files

    monkeypatch.setattr(Simulation, "get_data", staticmethod(mock_get_data))
    
    # Run test inside tmp_path so relative config lookups find 'ocean_terms.yaml'
    monkeypatch.chdir(tmp_path)

    output_dir = tmp_path / "run_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_steps = 2
    args = [
        "cli.py",
        "--data-path", str(nemo_sliced_dataset),
        "--start", "0",
        "--end", "6",
        "--steps", str(forecast_steps),
        "--output-path", str(output_dir),
        "--comp", "4",
        "--techniques-config", str(pipeline_configs),
        "--ye", "True",
    ]

    monkeypatch.setattr(sys, "argv", args)

    try:
        cli_main()
    except SystemExit as e:
        assert e.code == 0, f"CLI exited with error code {e.code}"

    # Output verification
    output_npy_files = list(output_dir.glob("**/*.npy"))
    assert len(output_npy_files) > 0, "No forecast files generated"

    # Verify the contents and shapes of the prediction files
    for npy_file in output_npy_files:
        pred_data = np.load(npy_file)
        # Check that the time dimension matches forecast steps
        assert pred_data.shape[0] == forecast_steps
        # Ensure values are valid and not all NaNs
        assert not np.isnan(pred_data).all()