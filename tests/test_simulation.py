import os
import sys
import pytest
import numpy as np
import xarray as xr
import torch
from pathlib import Path

# Ensure execution context includes the local source path definitions
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nemo_spinup_forecast.dimensionality_reduction import DimensionalityReductionCVAE

def create_mock_ocean_netcdf(file_path, shape, variable_name):
    """ Generates a dummy NetCDF file mimicking NEMO DINO tracking structures.
        Dynamically builds either 2D (time, y, x) or 3D (time, depth, y, x) layouts. """
    
    if len(shape) == 4:
        time_len, depth_len, y_len, x_len = shape
        dims = ("time_counter", "depth", "y", "x")
        coords = {
            "time_counter": np.arange(time_len),
            "depth": np.linspace(0, 500, depth_len),
            "y": np.linspace(-90, 90, y_len),
            "x": np.linspace(-180, 180, x_len),
        }
    else:
        time_len, y_len, x_len = shape
        dims = ("time_counter", "y", "x")
        coords = {
            "time_counter": np.arange(time_len),
            "y": np.linspace(-90, 90, y_len),
            "x": np.linspace(-180, 180, x_len),
        }

    # Generate random matrix space and punch out land mask holes (NaNs)
    raw_data = np.random.randn(*shape).astype(np.float32)
    
    # Introduce a deterministic island mask block to test NaN preservation
    if len(shape) == 4:
        raw_data[:, :, :5, :5] = np.nan
    else:
        raw_data[:, :5, :5] = np.nan

    ds = xr.Dataset(
        {variable_name: (dims, raw_data)},
        coords=coords
    )
    ds.to_netcdf(file_path)


@pytest.mark.parametrize(
    "var_name, data_shape, expected_channels",
    [
        ("ssh", (12, 40, 30), 1),        # 2D surface configuration layout
        ("toce", (12, 36, 40, 30), 36),  # 3D full profile volumetric layout
    ]
)
def test_cvae_decomposition_lifecycle(tmp_path, var_name, data_shape, expected_channels):
    """ Validates the CVAE interface end-to-end: dimensional parsing, 
        training, latent encoding extraction, and masked reconstructions. """
    
    test_file = tmp_path / f"test_{var_name}.nc"
    create_mock_ocean_netcdf(test_file, data_shape, var_name)
    
    # 1. Read standard NetCDF and extract clean arrays matching forecast.py execution
    ds = xr.open_dataset(test_file)
    raw_array = ds[var_name].values
    
    # Instantiate the target CVAE reducer 
    latent_dim = 8
    epochs = 2
    batch_size = 4
    
    cvae_reducer = DimensionalityReductionCVAE(
        comp=latent_dim, 
        epochs=epochs, 
        batch_size=batch_size, 
        lr=1e-3, 
        beta=0.01
    )
    
    # 2. Execute full forward optimization and component mapping decomposition
    components, model, mask = cvae_reducer.decompose(
        simulation_array=raw_array, 
        length=len(raw_array), 
        info_desc={"mean": 0.0, "std": 1.0}
    )
    
    # --- Structural Assertions ---
    assert components.shape == (len(raw_array), latent_dim), "Latent feature array dimensions are skewed"
    assert cvae_reducer.model.in_channels == expected_channels, "Channel allocations mapped incorrectly"
    assert mask.shape == raw_array.shape[1:], "Spatial logical validation mask mismatch"
    
    # 3. Verify Reconstruction Fidelity & Land-Mask Recovery
    reconstructed = cvae_reducer.reconstruct_components(n=latent_dim)
    
    assert reconstructed.shape == raw_array.shape, "Reconstruction tensor dimensions do not match the input data shape"
    
    # Assert that masked land locations safely return as NaNs in the final space
    if len(data_shape) == 4:
        assert np.isnan(reconstructed[0, 0, 0, 0]), "Failed to restore standard NaN boundaries on 3D layout"
        assert not np.isnan(reconstructed[0, 0, 10, 10]), "Fluid coordinates overwritten by mask bounds unexpectedly"
    else:
        assert np.isnan(reconstructed[0, 0, 0]), "Failed to restore standard NaN boundaries on 2D layout"


def test_cvae_loss_balance():
    """ Tests that the internal VAE loss calculation computes correctly 
        and that changing beta scales the KL regularizer. """
    
    # Initialize bare mock layers matching our class setup
    cvae_reducer = DimensionalityReductionCVAE(comp=4, beta=1.0)
    
    # Mock PyTorch variables: Batch=2, Channels=1, Y=10, X=10
    recon = torch.zeros(2, 1, 10, 10, requires_grad=True)
    target = torch.zeros(2, 1, 10, 10)
    
    # Explicit distributions: mean=0, logvar=0
    mu = torch.zeros(2, 4)
    logvar = torch.zeros(2, 4)
    
    # Evaluate baseline loss: perfect reconstruction, zero distribution divergence
    loss, recon_loss, kl_loss = cvae_reducer.vae_loss_function(recon, target, mu, logvar)
    
    assert loss.item() == 0.0, "Perfect convergence should output zero total error"
    assert recon_loss.item() == 0.0
    assert kl_loss.item() == 0.0
    
    # Introduce an offset to test divergence tracking: shift mu away from the prior normal distribution
    mu_skewed = torch.ones(2, 4) * 2.0
    cvae_reducer.beta = 0.5
    
    _, _, kl_loss_skewed = cvae_reducer.vae_loss_function(recon, target, mu_skewed, logvar)
    assert kl_loss_skewed.item() > 0.0, "Divergent distributions must return positive KL scores"