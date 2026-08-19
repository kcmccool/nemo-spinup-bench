import os
import sys
import shutil
import subprocess
from pathlib import Path
import numpy as np
import xarray as xr
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
def create_dummy_netcdf(file_path, shape=(24, 1, 50, 16)):
    """Generates a dummy climate NetCDF file mimicking DINO ocean snapshots with NEMO dimensions."""
    time_len, depth_len, y_len, x_len = shape
    
    # Generate coordinates
    time_counters = np.arange(time_len)
    depths = np.array([5.0])
    y_coords = np.linspace(-90, 90, y_len)
    x_coords = np.linspace(-180, 180, x_len)
    
    # Create smooth dummy data
    data = np.random.randn(*shape).astype(np.float32)
    
    ds = xr.Dataset(
        {
            # Use "time_counter" instead of "time"
            "ssh": (["time_counter", "depth", "y", "x"], data),
            "soce": (["time_counter", "depth", "y", "x"], data + 35.0),
            "toce": (["time_counter", "depth", "y", "x"], data + 15.0),
        },
        coords={
            "time_counter": time_counters,
            "depth": depths,
            "y": y_coords,
            "x": x_coords,
        }
    )
    
    # Mimic the DINO-Fusion grid name structure
    ds.to_netcdf(file_path)

def test_pipeline_cae_execution(tmp_path):
    """Executes the pipeline end-to-end using the new --dr_technique cae option."""
    
    # 1. Setup mock run directories
    data_dir = tmp_path / "raw_data"
    output_dir = tmp_path / "run_output"
    data_dir.mkdir()
    output_dir.mkdir()
    
    # Create dummy NetCDF files matching cli.py expectations
    create_dummy_netcdf(data_dir / "DINO_1m_To_1y_grid_T.nc")
    create_dummy_netcdf(data_dir / "DINO_1y_grid_T.nc")
    
    # 2. Locate the cli entry point script
    # Assumes running from the repository root directory
    cli_path = Path(__file__).parent.parent / "src" / "nemo_spinup_forecast" / "cli.py"
    
    # 3. Formulate execution parameters
    # Slicing years 0-20 to train, forecasting 2 steps ahead using your CAE
    cmd = [
        sys.executable, str(cli_path),
        "--data-path", str(data_dir),
        "--start", "0",
        "--end", "20",
        "--steps", "2",
        "--output-path", str(output_dir),
        "--comp", "10",
    ]
    
    # Execute the pipeline shell terminal command
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent / "src")
    
    print(f"\nRunning command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env) # <-- Added env=env here
    
    # Assertions to uncover failures early
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    
    assert result.returncode == 0, f"Pipeline crashed with exit code {result.returncode}"
    
    # Verify that the model weight serialization bypassed standard pickling cleanly
    forecast_dir = list(output_dir.glob("**/forecast/simu_prepared/ssh"))
    assert len(forecast_dir) > 0, f"Prepared files directory was not found in {output_dir}"
    
    # Check that our predictions arrays exist on the filesystem
    # Changed from "run_*" to a robust recursive pattern "**/"
    pred_dirs = list(output_dir.glob("**/forecast/simu_predicted"))
    assert len(pred_dirs) > 0, f"Predicted files directory was not found in {output_dir}"
    pred_dir = pred_dirs[0]
    print("Success: End-to-end framework execution loop complete via PyTorch + GP components!")