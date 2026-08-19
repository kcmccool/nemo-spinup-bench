import xarray as xr
from pathlib import Path

def evaluate_nc_compatibility(combined_path: str, reference_path: str):
    comb_p = Path(combined_path)
    ref_p = Path(reference_path)
    
    print("=" * 60)
    print("NEMO SPIN-UP FORECAST: NETCDF COMPATIBILITY AUDIT")
    print("=" * 60)
    print(f"Target File:    {comb_p.name}")
    print(f"Reference File: {ref_p.name}\n")

    if not comb_p.exists():
        print(f"CRITICAL: Target file path does not exist: {comb_p}")
        return False
    if not ref_p.exists():
        print(f"CRITICAL: Reference file path does not exist: {ref_p}")
        return False

    try:
        ds_comb = xr.open_dataset(comb_p, decode_times=False)
        ds_ref = xr.open_dataset(ref_p, decode_times=False)
    except Exception as e:
        print(f"CRITICAL: Failed to parse NetCDF datasets. Error: {e}")
        return False

    checks_passed = True

    # 1. Variable Mapping Validation based on ocean_terms.yaml configuration
    expected_variables = ["soce", "toce", "e3t", "rhop", "ssh"]
    print("[Check 1] Variable Presence Verification:")
    for var in expected_variables:
        in_comb = var in ds_comb.variables
        in_ref = var in ds_ref.variables
        status_comb = "EXISTS" if in_comb else "MISSING"
        status_ref = "EXISTS" if in_ref else "MISSING"
        print(f"  - Variable '{var}': Combined [{status_comb}] | Reference [{status_ref}]")
        if not in_comb:
            checks_passed = False

    print("\n[Check 2] Dimension Structure & Shape Compatibility:")
    # Identify primary data variable to check dimensions
    sample_var = next((v for v in expected_variables if v in ds_comb.variables), None)
    if not sample_var:
        print("CRITICAL: No valid target ocean variables found in the combined dataset.")
        return False

    da_comb = ds_comb[sample_var]
    da_ref = ds_ref[sample_var]

    print(f"  - Evaluated variable for shape analysis: '{sample_var}'")
    print(f"  - Combined dims: {dict(da_comb.sizes)}")
    print(f"  - Reference dims: {dict(da_ref.sizes)}")

    # 2. Time Dimension Check (Pipeline expects 'time_counter')
    time_dim_name = "time_counter"
    if time_dim_name not in da_comb.dims:
        # Check standard fallbacks
        alt_time = next((d for d in da_comb.dims if "time" in d.lower()), None)
        if alt_time:
            print(f"  - WARNING: Time dimension is named '{alt_time}' instead of '{time_dim_name}' [Pipeline requires renaming or config adjustment]")
        else:
            print(f"  - ERROR: No time dimension identified in combined dataset.")
            checks_passed = False
    else:
        print(f"  - Time dimension '{time_dim_name}' correctly present (Length: {da_comb.sizes[time_dim_name]})")

    # 3. Spatial Dimensions Check (y, x, and optional z depth)
    for spatial_dim in ["y", "x"]:
        if spatial_dim not in da_comb.dims:
            print(f"  - ERROR: Missing spatial dimension '{spatial_dim}'")
            checks_passed = False
        else:
            match_status = "MATCH" if da_comb.sizes[spatial_dim] == da_ref.sizes.get(spatial_dim) else "MISMATCH (Differs from reference grid)"
            print(f"  - Spatial dimension '{spatial_dim}': Size {da_comb.sizes[spatial_dim]} [{match_status}]")

    depth_dim = next((d for d in ["deptht", "olevel"] if d in da_comb.dims), None)
    if depth_dim:
        print(f"  - Vertical depth dimension '{depth_dim}' detected (Size: {da_comb.sizes[depth_dim]}) [3D Configuration]")
    else:
        print(f"  - Info: No vertical depth dimension detected [2D Surface Configuration]")

    # 4. Chunking Compatibility Check against Simulation.get_attributes()
    print("\n[Check 3] Pipeline Chunking & Dask Compatibility:")
    try:
        # Test lazy chunk evaluation matching forecast.py settings
        test_chunks = {"time": 200, "x": 120}
        ds_chunked = xr.open_dataset(comb_p, decode_times=False, chunks=test_chunks)
        print("  - Chunking test with `chunks={'time': 200, 'x': 120}` passed successfully.")
    except Exception as e:
        print(f"  - WARNING: Chunk initialization failed or threw warnings: {e}")

    print("\n" + "=" * 60)
    if checks_passed:
        print("RESULT: SUCCESS. The combined NetCDF file is structurally compatible with the pipeline.")
    else:
        print("RESULT: ACTION REQUIRED. Address the dimension or variable errors noted above before pipeline execution.")
    print("=" * 60)
    return checks_passed

# Run execution
evaluate_nc_compatibility(
    r"C:\2025\MSc CS AI\Thesis\NEMO Spin-Up Benchmark\nemo-spinup-forecast\200\DINO_1800_combined_grid_T.nc",
    r"C:\2025\MSc CS AI\Thesis\NEMO Spin-Up Benchmark\nemo-spinup-forecast\200\DINO_1y_grid_T.nc"
)