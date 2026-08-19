# SPDX-License-Identifier: Apache-2.0
import os
import numpy as np
import netCDF4 as nc
import xarray as xr

# Paths based on your directory structure
data_dir = r"C:\2025\MSc CS AI\Thesis\NEMO Spin-Up Benchmark\DINO-Fusion\dino_1_4_degree_coarse_240125"
ref_nc_path = r"C:\2025\MSc CS AI\Thesis\NEMO Spin-Up Benchmark\nemo-spinup-forecast\200\DINO_1y_grid_T.nc"
out_path = r"C:\2025\MSc CS AI\Thesis\NEMO Spin-Up Benchmark\nemo-spinup-forecast\200"
output_nc_path = os.path.join(out_path, "DINO_1800_combined_grid_T.nc")

num_timesteps = 1800
y_dim, x_dim = 199, 62
depth_dim = 36

# Extract coordinate templates from reference NetCDF
print("Extracting coordinate templates from reference NetCDF...")
with xr.open_dataset(ref_nc_path) as ds_ref:
    nav_lat = ds_ref["nav_lat"].values
    nav_lon = ds_ref["nav_lon"].values
    deptht = ds_ref["deptht"].values

# Create NetCDF file and stream data directly to disk (low memory footprint)
print("Initializing NetCDF file on disk...")
with nc.Dataset(output_nc_path, "w", format="NETCDF4") as ds:
    # Define dimensions (time_counter is set to None to allow unlimited streaming appends)
    ds.createDimension("time_counter", None)
    ds.createDimension("deptht", depth_dim)
    ds.createDimension("y", y_dim)
    ds.createDimension("x", x_dim)

    # Create coordinate variables (using standard 'float64' / 'float32' strings)
    tc = ds.createVariable("time_counter", "float64", ("time_counter",))
    tc[:] = np.arange(num_timesteps, dtype=np.float64)
    tc.units = "seconds since 1900-01-01 00:00:00"

    dep = ds.createVariable("deptht", "float32", ("deptht",))
    dep[:] = deptht
    dep.units = "m"

    lat = ds.createVariable("nav_lat", "float32", ("y", "x"))
    lat[:] = nav_lat
    lat.units = "degrees_north"

    lon = ds.createVariable("nav_lon", "float32", ("y", "x"))
    lon[:] = nav_lon
    lon.units = "degrees_east"

    # Create main prognostic variables with zlib compression and chunking
    ssh = ds.createVariable("ssh", "float32", ("time_counter", "y", "x"), zlib=True, complevel=4, chunksizes=(1, y_dim, x_dim))
    ssh.units = "m"
    ssh.long_name = "Sea Surface Height"

    soce = ds.createVariable("soce", "float32", ("time_counter", "deptht", "y", "x"), zlib=True, complevel=4, chunksizes=(1, depth_dim, y_dim, x_dim))
    soce.units = "PSU"
    soce.long_name = "Practical Salinity"

    toce = ds.createVariable("toce", "float32", ("time_counter", "deptht", "y", "x"), zlib=True, complevel=4, chunksizes=(1, depth_dim, y_dim, x_dim))
    toce.units = "degC"
    toce.long_name = "Potential Temperature"

    print("Streaming .npy files directly to disk...")
    for t in range(num_timesteps):
        t_str = f"{t:05d}"
        
        ssh_file = os.path.join(data_dir, f"{t_str}.ssh.npy")
        soce_file = os.path.join(data_dir, f"{t_str}.soce.npy")
        toce_file = os.path.join(data_dir, f"{t_str}.toce.npy")
        
        if os.path.exists(ssh_file):
            ssh[t, :, :] = np.load(ssh_file)
        if os.path.exists(soce_file):
            soce[t, :, :, :] = np.load(soce_file)
        if os.path.exists(toce_file):
            toce[t, :, :, :] = np.load(toce_file)
            
        if (t + 1) % 200 == 0:
            print(f"Processed and written {t + 1} / {num_timesteps} snapshots...")

print("Successfully generated combined NetCDF dataset via streaming write!")

# =============================================================================
# STRUCTURAL COMPARISON & VERIFICATION AGAINST REFERENCE NETCDF
# =============================================================================
print("\n=============================================")
print("VERIFYING STRUCTURAL ALIGNMENT WITH REFERENCE")
print("=============================================")

with xr.open_dataset(ref_nc_path) as ds_ref, xr.open_dataset(output_nc_path) as ds_out_verify:
    print(f"• Reference time_counter length : {ds_ref.dims.get('time_counter')}")
    print(f"• Combined output time_counter length: {ds_out_verify.dims.get('time_counter')} (Expected: 1800)")
    
    for dim in ["deptht", "y", "x"]:
        assert dim in ds_out_verify.dims, f"Missing core dimension: {dim}"
        assert ds_ref.dims[dim] == ds_out_verify.dims[dim], \
            f"Dimension mismatch for {dim}! Reference: {ds_ref.dims[dim]}, Output: {ds_out_verify.dims[dim]}"
        print(f"  [PASS] Dimension '{dim}' matches (size: {ds_out_verify.dims[dim]})")

    expected_vars = {
        "ssh": (1800, 199, 62),
        "soce": (1800, 36, 199, 62),
        "toce": (1800, 36, 199, 62),
        "nav_lat": (199, 62),
        "nav_lon": (199, 62),
        "deptht": (36,)
    }
    
    for var, expected_shape in expected_vars.items():
        assert var in ds_out_verify, f"Missing expected variable/coordinate: {var}"
        actual_shape = ds_out_verify[var].shape
        assert actual_shape == expected_shape, \
            f"Shape mismatch for '{var}'! Expected {expected_shape}, got {actual_shape}"
        print(f"  [PASS] Variable '{var}' verified with shape {actual_shape}")

print("\nStructural verification complete: Output NetCDF is fully compatible with pipeline expectations!")