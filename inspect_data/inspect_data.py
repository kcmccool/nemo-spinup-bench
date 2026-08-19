import sys
import numpy as np
import xarray as xr

FILE_PATH = r"C:\2025\MSc CS AI\Thesis\NEMO Spin-Up Benchmark\nemo-spinup-forecast\200\DINO_1y_grid_T.nc"

def inspect_dataset(path):
    print("=" * 60)
    print(f"INSPECTING NETCDF DATASET: {path}")
    print("=" * 60)

    try:
        # decode_times=False prevents xarray from throwing errors on complex calendar object types
        ds = xr.open_dataset(path, decode_times=False)
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return

    print("\n--- DIMENSIONS ---")
    for dim, size in ds.sizes.items():
        print(f"  • {dim}: {size}")

    print("\n--- ALL VARIABLES & COORDINATES ---")
    for var_name, da in ds.variables.items():
        print(f"\nVariable Name: '{var_name}'")
        print(f"  • Dimensions       : {da.dims}")
        print(f"  • Shape            : {da.shape}")
        print(f"  • Data Type        : {da.dtype}")

        # Safely evaluate values and compute statistics
        try:
            data = da.values
            if not np.issubdtype(data.dtype, np.number):
                print("  • [Non-numeric data, skipping numerical stats]")
                continue

            valid_mask = np.isfinite(data)
            total_elements = data.size
            valid_elements = np.sum(valid_mask)
            
            print(f"  • Valid Points     : {valid_elements}/{total_elements} ({100 * valid_elements / total_elements:.2f}%)")
            
            if valid_elements > 0:
                print(f"  • Range (Min, Max) : [{np.nanmin(data):.4f}, {np.nanmax(data):.4f}]")
                print(f"  • Mean (μ)         : {np.nanmean(data):.4f}")
                print(f"  • Std Dev (σ)      : {np.nanstd(data):.4f}")
                
        except Exception as e:
            print(f"  • [Could not compute numerical stats: {e}]")

    ds.close()
    print("\n" + "=" * 60)

if __name__ == "__main__":
    inspect_dataset(FILE_PATH)
    