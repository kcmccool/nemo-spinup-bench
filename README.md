# NEMO Spin-Up Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19474414.svg)](https://doi.org/10.5281/zenodo.19474414)

Reproducible research artifact for the NEMO spin-up acceleration method. This repository
contains the full end-to-end workflow as a citable, versioned snapshot.

---

## Repositories

| Submodule                                                                     | Description                              |
| ----------------------------------------------------------------------------- | ---------------------------------------- |
| [`nemo-spinup-forecast`](https://github.com/m2lines/nemo-spinup-forecast)     | Dimensionality reduction and forecasting |
| [`nemo-spinup-restart`](https://github.com/m2lines/nemo-spinup-restart)       | Restart file generation                  |
| [`nemo-spinup-evaluation`](https://github.com/m2lines/nemo-spinup-evaluation) | Evaluation and validation                |

---

## Reference Data

Reference data (DINO output, restart files, `mesh_mask.nc`) is archived on Zenodo:

> **Zenodo DOI:** [10.5281/zenodo.19474414](https://doi.org/10.5281/zenodo.19474414)

---

## Installation of Nemo-Spinup-Bench dependencies 

This installs all three nemo-spinup-{forecast, restart, evaluation} packages in a single virtual environment. 

1. **Clone this repository with submodules**

   ```bash
   git clone --recurse-submodules https://github.com/m2lines/nemo-spinup-bench.git
   cd nemo-spinup-bench
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install ./nemo-spinup-{forecast,restart,evaluation}
   ```


## Benchmark end-to-end steps

This describes the complete end-to-end pipeline to run the benchmark. We omit details like building and compiling NEMO/DINO.

> The entire pipeline assumes NEMO 4.2.0 and a completed cold-start NEMO run, i.e. output files, restart files, and a `mesh_mask.nc` are available before starting.
>
> The commands below assume reference data is downloaded to `data/50/`. Substitute this with your own data directory if not using the reference data.

### A. Data preparation

1. **Get simulation data**

   The entire benchmark will run using sample data hosted on Zenodo. Alternatively you may run NEMO/DINO yourself; we recommend running for at least 50–100 years. The Zenodo data contains 50 years of DINO output files to train on. 

   **Download data from Zenodo**

   Download `50.zip` from [Zenodo record 19474414](https://zenodo.org/records/19474414), unzip it, and place the contents in `data/50/`. The record also provides `200.zip` (200 years of DINO output) and `restart.zip` (200 annual restart files) for extended experiments.

   ```bash
   mkdir -p data
   curl -L -o 50.zip https://zenodo.org/records/19474414/files/50.zip
   unzip 50.zip -d data/
   ```

2. **(Optional) Combine restart files and mesh mask** using [REBUILD_NEMO](https://forge.nemo-ocean.eu/nemo/nemo/-/tree/4.2.0/tools/REBUILD_NEMO):

   This step is only required if you are using your own NEMO run. The Zenodo reference data already includes combined files. You can use the same module environment used to run NEMO/DINO to compile `rebuild_nemo`.

   ```bash
   ./rebuild_nemo -n ./nam_rebuild data/50/DINO_00576000_restart 36
   ./rebuild_nemo -n ./nam_rebuild data/50/mesh_mask 36
   ```

3. **(Optional) Resample data**

   This step is not required when using the Zenodo reference data, which already contains the resampled file `DINO_1m_to_1y_grid_T.nc`.

   All data must be temporally aligned before forecasting. If you are bringing your own NEMO output, convert monthly SSH to annual using [`cdo`](https://code.mpimet.mpg.de/projects/cdo):

   ```bash
   cdo yearmean DINO_1m_grid_T.nc DINO_1m_To_1y_grid_T.nc
   ```

   Temperature and salinity (3-D) are already annual (`DINO_1y_grid_T.nc`).

   If more training data is needed, concatenate monthly outputs `*grid_T.nc` with `ncrcat`, part of the [NCO (netCDF Operators)](https://nco.sourceforge.net/).

---

### B. Spin-up acceleration

The spin-up acceleration pipeline forecasts the ocean state forward in time using dimensionality reduction and Gaussian process regression, generates updated restart files, and evaluates the result against a reference numerical run. We begin with a baseline evaluation of the reference simulation so that the final evaluation can be compared against it.

4. **Establish a baseline evaluation** of the cold-start reference simulation:

   ```bash
   nemo-spinup-evaluation \
     --sim-path data/50 \
     --config configs/DINO-evaluation.yaml \
     --results-dir output \
     --result-file-prefix baseline \
     --mode both
   ```

   Results are written to `output/baseline_restart.csv` and `output/baseline_grid.csv`.

5. **Create the projected state**

   The default technique is PCA for dimensionality reduction with Gaussian process regression for forecasting. The key parameters to adjust are `--start` and `--steps`: `--start` controls how many years of spin-up are used for training (here 30 with 20 years thrown away), and `--steps` controls how many years are skipped forward (here 30). Increasing `--steps` gives a larger acceleration but may reduce accuracy.

   ```bash
   nemo-spinup-forecast \
     --ye True \
     --start 20 \
     --end 50 \
     --comp 1 \
     --steps 30 \
     --path data/50
   ```

   | Argument              | Description                                                                                                                               |
   | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
   | `--ye`                | Simulation expressed in years (`True`) or months (`False`)                                                                                |
   | `--start`             | Starting year for training data                                                                                                           |
   | `--end`               | Ending year (usually the last simulated year)                                                                                             |
   | `--comp`              | Number or ratio of components to use                                                                                                      |
   | `--steps`             | Jump size (years if `--ye True`, months otherwise)                                                                                        |
   | `--path`              | Directory containing the simulation files                                                                                                 |
   | `--ocean-terms`       | Path to `ocean_terms.yaml` mapping logical terms (SSH, Salinity, Temperature) to dataset variable names; uses packaged default if omitted |
   | `--techniques-config` | Path to `techniques_config.yaml` selecting DR and forecast techniques; uses packaged default if omitted                                   |

   The forecast outputs predicted ocean state variables to `<--path>/forecasts/latest/forecasts/simu_predicted/`.

   > Note: a `--prediction-dir` CLI argument will soon be added to `nemo-spinup-forecast` so that the output location can be specified explicitly rather than nested under the input data directory.

---

### C. Create the updated restart file

6. **Create the updated restart file**

   Using the forecasted ocean state from the previous step, `nemo-spinup-restart` injects the predicted variables (SSH, temperature, salinity, and derived velocities) into the original NEMO restart file. A new restart file is created with `NEW_` prepended to the filename, leaving the original intact and ready to initialise NEMO at the projected year.

   ```bash
   nemo-spinup-restart \
     --restart_path data/50/ \
     --radical DINO_00576000_restart \
     --mask_file data/50/mesh_mask.nc \
     --prediction_path data/50/forecasts/latest/forecasts/simu_predicted/ \
     --ocean_terms ./nemo-spinup-forecast/src/nemo_spinup_forecast/configs/ocean_terms.DINO.yaml
   ```

   > Note: the `--prediction_path` is currently nested under `data/50/forecasts/latest/forecasts/` because `nemo-spinup-forecast`'s CLI creates an extra `forecasts/` subdirectory inside each run. This will be simplified upstream so the path becomes `data/50/forecasts/latest/simu_predicted/`.

   - **`--radical`** is the prefix of the restart file (e.g. `DINO_00576000_restart`)
   - Output files are named as the originals but with `NEW` prepended

---

### D. Evaluate the projected restart file

7. **Move the projected restart to a separate directory and evaluate it:**

   ```bash
   mkdir -p data/50_projected
   mv data/50/NEW_DINO_00576000_restart*.nc data/50_projected/
   ln -s ../50/mesh_mask.nc data/50_projected/mesh_mask.nc

   nemo-spinup-evaluation \
     --sim-path data/50_projected \
     --config configs/DINO-evaluation.yaml \
     --results-dir output \
     --result-file-prefix projected \
     --mode restart
   ```

   Compare `output/projected_restart.csv` against the baseline from step 4 (`output/baseline_restart.csv`) to assess the impact of the spin-up acceleration.

---

### E. Running NEMO with the new state

1. **Copy the experiment directory** `EXP00` as a backup; the original will be overwritten in the next step.

2. **Copy the updated restart files** (`NEW_DINO_<time>_restart_<proc_id>.nc`) back to the original experiment directory.

3. **Update `namelist_cfg`** under `namrun`:

   | Parameter      | Description                                    |
   | -------------- | ---------------------------------------------- |
   | `nn_it000`     | First timestep (last timestep + 1)             |
   | `nn_itend`     | Final timestep                                 |
   | `cn_ocerst_in` | Restart filename (matches latest restart file) |
   | `ln_rstart`    | `.true.` to start from a restart file          |

4. **Restart DINO** using the updated restart file.

---

## Example Results

Results from running the benchmark with 50 years of DINO data, forecasting 30 years ahead from year 20–50 using PCA + Gaussian process regression (`--start 20 --end 50 --steps 30 --comp 1`).

| Metric | Baseline (50 yr) | Projected | Difference | % Change |
|---|---|---|---|---|
| check_density_from_file | 0.000020 | 0.011721 | +0.011701 | — |
| check_density_computed | 0.000032 | 0.011721 | +0.011689 | — |
| temperature_500m_30NS (°C) | 11.508 | 11.441 | −0.067 | −0.58% |
| temperature_BWbox (°C) | 5.197 | 5.203 | +0.005 | +0.10% |
| temperature_DWbox (°C) | 5.329 | 5.318 | −0.011 | −0.21% |
| ACC_Drake (Sv) | 188.69 | −102.75 | −291.44 | −154% |
| ACC_Drake_2 (Sv) | 188.69 | −102.75 | −291.44 | −154% |
| NASTG_BSF_max (Sv) | 35.52 | 16.81 | −18.70 | −52.7% |

**Observations:**
- Temperature metrics are well preserved (< 1% change), indicating the scalar field forecast is accurate.
- Density monotonicity violations increased from near-zero to ~1.2% of grid points.
- Transport metrics (ACC Drake, NASTG BSF) show large deviations. The geostrophic velocity reconstruction in `nemo-spinup-restart` produces physically unrealistic velocities — this is a known issue under investigation.

---

## Citation

If you use this work, please cite:

> _Citation to be added upon publication._
