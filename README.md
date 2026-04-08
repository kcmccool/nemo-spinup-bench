# NEMO Spin-Up Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

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
   git clone --recurse-submodules git@github.com:m2lines/nemo-spinup-bench.git
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
> The commands below assume reference data is downloaded to `data/DINO/`. Substitute this with your own data directory if not using the reference data.

### A. Data preparation

1. **Get simulation data**

   The entire benchmark will run using sample data hosted on Zenodo. Alternatively you may run NEMO/DINO yourself; we recommend running for at least 50–100 years. The Zenodo data contains 50 years of DINO output files to train on. 

   **Download data from Zenodo**

   Download `50.zip` from [Zenodo record 19474414](https://zenodo.org/records/19474414), unzip it, and place the contents in `data/DINO/`. The record also provides `200.zip` (200 years of DINO output) and `restart.zip` (200 annual restart files) for extended experiments.

   ```bash
   mkdir -p data/DINO
   curl -L -o 50.zip https://zenodo.org/records/19474414/files/50.zip
   unzip 50.zip -d data/DINO/
   ```

2. **(Optional) Combine restart files and mesh mask** using [REBUILD_NEMO](https://forge.nemo-ocean.eu/nemo/nemo/-/tree/4.2.0/tools/REBUILD_NEMO):

   This step is only required if you are using your own NEMO run. The Zenodo reference data already includes combined files. You can use the same module environment used to run NEMO/DINO to compile `rebuild_nemo`.

   ```bash
   ./rebuild_nemo -n ./nam_rebuild data/DINO/restart/DINO_00576000_restart 36
   ./rebuild_nemo -n ./nam_rebuild data/DINO/mesh_mask 36
   ```

3. **(Optional) Resample data**

   This step is not required when using the Zenodo reference data, which already contains the resampled file `DINO_1m_to_1y_grid_T.nc`.

   All data must be temporally aligned before forecasting. If you are bringing your own NEMO output, convert monthly SSH to annual using [`cdo`](https://code.mpimet.mpg.de/projects/cdo):

   ```bash
   cdo yearmean DINO_1m_grid_T.nc DINO_1m_to_1y_grid_T.nc
   ```

   Temperature and salinity (3-D) are already annual (`DINO_1y_grid_T.nc`).

   If more training data is needed, concatenate monthly outputs `*grid_T.nc` with `ncrcat`, part of the [NCO (netCDF Operators)](https://nco.sourceforge.net/).

---

### B. Spin-up acceleration

The spin-up acceleration pipeline forecasts the ocean state forward in time using dimensionality reduction and Gaussian process regression, generates updated restart files, and evaluates the result against a reference numerical run. We begin with a baseline evaluation of the reference simulation so that the final evaluation can be compared against it.

4. **Establish a baseline evaluation** of the cold-start reference simulation:

   ```bash
   nemo-spinup-evaluation \
     --sim-path data/DINO \
     --config nemo-spinup-evaluation/configs/DINO-setup.yaml \
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
     --path data/DINO \
     --ocean-terms nemo-spinup-forecast/ocean_terms.yaml \
     --techniques-config nemo-spinup-forecast/src/nemo_spinup_forecast/techniques_config.yaml
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

   The forecast outputs predicted ocean state variables to `forecasts/simu_predicted/`.

6. **Create the updated restart file**

   Using the forecasted ocean state from the previous step, `nemo-spinup-restart` injects the predicted variables (SSH, temperature, salinity, and derived velocities) into the original NEMO restart file. A new restart file is created with `NEW_` prepended to the filename, leaving the original intact and ready to initialise NEMO at the projected year.

   ```bash
   nemo-spinup-restart \
     --restart_path data/DINO/restart/ \
     --radical DINO_00576000_restart \
     --mask_file data/DINO/mesh_mask.nc \
     --prediction_path forecasts/simu_predicted/ \
     --ocean_terms nemo-spinup-forecast/ocean_terms.yaml
   ```

   - **`--radical`** is the prefix of the restart file (e.g. `DINO_00576000_restart`)
   - Output files are named as the originals but with `NEW` prepended

7. **Evaluate** the projected state and compare against the baseline:

   This is the evaluation on the updated restart state compared to the reference at 80 years. We recommend outputting restart files fairly frequently to test different step settings. 

   ```bash
   nemo-spinup-evaluation \
     --sim-path ./data/DINO \
     --ref-sim-path ./data/DINO/reference/ \  # optional: offline ground truth reference simulation
     --config nemo-spinup-evaluation/configs/DINO-setup.yaml \
     --results-dir output \
     --result-file-prefix spinup_evaluation \
     --mode restart
   ```

   - **`--ref-sim-path`** — the offline reference simulation used as ground truth for comparison. TODO: upload full reference simulation data to Zenodo.

   Results are written to `output/spinup_evaluation_restart.csv`. Compare these against the 80 year reference restart file output to understand how the ML forecast from the spin-up acceleration compares to the simulation.

   > TODO: Does the DINO-setup.yaml need to modified here to point to the 80 year restart file. 

---

### C. Running NEMO with the new state

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

## Citation

If you use this work, please cite:

> _Citation to be added upon publication._
