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

> **Zenodo DOI:** _to be added_

---

## Installation of Nemo-Spinup-Bench dependencies 

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
> The commands below use paths from the Zenodo reference dataset. Substitute `/path/to/reference/data` with your own data directory if not using the reference data.

### Data preparation

1. **Get simulation data**

   The entire benchmark will run using sample data hosted on Zenodo. Alternatively you may run NEMO/DINO yourself; we recommend running for at least 50–100 years. A Slurm script is provided in the NEMO [notes](https://github.com/m2lines/Spinup-NEMO-notes/blob/main/nemo/buildandrun_NEMODINO.md). The Zenodo reference data contains 50 years of DINO output files to train on. 
   **Download reference data from Zenodo**

   ```bash
   # TODO: add download instructions once Zenodo record is created
   ```

2. **(Optional) Combine restart files and mesh mask** using [REBUILD_NEMO](https://forge.nemo-ocean.eu/nemo/nemo/-/tree/4.2.0/tools/REBUILD_NEMO):

   This step is only required if you are using your own NEMO run. The Zenodo reference data already includes combined files. You can use the same module environment used to run NEMO/DINO to compile `rebuild_nemo`.

   ```bash
   ./rebuild_nemo -n ./nam_rebuild /path/to/reference/data/restart/DINO_00576000_restart 36
   ./rebuild_nemo -n ./nam_rebuild /path/to/reference/data/mesh_mask 36
   ```

If more training data is needed, concatenate monthly outputs `*grid_T.nc` with `ncrcat`, part of the [NCO (netCDF Operators)](https://nco.sourceforge.net/).


### Spin-up acceleration

3. **Establish a baseline evaluation** of the cold-start reference simulation:

   ```bash
   nemo-spinup-evaluation \
     --sim-path /path/to/reference/data \
     --config nemo-spinup-evaluation/configs/DINO-setup.yaml \
     --mode both
   ```

5. **Resample data**

   > TODO: This step will soon use `cdo` to resample data. This is currently being done with [nemo-spinup-forecast/Notebooks/Resample_ssh.ipynb](https://github.com/m2lines/nemo-spinup-forecast/blob/main/Notebooks/Resample_ssh.ipynb)

   All data must be temporally aligned before forecasting. Use the [Resample_ssh.ipynb](https://github.com/m2lines/nemo-spinup-forecast/blob/main/Notebooks/Resample_ssh.ipynb) notebook to convert monthly SSH (`DINO_1m_grid_T.nc`) to annual (`DINO_1m_To_1y_grid_T.nc`). Temperature and salinity (3-D) are already annual (`DINO_1y_grid_T.nc`).

6. **Create the projected state**

   Set `--path` to the NEMO/DINO data directory:

   ```bash
   nemo-spinup-forecast \
     --ye True \
     --start 20 \
     --end 50 \
     --comp 1 \
     --steps 30 \
     --path /path/to/reference/data \
     --ocean-terms nemo-spinup-forecast/ocean_terms.yaml \
     --techniques-config nemo-spinup-forecast/src/nemo_spinup_forecast/techniques_config.yaml
   ```

   - **`ye`** — simulation expressed in years (`True`) or months (`False`)
   - **`start`** — starting year for training data
   - **`end`** — ending year (usually the last simulated year)
   - **`comp`** — number or ratio of components to use
   - **`steps`** — jump size (years if `ye=True`, months otherwise)
   - **`path`** — directory containing the simulation files
   - **`ocean-terms`** — path to `ocean_terms.yaml` mapping logical terms (SSH, Salinity, Temperature) to dataset variable names; uses packaged default if omitted
   - **`techniques-config`** — path to `techniques_config.yaml` selecting DR and forecast techniques; uses packaged default if omitted

7. **Create the updated restart file**

   ```bash
   nemo-spinup-restart \
     --restart_path /path/to/reference/data/restart/ \
     --radical DINO_00576000_restart \
     --mask_file /path/to/reference/data/mesh_mask.nc \
     --prediction_path /path/to/forecasts/latest/simu_predicted/ \
     --ocean_terms nemo-spinup-forecast/ocean_terms.yaml
   ```

   - **`--radical`** is the prefix of the restart file (e.g. `DINO_00576000_restart`)
   - Output files are named as the originals but with `NEW` prepended

8. **Evaluate** the projected state and compare against the baseline:

   ```bash
   nemo-spinup-evaluation \
     --sim-path /path/to/new/simulation \
     --ref-sim-path /path/to/simulation/data \  # optional: offline ground truth reference simulation
     --config nemo-spinup-evaluation/configs/DINO-setup.yaml \
     --mode both
   ```

   - **`--ref-sim-path`** — the offline reference simulation used as ground truth for comparison. TODO: upload full reference simulation data to Zenodo.

---

## Running NEMO with the new state

1. **Copy the experiment directory** inside the NEMO repository as a backup; the original will be overwritten in the next step.

2. **Copy the updated restart files** (`mesh_mask_<proc_id>.nc` and `DINO_<time>_restart_<proc_id>.nc`) back to the original experiment directory.

3. **Update `namelist_cfg`** under `namrun`:

   - `nn_it000` — first timestep (last timestep + 1)
   - `nn_itend` — final timestep
   - `cn_ocerst_in` — restart filename (matches latest restart file)
   - `ln_rstart` — `.true.` to start from a restart file

4. **Restart DINO** using the updated restart file.

---

## Citation

If you use this work, please cite:

> _Citation to be added upon publication._
