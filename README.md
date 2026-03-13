# NEMO Spin-Up Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Reproducible research artifact for the NEMO spin-up acceleration method. This repository
contains the full end-to-end workflow as a citable, versioned snapshot.

---

## Repositories

| Submodule | Description |
|-----------|-------------|
| [`nemo-spinup-forecast`](nemo-spinup-forecast/) | Dimensionality reduction and forecasting |
| [`nemo-spinup-restart`](nemo-spinup-restart/) | Restart file generation |
| [`nemo-spinup-evaluation`](nemo-spinup-evaluation/) | Evaluation and validation |

---

## Reference Data

Reference data (DINO output, restart files, `mesh_mask.nc`) is archived on Zenodo:

> **Zenodo DOI:** _to be added_

---

## End-to-End Steps for Running Spin-Up NEMO

1. **Clone this repository with submodules**

   ```bash
   git clone --recurse-submodules git@github.com:m2lines/nemo-spinup-bench.git
   cd nemo-spinup-bench
   ```

2. **Download reference data from Zenodo**

   ```bash
   # TODO: add download instructions once Zenodo record is created
   ```

3. **Create a virtual environment and install dependencies**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install ./nemo-spinup-forecast
   pip install ./nemo-spinup-restart
   pip install ./nemo-spinup-evaluation
   ```

4. **Resample SSH** from monthly to annual using the notebook in `nemo-spinup-forecast/Notebooks/Resample_ssh.ipynb`.

5. **Create the projected state**

   ```bash
   python -m nemo_spinup_forecast \
     --ye True \
     --start 20 \
     --end 50 \
     --comp 1 \
     --steps 30 \
     --path /path/to/reference/data \
     --ocean-terms nemo-spinup-forecast/ocean_terms.yaml \
     --techniques-config nemo-spinup-forecast/src/nemo_spinup_forecast/techniques_config.yaml
   ```

6. **Prepare restart files** using REBUILD\_NEMO:

   ```bash
   ./rebuild_nemo -n ./nam_rebuild /path/to/reference/data/restart/DINO_00576000_restart 36
   ./rebuild_nemo -n ./nam_rebuild /path/to/reference/data/mesh_mask 36
   ```

7. **Create the updated restart file**

   ```bash
   nemo-spinup-restart \
     --restart_path /path/to/reference/data/restart/ \
     --radical DINO_00576000_restart \
     --mask_file /path/to/reference/data/mesh_mask.nc \
     --prediction_path /path/to/forecasts/latest/simu_predicted/ \
     --ocean_terms nemo-spinup-forecast/ocean_terms.yaml
   ```

8. **Copy the updated restart files** back to the NEMO experiment directory and update
   `namelist_cfg` (`nn_it000`, `nn_itend`, `cn_ocerst_in`, `ln_rstart = .true.`).

9. **Restart DINO** using the updated restart file.

10. **Evaluate** the results:

    ```bash
    spinup-eval \
      --sim-path /path/to/new/simulation \
      --ref-sim-path /path/to/reference/data \
      --config nemo-spinup-evaluation/configs/DINO-setup.yaml \
      --mode both
    ```

---

## Citation

If you use this work, please cite:

> _Citation to be added upon publication._
