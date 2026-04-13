# 2 - Diffusion state restart file generation and evaluation

This example demonstrates the full pipeline for generating upscaled NEMO restart files from diffusion model predictions and evaluating them. It uses the GORCE configuration with 1-degree coarse predictions upscaled to 0.25-degree resolution.

Data is available from two Zenodo repositories:
- Diffusion model outputs: https://zenodo.org/records/16941776
- Reference restart files and mesh masks: https://zenodo.org/records/19557419

## Prerequisites

Install xESMF for regridding:

```bash
conda install -c conda-forge xesmf
```

Install the nemo-spinup-restart tool (from the repo root):

```bash
pip install ./nemo-spinup-restart
```

Install nemo-spinup-evaluation (from the repo root):

```bash
pip install ./nemo-spinup-evaluation
```

## Data setup

Download data from both Zenodo repositories and organise into the following structure:

```
data/
├── diffusion_states/
│   └── chamon_C2_clean/       # from generated_npy_files.zip (zenodo/16941776)
│       ├── toce.npy
│       ├── soce.npy
│       └── ssh.npy
├── 100-reference/             # from upscale-example.zip (zenodo/19557419)
│   ├── DINO_00000002_restart.nc
│   ├── mesh_mask.nc
│   └── namelist_cfg
└── 025-reference/             # from upscale-example.zip (zenodo/19557419)
    ├── DINO_10800000_restart.nc
    └── mesh_mask.nc
```

## Regrid and upscale

Generate a 0.25-degree restart file from the diffusion model predictions:

```bash
bash upscale.sh
```

This runs the `nemo-upscale` tool which:
1. Loads the diffusion model numpy predictions (temperature, salinity, SSH)
2. Creates a coarse (1-degree) restart file from the predictions
3. Regrids to fine (0.25-degree) resolution using xESMF bilinear interpolation
4. Applies the fine resolution mask and zeros out velocities for NEMO to recompute

Output is written to `./generated/`.

## Evaluate using nemo-spinup-evaluationuation

Evaluate the generated restart file against reference diagnostics:

```bash
bash evaluate.sh
```

Results are saved to `./results/`.

## Running the full pipeline

To run both steps in sequence:

```bash
bash upscale.sh && bash evaluate.sh
```
