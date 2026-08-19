ln -sf ../50/mesh_mask.nc data/50_projected/mesh_mask.nc

nemo-spinup-evaluation \
   --sim-path data/50_projected \
   --config configs/DINO-evaluation.yaml \
   --results-dir evaluation-output \
   --result-file-prefix projected \
   --mode restart
