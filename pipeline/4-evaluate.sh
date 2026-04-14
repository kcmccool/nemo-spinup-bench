mkdir -p data/50_projected
mv data/50/NEW_DINO_00576000_restart*.nc data/50_projected/
ln -s ../50/mesh_mask.nc data/50_projected/mesh_mask.nc

nemo-spinup-evaluation \
   --sim-path data/50_projected \
   --config configs/DINO-evaluation.yaml \
   --results-dir output \
   --result-file-prefix projected \
   --mode restart
