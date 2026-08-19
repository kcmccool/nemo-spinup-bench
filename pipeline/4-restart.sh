ln -sf ../50/DINO_00576000_restart.nc data/50_projected/DINO_00576000_restart.nc

nemo-spinup-restart \
     --restart_path data/50_projected/ \
     --radical DINO_00576000_restart \
     --mask_file data/50/mesh_mask.nc \
     --prediction_path data/50_projected/latest/forecast/simu_predicted/ \
     --ocean_terms configs/ocean_terms.DINO.yaml
