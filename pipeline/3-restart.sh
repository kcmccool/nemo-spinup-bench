nemo-spinup-restart \
     --restart_path data/50/ \
     --radical DINO_00576000_restart \
	 --mask_file data/50/mesh_mask.nc \
	 --prediction_path data/50/predictions/latest/forecast/simu_predicted/ \
     --ocean_terms ./nemo-spinup-forecast/src/nemo_spinup_forecast/configs/ocean_terms.DINO.yaml
