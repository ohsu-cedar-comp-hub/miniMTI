python create_panel_selection_data.py

python ../eval/run_panel_selection.py \
--val-dataset /home/groups/ChangLab/dataset/lunaphore-immune-unnorm/lunaphore_panel_select_data.h5 \
--ckpt /home/groups/ChangLab/simsz/cycif-panel-reduction/training/MVTM/MVTM-panel-reduction/ota9pmon \
--param-file /home/groups/ChangLab/simsz/cycif-panel-reduction/eval/params_mvtm-256-lunaphore.json \
--max-panel-size 43 \
--gpu-id 6 \
--downscale 
#TODO: should output ordering into "orderings/" directory, but there is currently no way to pass in the path, so the directory will be created wherever the script is called.