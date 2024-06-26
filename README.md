#Basic Workflow


## Data Preprocessing
- each dataset will have its own preprocessing script:
  - `python process_biolib_immune.py`
  - `python process_aced_immune.py`
  - `python process_crc_orion.py`

## Training
- `python run_training.py` will handle model training

## Model Evaluation
- First do iterative panel selection to determine marker order:
  - `run_panel_selection.py --ckpt <\path\to\model\checkpoint> --val_dataset </path/to/training/dataset> --val_dataset_size <NUM_CELLS> --max_panel_size <MAX_NUM_MARKERS> --gpu_id <GPU_ID> --batch_size <BATCH_SIZE> --params_file </path/to/params/file>`
- Then generate plots for model + marker order: `python model_panel_evaluation.py`. You can also generate embedding plots: `python embedding_eval.py`
  - currenly no CLI setup for these two scripts, so you must change the variables in the script  
  
