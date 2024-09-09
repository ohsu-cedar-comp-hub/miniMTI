# Basic Workflow


## Data Preprocessing
Each dataset will have its own preprocessing script:
  - `python process_biolib_immune.py`
  - `python process_aced_immune.py`
  - `python process_crc_orion.py`
  - 
These scripts will generate a series of HDF5 files with three datasets in each `images`, `masks`, and `metadata`. Each dataset is described in detail below:
  - `images`: dataset containing all 8 bit images centered around a single cell with shape=(N, H, W, C)
  - `masks`: dataset containing the binary masks for the cell at the center of each image with shape=(N, H, W)
  - `metadata`: dataset containing strings that include the cell ID and (x,y) coordinates for the cell at the center of each image. string is formatted like so: `"{sample_name}-CellID-{rp.label}-x={center_x}-y={center_y}"`

For training, we want to combine these HDF5 datasets so that batches cells selected from each sample. To create a set of  unified training and validation files, use the script `data/create_training_files.py` like so: `python create_training_file.py --data_dir </path/to/directory/containing/HDF5/files> --val_samples </path/to/text/file/listing/validation/samples.txt> --batch-name <name_for_validation_batch>` details for the input parameters are below:
  - `data-dir`: this should be a path to a directory containing the HDF5 files, one per sample
  - `val-samples`: You will need to create a .txt file that has one sample name per row that will be in the validation set. Every file not named in this file will be included in the training set. IMPORTANT: The sample names in this file must also be present in the filename of the HDF5 file for that sample. e.g. "CRC01" is the sample name for the file "orion_crc_dataset_sid=CRC01.h5"
  
## Training
- `python run_training.py` will handle model training

## Model Evaluation
- First do iterative panel selection to determine marker order:
  - `python run_panel_selection.py --ckpt <\path\to\model\checkpoint> --val_dataset </path/to/training/dataset> --val_dataset_size <NUM_CELLS> --max_panel_size <MAX_NUM_MARKERS> --gpu_id <GPU_ID> --batch_size <BATCH_SIZE> --params_file </path/to/params/file>`
- Then generate plots for model + marker order: `python model_panel_evaluation.py`. You can also generate embedding plots: `python embedding_eval.py`
  - currenly no CLI setup for these two scripts, so you must change the variables in the script  
  
