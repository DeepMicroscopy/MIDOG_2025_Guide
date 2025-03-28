import argparse
import os
import pandas as pd
import numpy as np
import pprint
import logging

from pathlib import Path
from tqdm.autonotebook import tqdm 

from utils.inference import load_model_from_config, setup_inference
from utils.eval_utils import optimize_threshold 
from utils.factory import ConfigCreator, ModelFactory


# Set default configurations
BATCH_SIZE = 8
DEVICE = 'cuda'
NMS_THRESH = 0.3
NUM_WORKERS = 8
OVERLAP = 0.3
VERBOSE = False
SPLIT = 'val'
BOX_FORMAT = 'cxcy'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",     type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--box_format",     type=str, default=BOX_FORMAT, help='Box format (default: xyxy).')
    parser.add_argument("--config_file",    type=str, help="Existing config file.", required=True)
    parser.add_argument("--dataset",        type=str, help="Dataset filepath.", required=True)
    parser.add_argument("--device",     	type=str, default=DEVICE, help="Device.")
    parser.add_argument("--img_dir",        type=str, help="Image directory.", required=True)
    parser.add_argument("--nms_thresh",     type=float, default=NMS_THRESH, help="Final NMS threshold.")
    parser.add_argument("--num_workers",    type=int, default=NUM_WORKERS, help="Number of processes.")
    parser.add_argument("--overlap",        type=float, default=OVERLAP, help="Overlap between patches.")
    parser.add_argument("--overwrite",      action="store_true", help="If true, existing results are overwritten.")
    parser.add_argument("--split",          type=str, default=SPLIT, help="Data split to evaluate.")
    parser.add_argument("--wsi",            action="store_true", help="Processes WSI")
    return parser.parse_args()



def optimize(
        batch_size: int,
        box_format: str, 
        config_file: str, 
        dataset: str | pd.DataFrame, 
        device: str,  
        img_dir: str, 
        num_workers: int,
        overlap: float, 
        overwrite: bool,
        split: str,
        wsi: bool,
        logger: logging.Logger = None
):

    # Check existing config_file 
    if not Path(config_file).exists():
        raise FileNotFoundError(f"Could not find this config_file: {config_file}. Provide existing config_file.")
    
    # Check dataset 
    if isinstance(dataset, str):
        if not Path(dataset).exists():
            raise FileNotFoundError(f"Could not find dataset: {dataset}")
        dataset = pd.read_csv(dataset)
    
    # Convert to xmin, ymin, xmax, ymax
    if box_format == 'cxcy':
        radius = 25
        dataset = dataset.assign(xmin=dataset['x'] - radius)
        dataset = dataset.assign(ymin=dataset['y'] - radius)
        dataset = dataset.assign(xmax=dataset['x'] + radius)
        dataset = dataset.assign(ymax=dataset['y'] + radius)

    # Check image directory
    img_dir = Path(img_dir)
    if not img_dir.is_dir():
        raise ValueError(f"This is not a directory: {str(img_dir)}")
    
    # Load the model 
    model, config = load_model_from_config(config_file)

    # Setup the inference 
    processor, patch_config = setup_inference(
        model=model,
        is_wsi=wsi,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        patch_size=config.patch_size,
        overlap=overlap,
        overwrite=overwrite,
        logger=logger
        )

    print('Loaded model configurations:')
    pprint.pprint(config)
    print()

    # Filter dataset
    optim_dataset = dataset.query("split == @split")
    filenames = optim_dataset.filename.unique()

    # Initialize predictions
    preds = {}

    # Run inference over all images 
    for file in tqdm(filenames, desc="Running inference"):

        # Create image path
        image_path = img_dir.joinpath(file)
        
        # Process images individually
        results = processor.process_single(image_path, patch_config)

        # Collect predictions
        preds[file] = results

    # Filter only mitotic figures
    filtered_dataset = optim_dataset.query('label == 1')

    # Optimize detection threshold
    best_thresh, best_f1, _, _ = optimize_threshold(
        dataset=filtered_dataset,
        preds=preds, 
        min_thresh=config.det_thresh
    )
    print(f'Best threshold: F1={best_f1:.4f}, Threshold={best_thresh:.2f}\n')

    rounded_thresh = float(np.round(best_thresh, decimals=4))
    # Updating model configs
    config.update({'det_thresh': rounded_thresh}) 
    print(f'Updated model configs with optimized threshold: {rounded_thresh}.')
    config.save(config_file)
    print(f'Updated config file at: {config_file}.')



def main(args):

    # Convert args
    optimize_kwargs = vars(args)

    # Run optimize
    optimize(**optimize_kwargs)

    

if __name__ == "__main__":
    args = get_args()
    main(args)
    print('End of script.')






