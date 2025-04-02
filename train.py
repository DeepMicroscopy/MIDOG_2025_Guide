import argparse
import lightning.pytorch as pl
import logging
import numpy as np
import os 
import pandas as pd
import pickle
import pprint
import wandb

from datetime import datetime
from pathlib import Path
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

from utils.datamodule import ObjectDetectionDataModule
from utils.factory import ModelFactory, ConfigCreator
from optimize_threshold import optimize


# Set default configurations
ACCELERATOR = 'gpu'
ANCHOR_RATIOS = (0.5, 1.0, 2.0)
ANCHOR_SIZES = (32, 48, 64, 96, 128)
ARB_PROB = 0.25
BACKBONE = 'resnext50_32x4d'
BATCH_SIZE = 6
BOX_FORMAT = 'cxcy'
DET_THRESH = 0.05
DEVICE = 'cuda'
DOMAIN_COL = 'tumortype'
EXP_DIR = 'experiments'
FG_PROB = 0.5
GRADIENT_CLIP_VAL = 1
LR = 1e-4
MAX_EPOCHS = 150
MODEL = 'FCOS'
NMS_THRESH = 0.3
NUM_CLASSES = 2
NUM_TRAIN_SAMPLES = 1024
NUM_VAL_SAMPLES = 512
NUM_WORKERS = 8
OPTIMIZER = 'AdamW'
PATCH_SIZE = 1024
PATIENCE = 10
PROJECT = 'MIDOG_2025'
RETURNED_LAYERS = [1, 2, 3, 4]
RUN_NAME = 'exp_0'
SAVE_TOP_K = 1
SCHEDULER = 'CosineAnnealingLR'
TOP_K = 1
TRAINABLE_BACKBONE_LAYERS = 5
WEIGHTS = 'IMAGENET1K_V2'
MIN_THRESH = 0.2
OVERLAP = 0.3
SPLIT = 'optim'
ENTITY = 'jonas_amme'
PRECISION = '16-mixed'



def get_args():
    parser = argparse.ArgumentParser()

    # Model specific parameters
    parser.add_argument("--anchor_ratios",              type=float, nargs='+' ,default=ANCHOR_RATIOS, help="Anchor ratios.")
    parser.add_argument("--anchor_sizes",               type=int, nargs='+' ,default=ANCHOR_SIZES, help="Anchor sizes.")
    parser.add_argument("--backbone",                   type=str, default=BACKBONE, help="Backbone.")
    parser.add_argument("--extra_blocks",               action='store_true', help="Adds P6P7 level to FPN.")
    parser.add_argument("--model",                      type=str, default=MODEL, help="Model type.")
    parser.add_argument("--normalize_stats",            type=str, default=None, help="Use specific normalization statistics.")
    parser.add_argument("--patch_size",                 type=int, default=PATCH_SIZE, help="Patch size.")

    # Experiment specific parameters
    parser.add_argument("--exp_dir",                    type=str, default=EXP_DIR, help='Directory to save models.',           )
    parser.add_argument("--run_name",                   type=str, default=RUN_NAME, help="Directory within exp_dir to save results for that run.")
    parser.add_argument("--project",                    type=str, default=PROJECT, help="WandB project name.")
    parser.add_argument("--entity",                     type=str, default=ENTITY, help="WandB username.")

    # Training specific parameters 
    parser.add_argument("--accelerator",                type=str, default=ACCELERATOR, help="Accelerator (gpu or cpu)")
    parser.add_argument("--arb_prob",                   type=float, default=ARB_PROB, help="Percentage of random patches.")
    parser.add_argument("--batch_size",                 type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--box_format",                 type=str, default=BOX_FORMAT, help='Box format (default: xyxy).')
    parser.add_argument("--dataset_file",               type=str, help="Your path/to/dataset_file.")
    parser.add_argument("--det_thresh",                 type=float, default=DET_THRESH, help="Box score threshold.")
    parser.add_argument("--device",     	            type=str, default=DEVICE, help="Device.")
    parser.add_argument("--domain_col",                 type=str, default=DOMAIN_COL, help='Column with domain identifier.')
    parser.add_argument("--early_stopping",             action="store_true", help="Use early stopping callback.")
    parser.add_argument("--fast_dev_run",               action="store_true", help="Fast dev run.")
    parser.add_argument("--fg_prob",                    type=float, default=FG_PROB, help="Mitosis percentage.")
    parser.add_argument("--gradient_clip_val",          type=int, default=GRADIENT_CLIP_VAL, help="Norm for clipping gradients.")
    parser.add_argument("--img_dir",                    type=str, help="Your path/to/images.")
    parser.add_argument("--lr",                         type=float, default=LR, help="Learning rate.")
    parser.add_argument("--max_epochs",                 type=int, default=MAX_EPOCHS, help="Maximum epochs of training.")
    parser.add_argument("--num_classes",                type=int, default=NUM_CLASSES, help="Number of classes.")
    parser.add_argument("--num_train_samples",          type=int, default=NUM_TRAIN_SAMPLES, help="Number of training samples.")
    parser.add_argument("--num_val_samples",            type=int, default=NUM_VAL_SAMPLES, help="Number of validation samples.")
    parser.add_argument("--num_workers",                type=int, default=NUM_WORKERS, help="Number of processes.")
    parser.add_argument("--optimizer",                  type=str, default=OPTIMIZER, help="Opimizer.")
    parser.add_argument("--returned_layers",            type=int, nargs='+', default=RETURNED_LAYERS, help="Layer to return from FPN.")
    parser.add_argument("--save_top_k",                 type=int, default=SAVE_TOP_K, help="Save top k checkpoints.")
    parser.add_argument("--scheduler",                  type=str, default=SCHEDULER, help="Learning rate scheduler.")
    parser.add_argument("--top_k",                      type=int, default=TOP_K, help="Monitor checkpoints")
    parser.add_argument("--trainable_backbone_layers",  type=int, default=TRAINABLE_BACKBONE_LAYERS, help="No. trainable backbone layers")
    parser.add_argument("--weights",                    type=str, default=WEIGHTS, help="Pretraining weights.")
    parser.add_argument("--wsi",                        action="store_true", help="Processes WSI")

    # Optimize specific parameters 
    parser.add_argument("--nms_thresh",                 type=float, default=NMS_THRESH, help="Final NMS threshold.")
    parser.add_argument("--overlap",                    type=float, default=OVERLAP, help="Overlap between patches.")
    parser.add_argument("--overwrite",                  action="store_true", help="If true, existing results are overwritten.")
    parser.add_argument("--split",                      type=str, default=SPLIT, help="Data split to use for threshold optimization.")
    return parser.parse_args()


def setup_logger(run_dir: Path):
    """Set up logger with file and console handlers."""
    # Create logs directory if it doesn't exist
    log_dir = run_dir.joinpath('logs')
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Configure the root logger
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_dir.joinpath(f'{run_dir.name}_{timestamp}.log')),
                            logging.StreamHandler()
                        ]) 
    logger = logging.getLogger('object_detection')
    return logger


def train(args):

    # Set up experiment directory
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Set up directory for the run
    run_dir = exp_dir.joinpath(args.run_name)
    run_dir.mkdir(exist_ok=True)
        
    # Set up logger
    logger = setup_logger(run_dir)
    logger.info(f"Created experiment directoy: {str(exp_dir)}.")
    logger.info(f"Created run directory: {str(run_dir)}.")
    logger.info("Starting training process")
    logger.info(f"Arguments: {vars(args)}")

    
    # Load stats if provided
    if args.normalize_stats is not None:
        logger.info(f"Loading normalization stats from {args.normalize_stats}")
        with open(args.normalize_stats, 'rb') as f:
            stats = pickle.load(f)
        means = stats['mean']
        stds = stats['std']
        logger.info(f"Loaded means: {means}, stds: {stds}")
    else:
        means = None
        stds = None
        logger.info("No normalization stats provided")


    # Set up wandb logging
    logger.info("Setting up WandB logger")
    wandb_logger = WandbLogger(project=args.project, name=args.run_name, entity=args.entity)
    wandb_logger.experiment.config.update(args)

    # Set up model kwargs
    model_kwargs = {
        'num_classes': args.num_classes,
        'backbone': args.backbone,
        'weights': args.weights,
        'trainable_backbone_layers': args.trainable_backbone_layers,
        'det_thresh': args.det_thresh,
        'extra_blocks': args.extra_blocks,
        'returned_layers': args.returned_layers,
        'image_mean': means,
        'image_std': stds,
        'patch_size': args.patch_size
    }

    # Set up module kwargs
    module_kwargs = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler
    }

    # Model creation
    logger.info(f"Creating {args.model} model")
    if args.model == 'FCOS':
        model = ModelFactory.create('FCOS', model_kwargs, module_kwargs)
    elif args.model == 'RetinaNet' or args.model == 'FasterRCNN':
        model_kwargs.update({
            'anchor_sizes': tuple(args.anchor_sizes),
            'anchor_ratios': tuple(args.anchor_ratios)
        })
        model = ModelFactory.create(args.model, model_kwargs, module_kwargs)
    else:
        logger.error(f'Unsupported model type: {args.model}')
        raise ValueError(f'Unsupported model type for {args.model}.')
    
    print(f'\nCreate model {args.model} with model parameters: ')
    pprint.pprint(model_kwargs)

    print(f'\nCreate lightning detction model {args.model} with module parameters: ')
    pprint.pprint(module_kwargs)
    
    # Set up datamodule 
    dm = ObjectDetectionDataModule(
        img_dir=args.img_dir,
        dataset=args.dataset_file,
        domain_col=args.domain_col, 
        box_format=args.box_format,
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        fg_prob=args.fg_prob,
        arb_prob=args.arb_prob,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
        )
    
    # Log gradients, params and topology
    wandb_logger.watch(model, log='all')

    # Set up callbacks 
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir, 
        monitor='val/map',
        mode='max', 
        save_top_k=args.top_k,
        filename=args.run_name)
    tqdm_callback = TQDMProgressBar(refresh_rate=10)
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    
    callbacks = [checkpoint_callback, tqdm_callback, lr_monitor_callback]

    if args.early_stopping:
        early_stopping_callback = EarlyStopping(monitor='val/map', patience=args.patience, mode='max')
        callbacks.append(early_stopping_callback)


    # Set up trainer 
    trainer = pl.Trainer(
        fast_dev_run=args.fast_dev_run,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        logger=wandb_logger,
        reload_dataloaders_every_n_epochs=1,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val
        )

    # Start training
    logger.info("Starting model training")
    trainer.fit(model, datamodule=dm)

    logger.info("Training completed")
    wandb.finish()

    # Create model config settings
    settings = {
        'model_name': args.run_name,
        'detector': args.model,
        'backbone': args.backbone,
        'weights': args.weights, 
        'checkpoint': checkpoint_callback.best_model_path,
        'det_thresh': args.det_thresh,
        'num_classes': args.num_classes,
        'extra_blocks': args.extra_blocks,
        'returned_layers': args.returned_layers,
        'patch_size': args.patch_size
    }

    if args.model == 'RetinaNet' or args.model == 'FasterRCNN':
        settings.update({
            'anchor_sizes': tuple(args.anchor_sizes),
            'anchor_ratios': tuple(args.anchor_ratios)
        })

    # Init config file
    config_file = ConfigCreator.create(settings)

    # Save model configs
    save_path = run_dir.joinpath(f"{args.model}_{args.run_name}.yaml")
    config_file.save(save_path)

    # Set up optim configs
    optimize_kwargs = {
        'batch_size': args.batch_size,
        'box_format': args.box_format,
        'config_file': str(save_path),
        'dataset': args.dataset_file,
        'device': args.device,
        'img_dir': args.img_dir,
        'logger': logger,
        'num_workers': args.num_workers,
        'overlap': args.overlap,
        'overwrite': args.overwrite,
        'patch_size': args.patch_size,
        'split': args.split,
        'wsi': args.wsi
    } 

    # Optimize detection threshold
    logger.info("Starting threshold optimization")
    optimize(**optimize_kwargs)
    logger.info("Training process completed successfully")


if __name__ == "__main__":
    args = get_args()
    train(args)
    print('End of script.')






    