from typing import Dict, List, Optional, Tuple, Union
import lightning.pytorch as pl
import torch
from torch import nn 
import torch.nn.functional as F

from torchmetrics.detection import MeanAveragePrecision
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, ChainedScheduler, OneCycleLR, CyclicLR, ExponentialLR, CosineAnnealingLR
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class BaseDetectionModule(pl.LightningModule):
    """Base PyTorch Lightning module for object detection tasks.

    This module provides a foundation for training and evaluating object detection models,
    including standard training loops, validation metrics, and optimizer configuration.

    Args:
        model (nn.Module): The detection model to train/evaluate.
        batch_size (int, optional): Batch size for logging purposes. Defaults to 16.
        lr (float, optional): Learning rate for optimization. Defaults to 0.0001.
        optimizer (str, optional): Optimizer type ('SGD', 'Adam', or 'AdamW'). Defaults to 'AdamW'.
        scheduler (Union[str, None], optional): Learning rate scheduler type.
            Currently supports 'CosineAnnealingLR'. Defaults to None.

    Attributes:
        metric (MeanAveragePrecision): Evaluation metric for object detection.
    """

    def __init__(
            self,
            model: nn.Module,
            batch_size: int = 16,
            lr: float = 0.0001,
            optimizer: str = 'AdamW',
            scheduler: Union[str, None] = None):
        super().__init__()

        # Validate input parameters
        allowed_optimizers = ['SGD', 'Adam', 'AdamW']
        allowed_schedulers = [None, 'CosineAnnealingLR']

        if optimizer not in allowed_optimizers:
            raise ValueError(f"Optimizer must be one of {allowed_optimizers}")
        if scheduler not in allowed_schedulers:
            raise ValueError(f"Scheduler must be one of {allowed_schedulers}")

        # save hparams
        self.save_hyperparameters(ignore=['model'])

        # store model
        self.model = model

        # set up metric
        self.metric = MeanAveragePrecision(
            iou_thresholds=[0.5],
            class_metrics=False,
            iou_type='bbox',
            box_format='xyxy',
            max_detection_thresholds=[1, 10, 500],
            backend='faster_coco_eval'
        )

    def forward(self,
                x: torch.Tensor,
                y: Optional[List[Dict[str, torch.Tensor]]] = None
                ) -> List[Dict[str, torch.Tensor]]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input images tensor of shape (B, C, H, W).
            y (Optional[List[Dict[str, torch.Tensor]]], optional): List of target dictionaries
                containing 'boxes' and 'labels'. Required for training. Defaults to None.

        Returns:
            List[Dict[str, torch.Tensor]]: Model predictions or loss dictionary during training.
        """
        return self.model(x, y)

    def training_step(self, batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]],
                     batch_idx: int) -> torch.Tensor:
        """Performs a single training step.

        Args:
            batch (Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]): Tuple of images and targets.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Total loss value for the batch.
        """
        images, targets = batch
        loss_dict = self(images, targets)

        # Compute total loss
        loss = sum(loss_dict.values())

        # Log losses
        self.log('train/loss', loss, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log_dict(
            {f'train/{name}': value for name, value in loss_dict.items()},
            on_epoch=True,
            batch_size=self.hparams.batch_size
        )
        return loss
    
    def _shared_evaluation_step(self, batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]],
                       batch_idx: int) -> None:
        """Performs a single validation step to be used for patch-based validation and testing.

        Args:
            batch (Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]): Tuple of images and targets.
            batch_idx (int): Index of the current batch.
        """
        images, targets = batch
        preds = self(images)
        self.metric.update(preds, targets)

    def _shared_evaluation_epoch_end(self) -> None:
        """Computes and logs validation metrics at the end of the epoch."""
        metrics = self.metric.compute()

        # Log metrics 
        ap = metrics['map_50']
        ar = metrics['mar_500']
        self.log('val/map', ap, prog_bar=True)
        self.log('val/mar', ar, prog_bar=True)

        self.metric.reset()


    def validation_step(self, batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]],
                       batch_idx: int) -> None:
        """Performs the validation step."""
        self._shared_evaluation_step(batch, batch_idx)


    def test_step(self, batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]],
                       batch_idx: int) -> None:
        """Performs the test step."""
        self._shared_evaluation_step(batch, batch_idx)


    def on_evaluation_epoch_end(self) -> None:
        """Computes the validation metrics."""
        self._shared_evaluation_epoch_end()


    def on_test_epoch_end(self) -> None:
        """Computes the test metrics."""
        self._shared_evaluation_epoch_end()


    def configure_optimizers(self) -> Union[List[Optimizer], Tuple[List[Optimizer], List[_LRScheduler]]]:
        """Configures optimizers and learning rate schedulers.

        Returns:
            Union[List[Optimizer], Tuple[List[Optimizer], List[_LRScheduler]]]:
                Configured optimizer and optional scheduler.
        """
        parameters = [p for p in self.parameters() if p.requires_grad]

        optimizer = {
            'SGD': torch.optim.SGD(parameters, lr=self.hparams.lr),
            'Adam': torch.optim.Adam(parameters, lr=self.hparams.lr),
            'AdamW': torch.optim.AdamW(parameters, lr=self.hparams.lr)
        }.get(self.hparams.optimizer)

        if self.hparams.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=1e-7
            )
            return [optimizer], [scheduler]

        return [optimizer]