from typing import Callable, List, Optional, Tuple, Union

import albumentations as A
import cv2
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

from pathlib import Path 
from torch.utils.data import DataLoader

from .dataset import DetectionDataset


Coords = Tuple[int, int]


class ObjectDetectionDataModule(pl.LightningDataModule):
    def __init__(self,
                dataset: Union[str, pd.DataFrame], 
                img_dir: Union[Path, str], 
                filename_col: str = 'filename',
                label_col: str = 'label',
                domain_col: str = None,
                box_format: str = 'xyxy', 
                num_train_samples: int = 1024,
                num_val_samples: int = 512,
                fg_prob: float = 0.5,
                arb_prob: float = 0.25,
                patch_size: int = 512,
                batch_size: int = 8,
                num_workers: int = 4,
                ) -> None:
        """LightningDataModule for training and validation of the object detection model.

        Args:
            dataset (Union[str, pd.DataFrame]): Dataset with annotations.
            img_dir (str): Path to the directory containing the images.
            num_train_samples (int, optional): Number of samples during training. Defaults to 1024.
            num_val_samples (int, optional): Number of samples during validation. Defaults to 512.
            mit_prob (float, optional): Mitosis percentage. Defaults to 0.5.
            arb_prob (float, optional): Random patch percentage. Defaults to 0.25.
            crop_size (Coords, optional): Patch size. Defaults to (512,512).
            same_val_ds (bool, optional): Whether to create the same validation dataset each epoch. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 8.
            num_workers (int, optional): Number of workers. Defaults to 4.
        """
        super().__init__()
        self.label_col = label_col
        self.domain_col = domain_col
        self.filename_col = filename_col
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.fg_prob = fg_prob
        self.arb_prob = arb_prob
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_dir = img_dir
        self.box_format = box_format
        

        # load annotations
        self.dataset = self._load_dataset(dataset)

        # split dataset
        self.train_dataset = self._create_dataset(split='train', transforms=self.train_transform)
        self.valid_dataset = self._create_dataset(split='val', transforms=None)


    def _load_dataset(self, dataset: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Loads the dataset."""
        if isinstance(dataset, str):
            return pd.read_csv(dataset)
        elif isinstance(dataset, pd.DataFrame):
            return dataset
        else:
            raise TypeError('dataset must be a string or pd.DataFrame')


    def _create_dataset(self, 
                        split: str,
                        transforms: Union[List[Callable], Callable] = None
                        ) -> DetectionDataset:
        """Creates a dataset for the given split, tumor type and transforms."""

        if 'split' not in self.dataset.columns:
            raise ValueError(f"Dataset must have column 'split' with values 'train' and 'val'.")

        if split == 'train':
            num_samples = self.num_train_samples
        elif split == 'val':
            num_samples = self.num_val_samples
        elif split == 'test':
            num_samples = self.num_val_samples
        
        dataset = self.dataset[self.dataset.split == split]
        
        return DetectionDataset(
            dataset=dataset,
            img_dir=self.img_dir,
            filename_col=self.filename_col,
            label_col=self.label_col,
            domain_col=self.domain_col,
            box_format=self.box_format,
            num_samples=num_samples,
            fg_prob=self.fg_prob,
            arb_prob=self.arb_prob,
            patch_size=self.patch_size,
            transforms=transforms
            )

    @property
    def train_transform(self) -> List[Callable]:
        aug_pipeline = A.Compose([
            A.D4(p=1),
            A.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.1, 0.1), p=0.5),
            A.Defocus(radius=(1,3), p=0.3),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['class_labels'])
        )
        return aug_pipeline


    def train_dataloader(self) -> DataLoader:
        self.train_dataset.create_samples()
        return DataLoader(dataset=self.train_dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        collate_fn=self.train_dataset.collate_fn)
                        

    def val_dataloader(self) -> DataLoader:
        self.valid_dataset.create_samples()
        return DataLoader(dataset=self.valid_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    collate_fn=self.valid_dataset.collate_fn)



    def test_dataloader(self) -> DataLoader:
        test_dataset = self._create_dataset(split='test')
        return DataLoader(dataset=test_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    collate_fn=test_dataset.collate_fn)
            
        