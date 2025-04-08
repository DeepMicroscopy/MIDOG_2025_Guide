from pathlib import Path
import cv2 
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import openslide
import logging
import json

from dataclasses import dataclass
from openslide import OpenSlide
from abc import ABC, abstractmethod
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.ops.boxes import nms as torch_nms
from tqdm.autonotebook import tqdm
from typing import Callable, Dict, List, Optional, Tuple, Union

from .factory import ConfigCreator, ModelFactory



Coords = Tuple[int, int]
ImageType = Union[np.ndarray, torch.Tensor]


@dataclass
class PatchConfig:
    """Configuration for patch extraction parameters.

    Args:
        size: Size of patches (assumed square)
        overlap: Overlap between adjacent patches (0-1)
        level: Pyramid level for WSI
        tissue_threshold: Minimum tissue content required (0-1)
    """
    size: int = 1024
    overlap: float = 0.3
    level: int = 0
    tissue_threshold: float = 0.1



@dataclass
class InferenceConfig:
    """Configuration for inference parameters.

    Args:
        batch_size: Number of patches to process simultaneously
        num_workers: Number of worker processes for data loading
        device: Device to run inference on ('cuda' or 'cpu')
        nms_thresh: Non-maximum suppression threshold
        is_wsi: Whether to use WSI or ROI dataloader
    """
    batch_size: int = 8
    num_workers: int = 4
    device: str = 'cuda'
    nms_thresh: float = 0.3
    is_wsi: bool = False


@dataclass
class ProcessorConfig:
    """Configuration for image processing."""
    save_dir: Optional[str] = None
    overwrite: Optional[bool] = False


class BaseInferenceDataset(Dataset, ABC):
    """Base class for inference datasets handling patch-based processing.

    Args:
        patch_config: Configuration for patch extraction
        transforms: Optional transforms to apply to patches
    """
    def __init__(
        self,
        patch_config: PatchConfig,
        transforms: Optional[Union[List[Callable], Callable]] = None,
    ) -> None:
        self.config = patch_config
        self.transforms = self._setup_transforms(transforms)

        # To be set by child classes
        self.coords: List[Coords] = []
        self.image_size: Tuple[int, int] = (0, 0)

    def _setup_transforms(
        self,
        transforms: Optional[Union[List[Callable], Callable]]
    ) -> Optional[Callable]:
        """Set up transformation pipeline."""
        if transforms is None:
            return None
        if isinstance(transforms, (list, tuple)):
            return T.Compose(transforms)
        return transforms

    @abstractmethod
    def _load_image(self) -> None:
        """Load the image/slide and set necessary attributes."""
        pass

    @abstractmethod
    def _get_patch(self, coords: Coords) -> ImageType:
        """Extract a patch from the image at given coordinates."""
        pass

    def _normalize_patch(self, patch: ImageType) -> torch.Tensor:
        """Normalize patch and convert to tensor."""
        if isinstance(patch, np.ndarray):
            patch = torch.from_numpy(patch / 255.).permute(2, 0, 1).float()
        return patch

    def _get_coords(self) -> List[Coords]:
        """Generate patch coordinates based on image size and overlap."""
        width, height = self.image_size
        stride = int(self.config.size * (1 - self.config.overlap))

        coords = []
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # Adjust coordinates to prevent going out of bounds
                x_adj = min(x, width - self.config.size)
                y_adj = min(y, height - self.config.size)
                coords.append((x_adj, y_adj))

        return coords

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        x, y = self.coords[idx]
        patch = self._get_patch((x, y))

        if self.transforms is not None:
            patch = self.transforms(patch)

        patch = self._normalize_patch(patch)
        return patch, x, y

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, int, int]]) -> Tuple[List[torch.Tensor], List[int], List[int]]:
        """Custom collate function for batching."""
        patches, x_coords, y_coords = zip(*batch)
        return list(patches), list(x_coords), list(y_coords)


class ROI_InferenceDataset(BaseInferenceDataset):
    """Dataset for regular image inference."""

    def __init__(
        self,
        image_path: Union[str, Path],
        patch_config: Optional[PatchConfig] = None,
        transforms: Optional[Union[List[Callable], Callable]] = None
    ) -> None:
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        patch_config = patch_config or PatchConfig()
        super().__init__(patch_config, transforms)

        self._load_image()
        self.coords = self._get_coords()

    def _load_image(self) -> None:
        """Load image and set size."""
        self.image = Image.open(self.image_path).convert('RGB')
        self.image_size = self.image.size

    def _get_patch(self, coords: Coords) -> np.ndarray:
        """Extract patch from regular image."""
        x, y = coords
        patch = self.image.crop((x, y, x + self.config.size, y + self.config.size))
        return np.array(patch)
    


def create_active_map(slide: OpenSlide) -> Tuple[np.ndarray, int]:
    """Create a binary mask of tissue-containing regions in a whole slide image.

    This function generates a low-resolution map indicating regions containing tissue,
    using Otsu thresholding and morphological operations.

    Args:
        slide (OpenSlide): OpenSlide object of the whole slide image

    Returns:
        Tuple[np.ndarray, int]: Binary mask of tissue regions and downsampling factor
    """
    downsamples_int = [int(x) for x in slide.level_downsamples]
    if 32 in downsamples_int:
        ds = 32
    elif 16 in downsamples_int:
        ds = 16

    # get overview image
    level = np.where(np.abs(np.array(slide.level_downsamples)-ds)<0.1)[0][0]
    overview = np.array(slide.read_region(level=level, location=(0,0), size=slide.level_dimensions[level]))
    
    # remove transparent alpha channel 
    alpha_zero_mask = (overview[:, :, 3] == 0)
    overview[alpha_zero_mask, :] = 255
    
    # OTSU
    gray = cv2.cvtColor(overview[:,:,0:3],cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # closing
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    dil = cv2.dilate(thresh, kernel=elem)
    activeMap = cv2.erode(dil, kernel=elem)
    
    return activeMap, ds



class WSI_InferenceDataset(BaseInferenceDataset):
    """Dataset for whole slide image inference."""

    def __init__(
        self,
        slide_path: Union[str, Path],
        patch_config: Optional[PatchConfig] = None,
        transforms: Optional[Union[List[Callable], Callable]] = None
    ) -> None:
        self.slide_path = Path(slide_path)
        if not self.slide_path.exists():
            raise FileNotFoundError(f"Slide not found: {slide_path}")

        patch_config = patch_config or PatchConfig()
        super().__init__(patch_config, transforms)

        self._load_image()
        self.active_map, self.ds = self._create_active_map()
        self.coords = self._get_coords()

    def _load_image(self) -> None:
        """Load slide and set size."""
        self.slide = openslide.open_slide(str(self.slide_path))
        self.image_size = self.slide.dimensions
        self.level_downsample = self.slide.level_downsamples[self.config.level]

    def _create_active_map(self) -> np.ndarray:
        """Create tissue mask for the slide."""
        return create_active_map(self.slide)

    def _get_coords(self) -> List[Coords]:
        """Generate coordinates for tissue-containing regions."""
        coords = super()._get_coords()

        # Filter coordinates based on tissue content
        filtered_coords = []
        for x, y in coords:
            if self._check_tissue_content((x, y)):
                filtered_coords.append((x, y))

        return filtered_coords

    def _check_tissue_content(self, coords: Coords) -> bool:
        """Check if a patch contains sufficient tissue content."""
        x, y = coords
        x_ds = int(x / self.ds)
        y_ds = int(y / self.ds)
        patch_size_ds = int(self.config.size * self.level_downsample / self.ds)

        tissue_content = np.mean(self.active_map[
            y_ds:y_ds + patch_size_ds,
            x_ds:x_ds + patch_size_ds
        ])

        return tissue_content >= self.config.tissue_threshold

    def _get_patch(self, coords: Coords) -> np.ndarray:
        """Extract patch from whole slide image."""
        x, y = coords
        patch = self.slide.read_region(
            location=(x, y),
            level=self.config.level,
            size=(self.config.size, self.config.size)
        ).convert('RGB')
        return np.array(patch)




class Strategy(ABC):
    """Abstract base class defining the interface for inference strategies.

    This class serves as a template for implementing different inference strategies
    for processing images with deep learning models.
    """

    @abstractmethod
    def process_image(self, model: nn.Module, image: str, **kwargs) -> Dict[str, np.ndarray]:
        """Process an image using the specified model.

        Args:
            model (nn.Module): The neural network model to use for inference
            image (str): Path to the image file
            **kwargs: Additional keyword arguments for processing

        Returns:
            Dict[str, np.ndarray]: Dictionary containing inference results
        """
        pass


class Torchvision_Inference(Strategy):
    """Inference strategy for Torchvision-based object detection models.

    This class handles patch-based inference for both regular images and whole slide images,
    with support for various detection models (Faster R-CNN, Mask R-CNN, FCOS, etc.).

    Args:
        model: The detection model to use
        config: Inference configuration parameters
        logger: Optional logger instance
    """
    def __init__(
        self,
        model: nn.Module,
        config: Optional[InferenceConfig] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.model = model
        self.config = config or InferenceConfig()
        self.logger = logger or self._setup_logger()
        self.device = self._setup_device()

    def _setup_logger(self) -> logging.Logger:
        """Initialize logger with appropriate configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _setup_device(self) -> torch.device:
        """Set up and validate the processing device."""
        if self.config.device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available. Using CPU instead.")
            return torch.device('cpu')
        return torch.device(self.config.device)

    def _create_dataloader(
        self,
        image_path: Union[str, Path],
        patch_config: PatchConfig
    ) -> DataLoader:
        """Create appropriate dataloader based on image type."""
        dataset_class = WSI_InferenceDataset if self.config.is_wsi else ROI_InferenceDataset

        try:
            dataset = dataset_class(
                image_path,
                patch_config=patch_config
            )
        except Exception as e:
            self.logger.error(f"Failed to create dataset: {str(e)}")
            raise

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=dataset.collate_fn,
        )

    @torch.no_grad()
    def _process_batch(
        self,
        batch: List[torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        """Process a batch of patches."""
        images = [img.to(self.device) for img in batch]
        try:
            predictions = self.model(images)
            return predictions
        except RuntimeError as e:
            self.logger.error(f"Error during foward pass: {str(e)}")
            raise

    def _post_process_predictions(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        coords: List[Coords]
    ) -> Dict[str, torch.Tensor]:
        """Post-process predictions including coordinate adjustment and NMS."""
        boxes_list = []
        scores_list = []
        labels_list = []

        for pred, (x_orig, y_orig) in zip(predictions, coords):
            if len(pred['boxes']) > 0:
                # Adjust coordinates to original image space
                boxes = pred['boxes'] + torch.tensor(
                    [x_orig, y_orig, x_orig, y_orig],
                    device=pred['boxes'].device
                )               

                boxes_list.append(boxes.cpu())
                scores_list.append(pred['scores'].cpu())
                labels_list.append(pred['labels'].cpu())

        if not boxes_list:
            return {
                'boxes': torch.empty((0, 4), device='cpu'),
                'scores': torch.empty(0, device='cpu'),
                'labels': torch.empty(0, device='cpu')
            }

        # Concatenate all predictions
        boxes = torch.cat(boxes_list)
        scores = torch.cat(scores_list)
        labels = torch.cat(labels_list)

        # Apply NMS per class
        final_boxes = []
        final_scores = []
        final_labels = []

        for label in labels.unique():
            mask = labels == label
            class_boxes = boxes[mask]
            class_scores = scores[mask]

            keep = torch_nms(class_boxes, class_scores, self.config.nms_thresh)

            final_boxes.append(class_boxes[keep])
            final_scores.append(class_scores[keep])
            final_labels.append(labels[mask][keep])

        # Concatenate 
        final_boxes = torch.cat(final_boxes)
        final_scores = torch.cat(final_scores)
        final_labels = torch.cat(final_labels)

        return {
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels
        }

    def process_image(
        self,
        image_path: Union[str, Path],
        patch_config: Optional[PatchConfig] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Process an image using patch-based inference.

        Args:
            image_path: Path to the image or slide file
            patch_config: Configuration for patch extraction
            **kwargs: Additional arguments to override default configs

        Returns:
            Dict containing 'boxes', 'scores', and 'labels' as numpy arrays
        """
        # Update config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        patch_config = patch_config or PatchConfig()

        # Prepare model
        self.model.eval()
        self.model.to(self.device)

        # Create dataloader
        dataloader = self._create_dataloader(image_path, patch_config)

        # Initialize results storage
        all_predictions = []
        all_coords = []

        # Process batches
        with tqdm(dataloader, desc="Processing batches") as pbar:
            for batch_images, batch_x, batch_y in pbar:
                predictions = self._process_batch(batch_images)
                all_predictions.extend(predictions)
                all_coords.extend(zip(batch_x, batch_y))

        # Post-process results
        results = self._post_process_predictions(all_predictions, all_coords)

        # Convert to numpy arrays
        return {
            'boxes': results['boxes'].numpy(),
            'scores': results['scores'].numpy(),
            'labels': results['labels'].numpy()
        }
    

class ImageProcessor:
    """High-level processor for handling image detection tasks.

    This class orchestrates the image processing pipeline, handling both single images
    and batches, with support for various inference strategies.

    Args:
        strategy: An initialized inference strategy
        processor_config: Configuration for the processor
        logger: Optional logger instance
    """
    def __init__(
        self,
        strategy: Strategy,
        processor_config: Optional[ProcessorConfig] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.strategy = strategy
        self.config = processor_config or ProcessorConfig()
        self.logger = logger or self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Initialize logger with appropriate configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _save_results(
        self,
        results: Dict[str, np.ndarray],
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """Save detection results to a JSON file."""
        output_dir = output_dir or self.config.save_dir
        if output_dir:
            try:
                output_path = Path(output_dir) / f"{Path(image_path).stem}_detections.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Convert numpy arrays to lists for JSON serialization
                serializable_results = {
                    'boxes': results['boxes'].tolist(),
                    'scores': results['scores'].tolist(),
                    'labels': results['labels'].tolist(),
                    'image_path': str(image_path)
                }

                with open(output_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
            
            except Exception as e:
                self.logger.error(f"Error saving results {image_path}: {str(e)}.")


    def should_proceed(
            self, 
            image_path: Union[str, Path],
            output_dir: Optional[Union[str, Path]] = None
    ) -> bool:
        """Checks if image should be processed."""
        result_path = Path(output_dir) / f"{Path(image_path).stem}_detections.json"
        if result_path.exists() and not self.config.overwrite:
            return False
        elif result_path.exists() and self.config.overwrite:
            return True
        else:
            return True




    def process_single(
        self,
        image_path: Union[str, Path],
        patch_config: Optional[PatchConfig] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Process a single image.

        Args:
            image_path: Path to the image file
            patch_config: Configuration for patch extraction
            output_dir: Directory to save results
            **kwargs: Additional arguments passed to the strategy

        Returns:
            Dictionary containing detection results
        """
        # Run inference using strategy
        results = self.strategy.process_image(
            image_path,
            patch_config=patch_config,
            **kwargs
        )
        # Save results if configured
        if output_dir or self.config.save_dir:
            self._save_results(results, image_path, output_dir)

        return results 
        


    def process_multi(
        self,
        image_paths: List[Union[str, Path]],
        patch_config: Optional[PatchConfig] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Process multiple images.

        Args:
            image_paths: List of paths to image files
            patch_config: Configuration for patch extraction
            output_dir: Directory to save results
            **kwargs: Additional arguments passed to the strategy

        Returns:
            Dictionary mapping image paths to their detection results
        """
        results = {}
        
        # Run over all images
        with tqdm(sorted(image_paths), desc="Processing images") as pbar:
            for image_path in pbar:
                pbar.set_description(f"Processing {Path(image_path).name}")
                
                # Process each image individually 
                results[str(image_path)] = self.process_single(
                    image_path,
                    patch_config=patch_config,
                    output_dir=output_dir,
                    **kwargs
                )
        return results
    


def load_model_from_config(config_file: str, det_thresh: float=None):
    """Load model from configuration file."""
    try:
        config = ConfigCreator.load(config_file)
        model = ModelFactory.load(config, det_thresh)
        return model, config
    except Exception as e:
        print(f"Error loading the model: {str(e)}")
        raise 


def setup_inference(
    model: torch.nn.Module,
    is_wsi: bool,
    batch_size: int,
    num_workers: int,
    nms_thresh: float, 
    device: str,
    patch_size: int,
    overlap: float,
    overwrite: bool,
    logger: logging.Logger = None,
) -> tuple:
    """Setup inference components."""
    inference_config = InferenceConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        nms_thresh=nms_thresh,
        device=device,
        is_wsi=is_wsi
    )

    patch_config = PatchConfig(
        size=patch_size,
        overlap=overlap
    )

    processor_config = ProcessorConfig(
        overwrite=overwrite
    )

    strategy = Torchvision_Inference(
        model=model, 
        config=inference_config,
        logger=logger
        )
    
    processor = ImageProcessor(
        strategy=strategy,
        processor_config=processor_config, 
        logger=logger
        )

    return processor, patch_config