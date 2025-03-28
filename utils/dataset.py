from typing import Callable, Dict, List, Tuple, Union
import numpy as np
import pandas as pd 
import openslide 
import torch 

from openslide import OpenSlide
from dataclasses import dataclass, field
from pathlib import Path
from numpy.random import randint, choice 
from tqdm import tqdm 
from torch import Tensor
from torch.utils.data import Dataset

Coords = Tuple[int, int]



@dataclass
class SlideObject:
    """A class for handling whole slide images and extracting patches.

    This class provides an interface for loading and accessing whole slide images (WSI),
    with methods to extract patches at specified coordinates and pyramid levels.

    Attributes:
        slide_path (Union[str, Path]): Path to the whole slide image file
        annotations (pd.DataFrame): Bounding box annotations. Must have columns ['label', 'xmin', 'ymin', 'xmax',  'ymax']. Defaults to None
        domain (Union[str, int]): Domain information of the slide. Defaults to None
        size (Union[int, float]): Size of patches to extract (width=height). Defaults to 512
        level (Union[int, float]): Pyramid level for patch extraction. Defaults to 0
        slide (OpenSlide): OpenSlide object for the WSI (automatically initialized)
    """

    slide_path: Union[str, Path]
    annotations: pd.DataFrame = None
    domain: Union[str, int] = None
    size: Union[int, float] = 512
    level: Union[int, float] = 0

    slide: OpenSlide = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the OpenSlide object after instance creation.

        This method is automatically called after the dataclass initialization
        to create the OpenSlide object from the provided slide path.
        """
        self.slide = openslide.open_slide(str(self.slide_path))

    @property
    def patch_size(self) -> Coords:
        """Get the dimensions of patches to be extracted.

        Returns:
            Coords: Tuple of (width, height) for patches, both equal to self.size
        """
        return (self.size, self.size)

    @property
    def slide_size(self) -> Coords:
        """Get the dimensions of the slide at the current pyramid level.

        Returns:
            Coords: Tuple of (width, height) of the slide at the specified level
        """
        return self.slide.level_dimensions[self.level]

    def load_image(self, coords: Coords) -> np.ndarray:
        """Extract a patch from the slide at the specified coordinates.

        Args:
            coords (Coords): Tuple of (x, y) coordinates in the slide's coordinate space

        Returns:
            np.ndarray: RGB image patch as a numpy array with shape (H, W, 3)
        """
        patch = self.slide.read_region(location=coords, level=self.level, size=self.patch_size).convert('RGB')
        return np.array(patch)
    


    def load_labels(self, coords: Coords, label: int=None, delta: int=25) -> Tuple[np.ndarray, np.ndarray]:
        """Returns annotations for a given set of coordinates. Transforms the slide annotations to patch coordinates.

        Args:
            coords (Coords): Top left patch coordinates. 
            label (int, optional): Annotations to return. Defaults to None.
            delta (int, optional): Delta to ensure that all cells are well covered by patch coordinates. Defaults to 25.

        Returns:
            Tuple (np.ndarray, np.ndarray): Boxes in patch coordinates and the labels. 
        """
        assert self.annotations is not None, 'No annotations available.'
        assert isinstance(self.annotations, pd.DataFrame), f'Annotations must be of type pd.DataFrame, but found {type(self.annotations)}.'
        assert pd.Series(['xmin', 'ymin', 'xmax', 'ymax']).isin(self.annotations.columns).all(), f'DataFrame must have columns xmin, ymin, xmax, ymax.'
        
        # Filter annotations
        if label is not None:
            annos = self.annotations.query('label == @label')[['xmin', 'ymin', 'xmax', 'ymax']]
            labels = self.annotations.query('label == @label')['label']
        else:
            annos = self.annotations[['xmin', 'ymin', 'xmax', 'ymax']]
            labels = self.annotations['label']

        # Empty image 
        if len(annos) == 0: 
            boxes = np.zeros((0, 4), dtype=np.int64)
            labels = np.zeros(0, dtype=np.int64) 

        else:
            # Filter annotations by coordinates
            x, y = coords
            mask = ((x+delta) < annos.xmin) & (annos.xmax < (x+self.size-delta)) & \
                ((y+delta) < annos.ymin) & (annos.ymax < (y+self.size-delta))

            boxes = annos[mask].to_numpy()
            boxes[:, [0, 2]] -= x
            boxes[:, [1, 3]] -= y

            # Get labels
            labels = labels[mask].to_numpy()

        return boxes, labels




class DetectionDataset(Dataset):
    """A custom Dataset class for object detection tasks on whole slide images.

    This class handles the loading and preprocessing of whole slide images and their corresponding
    annotations for object detection tasks. It supports various sampling strategies and box formats.

    Args:
        img_dir (Union[Path, str]): Directory containing the image files.
        dataset (Union[pd.DataFrame, str, Path]): DataFrame or path to CSV containing annotations.
        box_format (str, optional): Format of bounding boxes ('xyxy' or 'cxcy'). Defaults to 'cxcy'.
        label_col (str, optional): Column name for object labels. Defaults to 'label'.
        domain_col (str, optional): Column name for domain information. Defaults to 'tumortype'.
        filename_col (str, optional): Column name for image filenames. Defaults to 'filename'.
        sampling_strategy (str, optional): Strategy for sampling images ('domain_based' or 'default'). Defaults to 'domain_based'.
        num_samples (int, optional): Number of samples per epoch. Defaults to 1024.
        fg_label (int, optional): Label value for foreground objects. Defaults to 1.
        fg_prob (float, optional): Probability of sampling foreground objects. Defaults to 0.5.
        arb_prob (float, optional): Probability of sampling arbitrary locations. Defaults to 0.25.
        patch_size (int, optional): Size of image patches. Defaults to 512.
        level (int, optional): Magnification level for whole slide images. Defaults to 0.
        radius (int, optional): Radius around center points for box generation. Defaults to 25.
        transforms (Union[List[Callable], Callable], optional): Transformations to apply to images and boxes.

    Raises:
        ValueError: If box_format is not 'xyxy' or 'cxcy'.
        ValueError: If required columns are missing in the dataset.
    """
    def __init__(
            self,
            img_dir: Union[Path, str], 
            dataset: Union[pd.DataFrame, str, Path], 
            box_format: str = 'cxcy',
            label_col: str = 'label', 
            domain_col: str = 'tumortype',
            filename_col: str = 'filename',
            sampling_strategy: str = 'domain_based',
            num_samples: int = 1024,
            fg_label: int = 1,
            fg_prob: float = 0.5,
            arb_prob: float = 0.25,
            patch_size: int = 512, 
            level: int = 0,
            radius: int = 25,
            transforms: Union[List[Callable], Callable] = None) -> None:
        

        allowed_box_formats = ("xyxy", "cxcy")
        if box_format not in allowed_box_formats:
                raise ValueError(f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}")
        
        self.box_format = box_format
        self.img_dir = Path(img_dir)
        self.dataset = dataset
        self.label_col = label_col
        self.domain_col = domain_col
        self.filename_col = filename_col
        self.sampling_strategy = sampling_strategy
        self.num_samples = num_samples
        self.fg_label = fg_label
        self.fg_prob =  fg_prob
        self.arb_prob = arb_prob
        self.patch_size = patch_size
        self.transforms = transforms
        self.radius = radius

        self._init_slide_objects()
        self.create_samples()


    def _init_slide_objects(self) -> None:
        """Initializes SlideObjects from given dataset.

        Raises:
            ValueError: If filename_col does not exist.
            ValueError: If label_col does not exist.
            ValueError: If box_format is xyxy and columns [xmin, ymin, xmax, ymax] do not exist.
            ValueError: If box_format is cxcy and columns [x, y] do not exist.
        """
        if not isinstance(self.dataset, pd.DataFrame):
            self.dataset = pd.read_csv(self.dataset)

        if self.filename_col not in self.dataset.columns:
            raise ValueError(f"Column '{self.filename_col}' with filenames (e.g. '012.tiff', '234.tiff') does not exist.")
        
        if self.label_col not in self.dataset.columns:
            raise ValueError(f"Column '{self.label_col}' with labels (e.g. 'Tumor cell'=1 and 'Background cell'=2) does not exist.")
        
        if self.box_format == "xyxy":
            if not pd.Series(['xmin', 'ymin', 'xmax', 'ymax']).isin(self.dataset.columns).all():
                raise ValueError(f"DataFrame expected to have columns ('xmin', 'ymin', 'xmax', 'ymax').")
            
        elif self.box_format == "cxcy":
            if not pd.Series(['x', 'y']).isin(self.dataset.columns).all():
                 raise ValueError(f"DataFrame expected to have columns ('x', 'y').")   

        if self.domain_col is not None:
            if self.domain_col not in self.dataset.columns:
                raise ValueError(f"Column '{self.domain_col}' with domain information (e.g. 'HNSCC', 'GC') does not exist.")   

        # transform boxes
        if self.box_format == 'cxcy':
            self.dataset = self.dataset.assign(xmin=self.dataset['x'] - self.radius)
            self.dataset = self.dataset.assign(ymin=self.dataset['y'] - self.radius)
            self.dataset = self.dataset.assign(xmax=self.dataset['x'] + self.radius)
            self.dataset = self.dataset.assign(ymax=self.dataset['y'] + self.radius)

        # select columns to get data
        columns = [self.filename_col, self.label_col, 'xmin', 'ymin', 'xmax', 'ymax']

        if self.domain_col is not None:
            columns.extend([self.domain_col])
            domains = list(self.dataset[self.domain_col].unique())
            self.domains = domains
            self.idx_to_domain = {idx: d for idx, d in enumerate(domains)}
            self.domain_to_files = self.dataset.groupby(self.domain_col)[self.filename_col].apply(lambda x: list(set(x))).to_dict()

        # get unique filenames 
        fns = self.dataset[self.filename_col].unique().tolist()

        # initialize slideobjects
        slide_objects = {}
        for fn in tqdm(fns, desc='Initializing slide objects'):
            slide_path = self.img_dir.joinpath(fn)
            annos = self.dataset.query('filename == @fn')[columns] 
            slide_objects[fn] = SlideObject(
                slide_path=slide_path,
                annotations=annos,
                domain=annos[self.domain_col].unique().item(),
                size=self.patch_size
                )
            
        # store slideobjects and dataset information
        self.slide_objects = slide_objects
        self.filenames = fns
        self.classes = self.dataset[self.label_col].unique().tolist()




    def _sample_coords(self, fn: str, fg_prob: float=None, arb_prob: float=None) -> Dict[str, Coords]:
        """Samples patch coordinates from a slide using different strategies.

        Args:
            fn (str): Filename of the slide to sample from.
            fg_prob (float, optional): Override probability for foreground sampling.
            arb_prob (float, optional): Override probability for arbitrary sampling.

        Returns:
            Dict[str, Coords]: Dictionary containing filename and sampled coordinates.
        """

        # set sampling probabilities
        fg_prob = self.fg_prob if fg_prob is None else fg_prob
        arb_prob = self.arb_prob if arb_prob is None else arb_prob

        # get slide object
        sl = self.slide_objects[fn]

        # get dims
        slide_width, slide_height = sl.slide_size
        patch_width, patch_height = sl.patch_size

        # create sampling probabilites
        sample_prob = np.array([self.arb_prob, self.fg_prob, 1-self.fg_prob-self.arb_prob])

        # sample case from probabilites (0 = random, 1 = fg, 2 = imposter)
        case = choice(3, p=sample_prob)

        # sample center coordinates
        if case == 0:       
            # random patch 
            x = randint(patch_width / 2, slide_width-patch_width / 2)
            y = randint(patch_height / 2, slide_height-patch_height / 2)

        elif case == 1:     
            # filter foreground cases
            mask = sl.annotations[self.label_col] == 1

            if np.count_nonzero(mask) == 0:
                # no annotations available -> random patch
                x = randint(patch_width / 2, slide_width-patch_width / 2)
                y = randint(patch_height / 2, slide_height-patch_height / 2)
            else:       
                # get annotations
                annos = sl.annotations[['xmin', 'ymin', 'xmax' ,'ymax']][mask]

                # sample foreground class
                idx = randint(annos.shape[0])
                xmin, ymin, xmax, ymax = annos.iloc[idx]
                x = (xmin + xmax) / 2
                y = (ymin + ymax) / 2

        elif case == 2:
            # sample imposter
            mask = sl.annotations[self.label_col] == 2

            if np.count_nonzero(mask) == 0:
                # no imposter available -> random patch
                x = randint(patch_width / 2, slide_width-patch_width / 2)
                y = randint(patch_height / 2, slide_height-patch_height / 2)

            else:
                # get annotations
                annos = sl.annotations[['xmin', 'ymin', 'xmax' ,'ymax']][mask]
                # sample imposter
                idx = randint(annos.shape[0])
                xmin, ymin, xmax, ymax = annos.iloc[idx]
                x = (xmin + xmax) / 2
                y = (ymin + ymax) / 2 


        # set offsets
        offset_scale = 0.5
        xoffset = randint(-patch_width, patch_width) * offset_scale
        yoffset = randint(-patch_height, patch_height) * offset_scale

        # shift coordinates and return top left corner
        x = int(x - patch_width / 2 + xoffset) 
        y = int(y - patch_height / 2 + yoffset)

        # avoid black borders
        if x + patch_width > slide_width:
            x = slide_width - patch_width
        elif x < 0:
            x = 0
        
        if y + patch_height > slide_height:
            y = slide_height - patch_height
        elif y < 0:
            y = 0


        return {'file': fn, 'coords': (x, y)}


    def _sample_with_equal_probability(self) -> np.ndarray:
        """Sampling strategy that samples with equal probability from all slides."""
        slide_ids = choice(list(self.slide_objects.keys()), size=self.num_samples, replace=True)
        return slide_ids
    

    def _sample_based_on_slides_per_domain(self) -> np.ndarray:
        """Sampling strategy that samples with equal probabilities based on total number of slides per domain."""
        assert self.domain_col is not None, 'domain_col needs to be available for this sampling strategy.'
        N = len(self.slide_objects)
        domains, counts = np.unique([v.domain for v in self.slide_objects.values()], return_counts=True)
        weights = N / counts
        weights = np.array([weights[domains == v.domain] for v in self.slide_objects.values()])
        weights = (weights / weights.sum()).reshape(-1)
        
        slide_ids = choice(list(self.slide_objects.keys()), size=self.num_samples, p=weights, replace=True)
        return slide_ids 


    def _sample_slides(self) -> np.ndarray:
        """Method to sample slide ids."""
        if self.sampling_strategy == 'default':
            return self._sample_with_equal_probability()
        elif self.sampling_strategy == 'domain_based':
            return self._sample_based_on_slides_per_domain()
        else:
            raise ValueError(f'Unsupported sampling strategy: {self.sampling_strategy}. Use onf of [default, domain_absed]')


    def create_samples(self) -> None:
        """Method to create training samples for one pseudo epoch."""

        # get slide ids 
        slide_ids = self._sample_slides()

        # get training samples 
        samples = dict()
        for sample_id, slide_id in enumerate(slide_ids):
            samples[sample_id] = self._sample_coords(slide_id, self.fg_prob, self.arb_prob)

        # store samples 
        self.samples = samples 



    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx) -> Tuple[Tensor, Dict[str, Tensor]]:
         # get sample
        sample = self.samples[idx]

        # extract file and coords
        file, coords = sample['file'], sample['coords']

        # get slide object
        slide = self.slide_objects[file]

        # load image and boxes
        img = slide.load_image(coords)
        boxes, labels = slide.load_labels(coords, label=1)

        if self.transforms is not None:
            if boxes.shape[0] > 0:
                transformed = self.transforms(image=img, bboxes=boxes, class_labels=labels)
                img = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['class_labels']
                if len(boxes) == 0:
                    boxes = np.zeros((0, 4) , dtype=np.int64)
                    labels = np.zeros((0,), dtype=np.int64)
            else:
                boxes = np.zeros((0, 4) , dtype=np.int64)
                labels = np.zeros((0,), dtype=np.int64)
        else:
            if boxes.shape[0] > 0:
                labels = np.ones(boxes.shape[0], dtype=np.int64)
            else:
                boxes = np.zeros((0, 4) , dtype=np.int64)
                labels = np.zeros((0,), dtype=np.int64)

        # convert to tensor
        img = torch.from_numpy(img / 255.).permute(2, 0, 1).type(torch.float32) 
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
        }

        # add domain information
        if self.domain_col is not None:
            domain = slide.annotations[self.domain_col].unique().item()
            domain_label = self.domains.index(domain)
            target.update(
                {'domain': torch.as_tensor(domain_label, dtype=torch.int64)}
            )


        return img, target
    
    
    @staticmethod
    def collate_fn(batch):
        """Collate function for the data loader."""
        images = list()
        targets = list()

        for b in batch:
            images.append(b[0])
            targets.append(b[1])
            
        images = torch.stack(images, dim=0)

        return images, targets
    