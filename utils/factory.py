import yaml
import torch

from dataclasses import dataclass
from typing import Any, Dict, List

from .litmodel import BaseDetectionModule
from .model import (
    make_retinanet_model,
    make_maskrccn_model,
    make_fcos_model,
    make_fasterrcnn_model,
)


def tuple_constructor(loader, node):
    # Load the sequence of values from the YAML node
    values = loader.construct_sequence(node)
    # Return a tuple constructed from the sequence
    return tuple(values)

# Register the constructor with PyYAML
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)



@dataclass(kw_only=True)
class ModelConfig:
    """Base configuration class for object detection models.

    This class serves as a base configuration container for all detection models,
    storing common parameters and providing methods for saving/loading configurations.

    Attributes:
        model_name (str): Name identifier for the model
        detector (str): Type of detector (e.g., 'FasterRCNN', 'RetinaNet')
        backbone (str): Backbone architecture (e.g., 'resnet50')
        checkpoint (str): Path to model checkpoint
        det_thresh (float): Detection confidence threshold
        num_classes (int): Number of object classes
        extra_blocks (bool): Whether to use extra FPN blocks
        weights (str, optional): Path to pretrained weights
        returned_layers (List[int], optional): Specific layers to return from backbone
        patch_size (int, optional): Size of input patches. Defaults to 512
    """
    model_name: str 
    detector: str
    backbone: str 
    checkpoint: str 
    det_thresh: float 
    num_classes: int 
    extra_blocks: bool
    weights: str = None
    returned_layers: List[int] = None
    patch_size: int = 512
    means: List[float] = None
    stds: List[float] = None

    def update(self, new: Dict[str, Any]) -> None:
        """Update configuration parameters with new values.

        Args:
            new (Dict[str, Any]): Dictionary of parameters to update
        """
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save(self, filepath: str) -> None:
        """Save configuration to a YAML file.

        Args:
            filepath (str): Path where to save the configuration
        """
        with open(filepath, 'w') as file:
            yaml.dump(self.__dict__, file)

    @classmethod
    def load(cls, filepath: str) -> 'ModelConfig':
        """Load configuration from a YAML file.

        Args:
            filepath (str): Path to the configuration file

        Returns:
            ModelConfig: Loaded configuration object
        """
        with open(filepath, 'r') as file:
            config_dict = yaml.load(file, Loader=yaml.SafeLoader)
        return cls(**config_dict)

@dataclass(kw_only=True)
class FasterRCNN_Config(ModelConfig):
    """Configuration class specific to Faster R-CNN models.

    Extends ModelConfig with Faster R-CNN specific parameters.

    Additional Attributes:
        anchor_sizes (List[int], optional): Sizes of anchor boxes
        anchor_ratios (List[float], optional): Aspect ratios of anchor boxes
    """
    anchor_sizes: List[int] = None
    anchor_ratios: List[float] = None

@dataclass(kw_only=True)
class MaskRCNN_Config(ModelConfig):
    """Configuration class specific to Mask R-CNN models.

    Extends ModelConfig with Mask R-CNN specific parameters.

    Additional Attributes:
        anchor_sizes (List[int], optional): Sizes of anchor boxes
        anchor_ratios (List[float], optional): Aspect ratios of anchor boxes
    """
    anchor_sizes: List[int] = None
    anchor_ratios: List[float] = None

@dataclass(kw_only=True)
class RetinaNet_Config(ModelConfig):
    """Configuration class specific to RetinaNet models.

    Extends ModelConfig with RetinaNet specific parameters.

    Additional Attributes:
        anchor_sizes (List[int], optional): Sizes of anchor boxes
        anchor_ratios (List[float], optional): Aspect ratios of anchor boxes
    """
    anchor_sizes: List[int] = None
    anchor_ratios: List[float] = None

@dataclass(kw_only=True)
class FCOS_Config(ModelConfig):
    """Configuration class specific to FCOS models.

    Extends ModelConfig without additional parameters.
    """


CONFIG_MAPPING = {
        'MaskRCNN': MaskRCNN_Config,
        'FasterRCNN': FasterRCNN_Config,
        'RetinaNet': RetinaNet_Config,
        'FCOS': FCOS_Config
    }


MODEL_MAPPINGS = {
        'MaskRCNN': make_maskrccn_model,
        'FasterRCNN': make_fasterrcnn_model, 
        'RetinaNet': make_retinanet_model,
        'FCOS': make_fcos_model
    }




class ConfigCreator:
    """Factory class for creating model configurations.

    This class provides static methods for creating and loading model configurations
    based on the detector type.
    """

    @staticmethod
    def create(settings: Dict[str, Any]) -> ModelConfig:
        """Create a model configuration from settings dictionary.

        Args:
            settings (Dict[str, Any]): Dictionary containing model configuration parameters

        Raises:
            ValueError: If the specified detector type is not supported

        Returns:
            ModelConfig: Appropriate configuration object for the specified detector
        """
        detector = settings['detector']
        if detector not in CONFIG_MAPPING:
            raise ValueError(f"Model {detector} not supported.")
        return CONFIG_MAPPING[detector](**settings)

    @staticmethod
    def load(filepath: str) -> ModelConfig:
        """Load a model configuration from a file.

        Args:
            filepath (str): Path to the configuration file

        Raises:
            ValueError: If the model type cannot be determined from the filepath

        Returns:
            ModelConfig: Loaded configuration object
        """
        for name, config in CONFIG_MAPPING.items():
            if name in filepath:
                return config.load(filepath)
        raise ValueError(f"Model {filepath} not recognized.")

class ModelFactory:
    """Factory class for creating and loading detection models.

    This class provides static methods for instantiating detection models
    based on configuration parameters.
    """

    @staticmethod
    def create(
            model_name: str,
            model_kwargs: Dict[str, Any] = None,
            module_kwargs: Dict[str, Any] = None) -> BaseDetectionModule:
        """Create a new detection model instance.

        Args:
            model_name (str): Name of the model architecture
            model_kwargs (Dict[str, Any], optional): Model-specific parameters
            module_kwargs (Dict[str, Any], optional): Lightning module parameters

        Raises:
            ValueError: If the specified model name is not recognized

        Returns:
            BaseDetectionModule: Instantiated detection model
        """
        if model_name not in MODEL_MAPPINGS:
            raise ValueError(f"Model {model_name} not recognized.")

        if model_kwargs is None:
            model_kwargs = {}
        if module_kwargs is None:
            module_kwargs = {}

        model = MODEL_MAPPINGS[model_name](**model_kwargs)
        return BaseDetectionModule(model, **module_kwargs)

    @staticmethod
    def load(
            config: ModelConfig,
            det_thresh: float = None,
            **kwargs) -> torch.nn.Module:
        """Load a detection model from a configuration and checkpoint.

        Args:
            config (ModelConfig): Model configuration object
            det_thresh (float, optional): Detection threshold to override config
            **kwargs: Additional keyword arguments for model loading

        Returns:
            torch.nn.Module: Loaded detection model
        """
        if det_thresh is None:
            det_thresh = config.det_thresh

        model_func = MODEL_MAPPINGS[config.detector]
        model_kwargs = {
            'backbone': config.backbone,
            'det_thresh': det_thresh,
            'num_classes': config.num_classes,
            'extra_blocks': config.extra_blocks,
            'returned_layers': config.returned_layers,
            'weights': config.weights,
            'patch_size': config.patch_size,
            'means': config.means,
            'stds': config.stds
        }

        if isinstance(config, (FasterRCNN_Config, MaskRCNN_Config, RetinaNet_Config)):
            model_kwargs.update({
                'anchor_sizes': config.anchor_sizes,
                'anchor_ratios': config.anchor_ratios
            })

        return BaseDetectionModule.load_from_checkpoint(
            model=model_func(**model_kwargs),
            checkpoint_path=config.checkpoint,
            strict=False
        )
    



