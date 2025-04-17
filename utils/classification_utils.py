from typing import List, Tuple, Union

import albumentations as A
import logging
import numpy as np 
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim


from pathlib import Path 
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision.models import get_model
from tqdm.notebook import tqdm 


MODELS = [
        'resnet18',
        'resnet50',
        'vit_b_16',
        'convnext_small',
        'efficientnet_b0',
        'efficientnet_v2_s',
        'densenet121'
    ]
    

class MitosisClassifier(nn.Module):
    """Mitosis Classifier

    Uses torchvision's pre-trained models to load a standard classifiation network.
    Adapts the final classification layer to have a single output node for 
    binary classifiction of mitosis vs. non-mitosis.

    Args:
        model (str): Model type (e.g. resnet18, resent50).
        weights (str, optional): Weight type. Defaults to 'DEFAULT'.
    """
    def __init__(
        self,
        model: str,
        weights: str = 'DEFAULT',
        num_classes: int = 1) -> None:
        super().__init__()

        self.model = model 
        self.weights = weights
        self.num_classes = num_classes

        self.classifier = self.build_model()


    def build_model(self):
        """Model constructor function.

        Loads a torchvision model and its pretrained weights. Then adapts the final layer
        to the binary classificaiton task.
        """
        assert self.weights in ['DEFAULT', 'IMAGENET1K_V1', 'IMAGENET1K_V2', 'None'], \
            'Unsupported weights for {}. Should be one of [DEFAULT, IMAGENET1K_V1, IMAGENET1K_V2, None]'.format(self.weights)
        
        weights = None if self.weights == 'None' else self.weights

        assert self.model in MODELS, \
            'Unsupported model for {}. Should be one of {}'.format(self.model, MODELS)

        # load model and/or pretrained weights
        classifier = get_model(self.model, weights=weights)

        # adapt final layer 
        if self.model in ['resnet18', 'resnet50']:
            classifier.fc = nn.Linear(classifier.fc.in_features, self.num_classes)
        elif self.model == 'vit_b_16':
            classifier.heads.head = nn.Linear(classifier.heads.head.in_features, self.num_classes)
        elif self.model == 'densenet121':
            classifier.classifier = nn.Linear(classifier.classifier.in_features, self.num_classes)
        else:
            classifier.classifier[-1] = nn.Linear(classifier.classifier[-1].in_features, self.num_classes)

        return classifier


    def forward(self, x):
        """Foward pass

        Args:
            x (Tensor): Tensor with shape [B, 3, W, H]

        Returns:
            Tuple[Tensor]: logits, probabilities and labels
        """
        logits = self.classifier(x)
        logits = logits.squeeze()
        Y_prob = torch.sigmoid(logits)
        Y_hat = (Y_prob > 0.5).float()
        return logits, Y_prob, Y_hat




class ClassificationDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label




class MitosisTrainer:
    """
    A simple trainer class for handling mitosis classification model training and evaluation.

    This class implements k-fold cross-validation training, validation, and test set evaluation
    for mitosis classification using PyTorch.

    Args:
        model_name (str): Name of the model architecture to use
        weights (str): Pre-trained weights to initialize the model
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training and evaluation
        num_folds (int): Number of folds for k-fold cross-validation
        device (torch.device): Device to run the model on (CPU or CUDA)
        criterion: Loss function for training
        classes (list): List of class names
        logger: Logger instance for tracking training progress
    """
    def __init__(
        self, 
        model_name: str, 
        weights: str, 
        experiment_dir: str,
        num_epochs: int=2, 
        batch_size: int=128, 
        lr: float=1e-4, 
        num_folds: int=5,
        ):
        self.model_name = model_name
        self.weights = weights
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_folds = num_folds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCEWithLogitsLoss()
        self.classes = ["Atypical", "Normal"]
        self.experiment_dir = experiment_dir 


    @property
    def train_transform(self):
        transform = A.Compose([
                A.D4(p=1),
                A.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.1, 0.1), p=0.5),
                A.Defocus(radius=(1,3), p=0.3),
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
                ])
        return transform


    @property
    def val_transform(self):
        transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        return transform



    def setup_experiment(self):
        """Set up experiment directory to save log files and model checkpoints"""
        self.exp_dir = Path(self.experiment_dir)
        self.exp_dir.mkdir(exist_ok=True, parents=True)
        self.log_file = self.exp_dir / 'training.log'
        logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(self.log_file),
                            logging.StreamHandler()
                        ]) 
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Created experiment directoy: {str(self.exp_dir)}.")


    def prepare_data_loaders(self, images, labels, transform, is_training=True):
        """
        Prepare DataLoader objects for training or evaluation.

        Args:
            images (list): List of image paths or data
            labels (list): List of corresponding labels
            transform: Transformations to apply to the images
            is_training (bool, optional): Whether this is for training (uses sampler) or evaluation.
                                        Defaults to True.

        Returns:
            DataLoader: PyTorch DataLoader object
        """
        dataset = ClassificationDataset(images, labels, transform=transform)

        if is_training:
            # Compute class weights for WeightedRandomSampler
            class_counts = [0, 0]
            for lbl in labels:
                class_counts[lbl] += 1

            class_weights = [1.0 / max(count, 1) for count in class_counts]
            sample_weights = [class_weights[lbl] for lbl in labels]

            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=8,
                pin_memory=True
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )

        return loader


    def train_epoch(self, model, train_loader, optimizer):
        """
        Train the model for one epoch.

        Args:
            model: The neural network model
            train_loader: DataLoader for training data
            optimizer: Optimizer for updating model parameters

        Returns:
            tuple: (average training loss, predictions, targets)
        """
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for images_batch, labels_batch in tqdm(train_loader, desc="Training"):
            images_batch, labels_batch = images_batch.to(self.device), labels_batch.to(self.device)

            optimizer.zero_grad()
            logits, Y_prob, Y_hat = model(images_batch)

            loss = self.criterion(logits, labels_batch.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(Y_hat.cpu().numpy())
            train_targets.extend(labels_batch.cpu().numpy())

        return train_loss / len(train_loader), train_preds, train_targets


    def evaluate(self, model, data_loader, phase="Validation"):
        """
        Evaluate the model on the provided data.

        Args:
            model: The neural network model
            data_loader: DataLoader for evaluation data
            phase (str, optional): Phase name for progress bar. Defaults to "Validation".

        Returns:
            tuple: (average loss, predictions, targets)
        """
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images_batch, labels_batch in tqdm(data_loader, desc=phase):
                images_batch, labels_batch = images_batch.to(self.device), labels_batch.to(self.device)

                logits, Y_prob, Y_hat = model(images_batch)
                loss = self.criterion(logits, labels_batch.float())

                total_loss += loss.item()
                all_preds.extend(Y_hat.cpu().numpy())
                all_targets.extend(labels_batch.cpu().numpy())

        return total_loss / len(data_loader), all_preds, all_targets


    def train_fold(self, fold, train_images, train_labels, val_images, val_labels):
        """
        Train and validate the model for one fold.

        Args:
            fold (int): Current fold number
            train_images (list): Training images for this fold
            train_labels (list): Training labels for this fold
            val_images (list): Validation images for this fold
            val_labels (list): Validation labels for this fold

        Returns:
            tuple: (best validation accuracy, path to best model checkpoint)
        """
        # Build DataLoaders
        train_loader = self.prepare_data_loaders(train_images, train_labels, self.train_transform, is_training=True)
        val_loader = self.prepare_data_loaders(val_images, val_labels, self.val_transform, is_training=False)

        # Initialize model, optimizer, scheduler
        model = MitosisClassifier(self.model_name, self.weights).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-7)

        best_val_balanced_accuracy = 0.0
        best_model_path = self.exp_dir / f'MIDOG25_binary_classification_baseline_fold{fold + 1}_best.pth'

        for epoch in range(self.num_epochs):
            # Training
            train_loss, train_preds, train_targets = self.train_epoch(model, train_loader, optimizer)
            train_balanced_accuracy = balanced_accuracy_score(train_targets, train_preds)

            # Validation
            val_loss, val_preds, val_targets = self.evaluate(model, val_loader, "Validation")
            val_balanced_accuracy = balanced_accuracy_score(val_targets, val_preds)

            # Save best model
            if val_balanced_accuracy > best_val_balanced_accuracy:
                best_val_balanced_accuracy = val_balanced_accuracy
                torch.save(model.state_dict(), best_model_path)

            scheduler.step()

            # Logging
            self.logger.info(
                f"Fold {fold + 1}, Epoch {epoch + 1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Balanced Acc: {train_balanced_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Balanced Acc: {val_balanced_accuracy:.4f}"
            )

        return best_val_balanced_accuracy, best_model_path


    def train_and_evaluate(self, train_images, train_labels, test_images=None, test_labels=None):
        """
        Perform k-fold cross-validation training and optional test set evaluation.

        Args:
            train_images (list): Complete set of training images
            train_labels (list): Complete set of training labels
            test_images (list, optional): Test set images. Defaults to None.
            test_labels (list, optional): Test set labels. Defaults to None.

        Returns:
            list: List of best validation accuracies for each fold
        """
        # Setup logging 
        self.setup_experiment()

        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold_accuracies = []
        best_model_paths = []

        # K-fold Cross Validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_images, train_labels)):
            self.logger.info(f"Starting Fold {fold + 1}")

            # Split data
            fold_train_images = [train_images[i] for i in train_idx]
            fold_train_labels = [train_labels[i] for i in train_idx]
            fold_val_images = [train_images[i] for i in val_idx]
            fold_val_labels = [train_labels[i] for i in val_idx]

            # Train fold
            best_acc, best_model_path = self.train_fold(
                fold,
                fold_train_images,
                fold_train_labels,
                fold_val_images,
                fold_val_labels
            )

            fold_accuracies.append(best_acc)
            best_model_paths.append(best_model_path)
            self.logger.info(f"Fold {fold + 1} - Best Validation Balanced Accuracy: {best_acc:.4f}")

        self.logger.info(f"Average Validation Balanced Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")


        # Test set evaluation
        if test_images is not None and test_labels is not None:
            self.logger.info("Evaluating on test set...")
            test_loader = self.prepare_data_loaders(test_images, test_labels, self.val_transform, is_training=False)

            # Evaluate each fold's best model on test set
            test_accuracies = []
            for fold, model_path in enumerate(best_model_paths):
                model = MitosisClassifier(self.model_name, self.weights).to(self.device)
                model.load_state_dict(torch.load(model_path))

                test_loss, test_preds, test_targets = self.evaluate(model, test_loader, "Test")
                test_balanced_accuracy = balanced_accuracy_score(test_targets, test_preds)
                test_accuracies.append(test_balanced_accuracy)

                self.logger.info(f"Fold {fold + 1} - Test Balanced Accuracy: {test_balanced_accuracy:.4f}")

            self.logger.info(f"Average Test Balanced Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")

            return fold_accuracies, test_accuracies

        return fold_accuracies