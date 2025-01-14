import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        """Initializes the AverageMeter with reset values."""
        self.reset()

    def reset(self) -> None:
        """Resets all statistics."""
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        """
        Updates the meter with a new value.

        Args:
            val (float): The new value to add.
            n (int, optional): The number of occurrences of the value. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        """Returns a string representation of the average value."""
        return f'Current: {self.val:.4f}, Average: {self.avg:.4f}'
    



class CICIDS2017Data:
    """Data loader for the CICIDS 2017 dataset."""

    def __init__(self, batch_size: int):
        """
        Initializes the dataset loader.

        Args:
            batch_size (int): The size of batches for the data loaders.
        """
        self.train_features, self.train_labels, self.test_features, self.test_labels = self._fetch_data()
        self.batch_size = batch_size
        self.train_dataset, self.test_dataset = self._split_data_to_tensor()
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
        self.class_weights = self._compute_class_weights(self.train_labels)

    def _fetch_data(self) -> tuple:
        """Fetches the training and testing data from files."""
        try:
            train_features_path = './data/train_features.pkl'
            train_labels_path = './data/train_labels.pkl'
            test_features_path = './data/test_features.pkl'
            test_labels_path = './data/test_labels.pkl'

            with open(train_features_path, 'rb') as f:
                train_features = pickle.load(f)
            with open(train_labels_path, 'rb') as f:
                train_labels = pickle.load(f)
            with open(test_features_path, 'rb') as f:
                test_features = pickle.load(f)
            with open(test_labels_path, 'rb') as f:
                test_labels = pickle.load(f)

            print(train_features.values.shape)
            return (train_features.values, 
                    train_labels.values.ravel(),  # Flatten the labels
                    test_features.values, 
                    test_labels.values.ravel())
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def _split_data_to_tensor(self) -> tuple:
        """Converts the training and testing data to TensorDataset objects."""
        train_dataset = TensorDataset(
            torch.from_numpy(self.train_features.astype(np.float32)),
            torch.from_numpy(self.train_labels.astype(np.int64))
        )
        test_dataset = TensorDataset(
            torch.from_numpy(self.test_features.astype(np.float32)),
            torch.from_numpy(self.test_labels.astype(np.int64))
        )
        return train_dataset, test_dataset
    
    def _compute_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """Computes class weights for imbalanced datasets.

        Args:
            labels (np.ndarray): The array of class labels.

        Returns:
            torch.Tensor: The class weights as a tensor.
        """
        classes = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=classes, y=labels)
        return torch.tensor(class_weights, dtype=torch.float)

class BalancedEntropyLoss(nn.Module):
    """Calculates the balanced entropy loss with optional class weights."""

    def __init__(self, class_weights: torch.Tensor = None):
        """
        Initializes the BalancedEntropyLoss.

        Args:
            class_weights (torch.Tensor, optional): A tensor of class weights. Defaults to None.
        """
        super(BalancedEntropyLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the balanced entropy loss.

        Args:
            inputs (torch.Tensor): The model outputs (logits).
            targets (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        # Ensure inputs and targets are of correct shape
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("The number of inputs and targets must match.")
        
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            loss = -weights * log_probs[torch.arange(inputs.size(0)), targets]
        else:
            loss = -log_probs[torch.arange(inputs.size(0)), targets]

        return loss.mean()