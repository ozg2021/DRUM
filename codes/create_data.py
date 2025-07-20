import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple


def create_X(input_data: np.ndarray, tw: int) -> List[np.ndarray]:
    """
    Creates the input data for time series, split into windows of the specified size (tw).

    :param input_data: The input time series data, typically a 2D array (time steps, feature_dim).
    :param tw: The size of the time window used to split the input data.
    :return: A list where each element is the input data for a specific time window.
    """
    X = []
    L = len(input_data)

    for i in range(L - tw + 1):
        train_seq = input_data[i: i + tw]
        X.append(train_seq)

    return X


def create_Y(label: np.ndarray, tw: int) -> List[np.ndarray]:
    """
    Creates the label data for time series, using the last value in each window as the label.

    :param label: The label data, typically a 1D array (time steps, target_value).
    :param tw: The time window size used to select the label data (last value in each window).
    :return: A list where each element is the label for a specific time window.
    """
    Y = []
    L = len(label)

    for i in range(L - tw + 1):
        train_label = label[i + tw - 1: i + tw]
        Y.append(train_label)

    return Y


class TimeseriesDataset(Dataset):
    """
    Dataset class for loading and batching time series data.
    """
    def __init__(self, x: List[np.ndarray], y: List[np.ndarray]):
        """
        Initializes the dataset by converting input data and labels to PyTorch tensors.

        :param x: Input data (time series features), a list where each element is a time window's data.
        :param y: Label data, a list where each element is the label (last timestep's label).
        """
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the input data and label for a given index.

        :param idx: The index of the data point to retrieve.
        :return: The input data and label at the specified index.
        """
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        """
        Returns the total length of the dataset.

        :return: The number of time windows (data points) in the dataset.
        """
        return self.len
