from typing import Set

from model import TEDD1104
import datetime
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm

try:
    import cupy as np

    cupy = True
except ModuleNotFoundError:
    import numpy as np


def check_valid_y(data: np.ndarray) -> bool:
    """
    Check if any key has been pressed in the datased. Some files may not have any key recorded due to windows
    permission errors on some computers, people not using WASD or other problems, we want to discard these files.
    Input:
     - data: ndarray [num_examples x 6]
    Output:
    - Bool: True if the file is valid, False is there no key recorded
    """
    seen_keys: Set[int] = set()
    for i in range(0, data.shape[0]):
        if np.array_equal(data[i][5], [0, 0, 0, 0]):
            seen_keys.add(0)
        elif np.array_equal(data[i][5], [1, 0, 0, 0]):
            seen_keys.add(1)
        elif np.array_equal(data[i][5], [0, 1, 0, 0]):
            seen_keys.add(2)
        elif np.array_equal(data[i][5], [0, 0, 1, 0]):
            seen_keys.add(3)
        elif np.array_equal(data[i][5], [0, 0, 0, 1]):
            seen_keys.add(4)
        elif np.array_equal(data[i][5], [1, 0, 1, 0]):
            seen_keys.add(5)
        elif np.array_equal(data[i][5], [1, 0, 0, 1]):
            seen_keys.add(6)
        elif np.array_equal(data[i][5], [0, 1, 1, 0]):
            seen_keys.add(7)
        elif np.array_equal(data[i][5], [0, 1, 0, 1]):
            seen_keys.add(8)

        if len(seen_keys) >= 3:
            return True

    else:
        return False


def evaluate(
    model: TEDD1104, data_loader: DataLoader, device: torch.device, fp16: bool,
) -> float:
    """
    Given a set of input examples and the golds for these examples evaluates the model accuracy
    Input:
     - model: TEDD1104 model to evaluate
     - data_loader: torch.utils.data.DataLoader with the examples to evaluate
     - device: string, use cuda or cpu
     -batch_size: integer batch size
    Output:
    - Accuracy: float
    """
    model.eval()
    correct = 0
    total = 0

    for batch in tqdm(data_loader, desc="Evaluating model"):
        x = torch.flatten(
            torch.stack(
                (
                    batch["image1"],
                    batch["image2"],
                    batch["image3"],
                    batch["image4"],
                    batch["image5"],
                ),
                dim=1,
            ),
            start_dim=0,
            end_dim=1,
        ).to(device)

        y = batch["y"]

        if fp16:
            with autocast():
                predictions: np.ndarray = model.predict(x).cpu()
        else:
            predictions: np.ndarray = model.predict(x).cpu()

        correct += (predictions == y).sum().numpy()
        total += len(predictions)

    return correct / total


def print_message(message: str) -> None:
    """
    Print a message in the <date> : message format
    Input:
     - message: string to print
    Output:
    """
    print(f"<{str(datetime.datetime.now()).split('.')[0]}> {message}")


def mse(image1: np.ndarray, image2: np.ndarray) -> np.float:
    """
    Mean squared error between two numpy ndarrays.
    If available we will use the GPU (cupy) else we will use the CPU (numpy)
    Input:
     - image1: fist numpy ndarray
     - image2: second numpy ndarray
    Ouput:
     - Mean squared error numpy.float
     """
    err = np.float(np.sum((np.asarray(image1) - np.asarray(image2)) ** 2))
    err /= np.float(image1.shape[0] * image1.shape[1])
    return err
