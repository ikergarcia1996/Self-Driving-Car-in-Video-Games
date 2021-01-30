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


def evaluate(
    model: TEDD1104, data_loader: DataLoader, device: torch.device, fp16: bool,
) -> float:
    """
    Given a set of input examples and the golds for these examples evaluates the model mse
    Input:
     - model: TEDD1104 model to evaluate
     - data_loader: torch.utils.data.DataLoader with the examples to evaluate
     - device: string, use cuda or cpu
     -batch_size: integer batch size
    Output:
    - mse loss: float
    """
    model.eval()
    loss: torch.tensor = 0
    criterion: torch.nn.MSELoss = torch.nn.MSELoss(reduction="sum")
    total_examples: int = 0
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

        y = batch["y"].to(device)

        with autocast(enabled=fp16):
            predictions: np.ndarray = model.predict(x)

        loss += criterion(predictions, y)
        total_examples += len(y)

    return (loss / total_examples).cpu().item()


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
