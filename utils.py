import numpy as np
import logging
from typing import Iterable, Sized
from model import TEDD1104
import glob
import datetime
import torch

try:
    import cupy as cp

    cupy = True
except ModuleNotFoundError:
    cupy = False
    logging.warning(
        "Cupy not found, dataset preprocessing is going to be slow. "
        "Installing copy is highly recommended (x10 speedup): "
        "https://docs-cupy.chainer.org/en/latest/install.html?highlight=cuda90#install-cupy"
    )


def reshape_y(data: np.ndarray) -> np.ndarray:
    """
    Get gold values from data
    Input:
     - data: ndarray [num_examples x 6]
    Output:
    - ndarray [num_examples]

    """
    reshaped = np.zeros(data.shape[0], dtype=np.int16)
    for i in range(0, data.shape[0]):
        if np.array_equal(data[i][5], [0, 0, 0, 0]):
            reshaped[i] = 0
        elif np.array_equal(data[i][5], [1, 0, 0, 0]):
            reshaped[i] = 1
        elif np.array_equal(data[i][5], [0, 1, 0, 0]):
            reshaped[i] = 2
        elif np.array_equal(data[i][5], [0, 0, 1, 0]):
            reshaped[i] = 3
        elif np.array_equal(data[i][5], [0, 0, 0, 1]):
            reshaped[i] = 4
        elif np.array_equal(data[i][5], [1, 0, 1, 0]):
            reshaped[i] = 5
        elif np.array_equal(data[i][5], [1, 0, 0, 1]):
            reshaped[i] = 6
        elif np.array_equal(data[i][5], [0, 1, 1, 0]):
            reshaped[i] = 7
        elif np.array_equal(data[i][5], [0, 1, 0, 1]):
            reshaped[i] = 8
    return reshaped


def reshape_x_numpy(data: np.ndarray, dtype=np.float16) -> np.ndarray:
    """
    Get images from data as a list and preprocess them.
    Input:
     - data: ndarray [num_examples x 6]
    Output:
    - ndarray [num_examples * 5, num_channels, H, W]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    reshaped = np.zeros((len(data) * 5, 3, 270, 480), dtype=dtype)
    for i in range(0, len(data)):
        for j in range(0, 5):
            reshaped[i * 5 + j] = np.rollaxis((data[i][j] / 255.0) - mean / std, 2, 0)

    return reshaped


def reshape_x_cupy(data: np.ndarray, dtype=cp.float16) -> np.ndarray:
    """
    Get images from data as a list and preprocess them (using GPU).
    Input:
     - data: ndarray [num_examples x 6]
    Output:
    - ndarray [num_examples * 5, num_channels, H, W]

    """
    mean = cp.array([0.485, 0.456, 0.406])
    std = cp.array([0.229, 0.224, 0.225])
    reshaped = cp.zeros((len(data) * 5, 3, 270, 480), dtype=cp.dtype)
    for i in range(0, len(data)):
        for j in range(0, 5):
            reshaped[i * 5 + j] = cp.rollaxis(
                (cp.array(data[i][j]) / 255.0) - mean / std, 2, 0
            )
    return cp.asnumpy(reshaped)


def reshape_x(data: np.ndarray, fp=16) -> np.ndarray:
    """
    Get images from data as a list and preprocess them, if cupy is available it uses the GPU,
    else it uses the CPU (numpy)
    Input:
     - data: ndarray [num_examples x 6]
     - fp: floating-point precision: Available values: 16, 32, 64
    Output:
    - ndarray [num_examples * 5, num_channels, H, W]
    """
    if cupy:
        if fp == 16:
            return reshape_x_cupy(data, dtype=cp.float16)
        elif fp == 32:
            return reshape_x_cupy(data, dtype=cp.float32)
        elif fp == 64:
            return reshape_x_cupy(data, dtype=cp.float64)
        else:
            raise ValueError(
                f"Invalid floating-point precision: {fp}: Available values: 16, 32, 64"
            )
    else:
        if fp == 16:
            return reshape_x_numpy(data, dtype=np.float16)
        elif fp == 32:
            return reshape_x_numpy(data, dtype=np.float32)
        elif fp == 64:
            return reshape_x_numpy(data, dtype=np.float64)
        else:
            raise ValueError(
                f"Invalid floating-point precision: {fp}: Available values: 16, 32, 64"
            )


def batch(iterable: Sized, n: int = 1) -> Iterable:
    """
    Given a iterable generate batches of size n
    Input:
     - Sized that will be batched
     - n: Integer batch size
    Output:
    - Iterable
    """
    l: int = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def nn_batchs(X: Sized, y: Sized, n: int = 1, sequence_size: int = 5) -> Iterable:
    """
    Given the input examples and the golds generate batches of sequence_size
    Input:
     - X: Sized input examples
     - y: Sized golds
     - n: Integer batch size
     -sequence_size: Number of images in a training example. len(x) = len(y) * sequence_size
    Output:
    - Iterable
    """

    assert len(X) == len(y) * sequence_size, (
        f"Inconsistent data, len(X) must equal len(y)*sequence_size."
        f" len(X)={len(X)}, len(y)={len(y)}, sequence_size={sequence_size}"
    )
    bg_X: Iterable = batch(X, n * sequence_size)
    bg_y: Iterable = batch(y, sequence_size)

    for b_X, bg_y in zip(bg_X, bg_y):
        yield b_X, bg_y


def evaluate(
    model: TEDD1104,
    X: torch.tensor,
    golds: torch.tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    """
    Given a set of input examples and the golds for these examples evaluates the model accuracy
    Input:
     - model: TEDD1104 model to evaluate
     - X: input examples [num_examples, sequence_size, 3, H, W]
     - golds: golds for the input examples [num_examples]
     - device: string, use cuda or cpu
     -batch_size: integer batch size
    Output:
    - Accuracy: float
    """
    model.eval()
    correct = 0
    for X_batch, y_batch in nn_batchs(X, golds, batch_size):
        predictions: np.ndarray = model.predict(X_batch.to(device)).cpu().numpy()
        correct += sum(predictions == y_batch)

    return correct / len(golds)


def load_file(path: str, fp: int = 16) -> (np.ndarray, np.ndarray):
    """
    Load dataset from file: Load, reshape and preprocess data.
    Input:
     - path: Path of the dataset
     - fp: floating-point precision: Available values: 16, 32, 64
    Output:
    - X: input examples [num_examples, 5, 3, H, W]
    - y: golds for the input examples [num_examples]
    """
    data = np.load(path, allow_pickle=True)["arr_0"]
    X = reshape_x(data, fp)
    y = reshape_y(data)
    return X, y


def load_dataset(path: str, fp: int = 16) -> (np.ndarray, np.ndarray):
    """
    Load dataset from directory: Load, reshape and preprocess data for all the files in a directory
    Input:
     - path: Path of the directory
     - fp: floating-point precision: Available values: 16, 32, 64
    Output:
    - X: input examples [num_examples_per_file * num_files, 5, 3, H, W]
    - y: golds for the input examples [num_examples_per_file * num_files]
    """
    X: np.ndarray = np.array([])
    y: np.ndarray = np.array([])

    for file in glob.glob(path + "*.npz"):
        X_batch, y_batch = load_file(file, fp)
        if len(X) == 0:
            X = X_batch
            y = y_batch
        else:
            X = np.concatenate((X, X_batch), axis=0)
            y = np.concatenate((y, y_batch), axis=0)

    return X, y


def printTrace(message: str) -> None:
    """
    Print a message in the <date> : message format
    Input:
     - message: string to print
    Output:
    """
    print("<" + str(datetime.datetime.now()) + ">  " + str(message))
