import numpy as np
import logging
from typing import Iterable, Sized, List, Set

from model import TEDD1104
import glob
import datetime
import torch
import os
import random


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


def reshape_y(data: np.ndarray) -> np.ndarray:
    """
    Get gold values from data. multi-hot vector to one-hot vector
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


def reshape_x_numpy(
    data: np.ndarray, dtype=np.float16, hide_map_prob: float = 0.0
) -> np.ndarray:
    """
    Get images from data as a list and preprocess them.
    Input:
     - data: ndarray [num_examples x 6]
     -dtype: numpy dtype for the output array
     -hide_map_prob: Probability for removing the minimap (black square)
      from the sequence of images (0<=hide_map_prob<=1)
    Output:
    - ndarray [num_examples * 5, num_channels, H, W]
    """
    mean = np.array([0.485, 0.456, 0.406], dtype)
    std = np.array([0.229, 0.224, 0.225], dtype)
    reshaped = np.zeros((len(data) * 5, 3, 270, 480), dtype=dtype)
    for i in range(0, len(data)):
        black_minimap: bool = (random.random() <= hide_map_prob)
        for j in range(0, 5):
            img = np.array(data[i][j], dtype=dtype)
            if black_minimap:  # Put a black square over the minimap
                img[215:, :80] = np.zeros((55, 80, 3), dtype=dtype)

            reshaped[i * 5 + j] = np.rollaxis((img / dtype(255.0)) - mean / std, 2, 0)

    return reshaped


def reshape_x_cupy(
    data: np.ndarray, dtype=cp.float16, hide_map_prob: float = 0.0
) -> np.ndarray:
    """
    Get images from data as a list and preprocess them (using GPU).
    Input:
     - data: ndarray [num_examples x 6]
     -dtype: numpy dtype for the output array
     -hide_map_prob: Probability for removing the minimap (black square)
      from the sequence of images (0<=hide_map_prob<=1)
    Output:
    - ndarray [num_examples * 5, num_channels, H, W]

    """

    mean = cp.array([0.485, 0.456, 0.406], dtype=dtype)
    std = cp.array([0.229, 0.224, 0.225], dtype=dtype)
    reshaped = np.zeros((len(data) * 5, 3, 270, 480), dtype=dtype)
    for i in range(0, len(data)):
        black_minimap: bool = (random.random() <= hide_map_prob)
        for j in range(0, 5):
            img = cp.array(data[i][j], dtype=dtype)
            if black_minimap:  # Put a black square over the minimap
                img[215:, :80] = cp.zeros((55, 80, 3), dtype=dtype)

            reshaped[i * 5 + j] = cp.asnumpy(
                cp.rollaxis((img / dtype(255.0)) - mean / std, 2, 0,)
            )

    return reshaped


def reshape_x(data: np.ndarray, fp=16, hide_map_prob: float = 0.0) -> np.ndarray:
    """
    Get images from data as a list and preprocess them, if cupy is available it uses the GPU,
    else it uses the CPU (numpy)
    Input:
     - data: ndarray [num_examples x 6]
     - fp: floating-point precision: Available values: 16, 32, 64
     -hide_map_prob: Probability for removing the minimap (black square) from the image (0<=hide_map_prob<=1)
    Output:
    - ndarray [num_examples * 5, num_channels, H, W]
    """
    assert (
        0 <= hide_map_prob <= 1
    ), f"Hide map prob must be between 0.0 and 1.0. Hide map prob: {hide_map_prob}"
    if cupy:
        if fp == 16:
            return reshape_x_cupy(data, dtype=cp.float16, hide_map_prob=hide_map_prob)
        elif fp == 32:
            return reshape_x_cupy(data, dtype=cp.float32, hide_map_prob=hide_map_prob)
        elif fp == 64:
            return reshape_x_cupy(data, dtype=cp.float64, hide_map_prob=hide_map_prob)
        else:
            raise ValueError(
                f"Invalid floating-point precision: {fp}: Available values: 16, 32, 64"
            )
    else:
        if fp == 16:
            return reshape_x_numpy(data, dtype=np.float16, hide_map_prob=hide_map_prob)
        elif fp == 32:
            return reshape_x_numpy(data, dtype=np.float32, hide_map_prob=hide_map_prob)
        elif fp == 64:
            return reshape_x_numpy(data, dtype=np.float64, hide_map_prob=hide_map_prob)
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
    bg_y: Iterable = batch(y, n)

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
        correct += np.sum(predictions == y_batch)

    return correct / len(golds)


def load_file(
    path: str, fp: int = 16, hide_map_prob: float = 0.0
) -> (np.ndarray, np.ndarray):
    """
    Load dataset from file: Load, reshape and preprocess data.
    Input:
     - path: Path of the dataset
     - fp: floating-point precision: Available values: 16, 32, 64
     -hide_map_prob: Probability for removing the minimap (black square) from the image (0<=hide_map_prob<=1)
    Output:
    - X: input examples [num_examples, 5, 3, H, W]
    - y: golds for the input examples [num_examples]
    """
    try:
        data = np.load(path, allow_pickle=True)["arr_0"]
    except (IOError, ValueError) as err:
        logging.warning(f"[{err}] Error in file: {path}, ignoring the file.")
        return np.array([]), np.array([])
    except:
        logging.warning(
            f"[Unknown exception, probably corrupted file] Error in file: {path}, ignoring the file."
        )
        return np.array([]), np.array([])

    if check_valid_y(data):
        X = reshape_x(data, fp, hide_map_prob)
        y = reshape_y(data)
        return X, y

    else:
        logging.warning(f"Invalid file, no keys recorded: {path}, ignoring the file.")
        return np.array([]), np.array([])


def load_dataset(path: str, fp: int = 16) -> (np.ndarray, np.ndarray):
    """
    Load dataset from directory: Load, reshape and preprocess data for all the files in a directory.
    Input:
     - path: Path of the directory
     - fp: floating-point precision: Available values: 16, 32, 64
    Output:
    - X: input examples [num_examples_per_file * num_files, 5, 3, H, W]
    - y: golds for the input examples [num_examples_per_file * num_files]
    """
    X: np.ndarray = np.array([])
    y: np.ndarray = np.array([])

    files = glob.glob(os.path.join(path, "*.npz"))
    for file_n, file in enumerate(files):
        print(f"Loading file {file_n+1} of {len(files)}...")
        X_batch, y_batch = load_file(file, fp)
        if len(X_batch) > 0 and len(y_batch) > 0:
            if len(X) == 0:
                X = X_batch
                y = y_batch
            else:
                X = np.concatenate((X, X_batch), axis=0)
                y = np.concatenate((y, y_batch), axis=0)

    if len(X) == 0 or len(y) == 0:
        # Since this function is used for loading the dev and test set, we want to stop the execution if we don't
        # have a valid test of dev set.
        raise ValueError(f"Empty dataset, all files invalid. Path: {path}")

    return X, y


def load_and_shuffle_datasets(
    paths: List[str], hide_map_prob: float, fp: int = 16
) -> (np.ndarray, np.ndarray):
    """
    Load multiple dataset files and shuffle the data, useful for training
    Input:
     - paths: List of paths to dataset files
     - fp: floating-point precision: Available values: 16, 32, 64
    Output:
    - X: input examples [num_examples_per_file * num_files, 5, 3, H, W]
    - y: golds for the input examples [num_examples_per_file * num_files]
    """
    data_array: np.ndarray = np.array([])

    for file_no, file in enumerate(paths):
        # print(f"Loading file {file_no+1} of {len(paths)}...")
        try:
            data: np.ndarray = np.load(file, allow_pickle=True)["arr_0"]
        except (IOError, ValueError) as err:
            logging.warning(f"[{err}] Error in file: {file}, ignoring the file.")
            continue
        except:
            logging.warning(
                f"[Unknown exception, probably corrupted file] Error in file: {file}, ignoring the file."
            )
            continue

        if check_valid_y(data):
            if len(data_array) == 0:
                data_array = data
            else:
                data_array = np.concatenate((data_array, data), axis=0)
        else:
            logging.warning(
                f"Invalid file, no keys recorded: {file}, ignoring the file."
            )

    if len(data_array) > 0:
        np.random.seed(None)
        np.random.shuffle(data_array)
    else:
        # Since this function is used for training, we want to continue training with the next files,
        # so we return two empty arrays
        logging.warning(f"Empty dataset, all files invalid. Path: {paths}")
        return np.array([]), np.array([])

    X: np.ndarray = reshape_x(data_array, fp, hide_map_prob)
    y: np.ndarray = reshape_y(data_array)

    return X, y


def printTrace(message: str) -> None:
    """
    Print a message in the <date> : message format
    Input:
     - message: string to print
    Output:
    """
    print("<" + str(datetime.datetime.now()) + ">  " + str(message))


def mse_numpy(image1: np.ndarray, image2: np.ndarray) -> np.float:
    """
    Mean squared error between two numpy ndarrays
    Input:
     - image1: fist array
     - image2: second numpy ndarray
    Ouput:
     - Mean squared error numpy.float
    """
    err = np.float(np.sum((image1 - image2) ** 2))
    err /= np.float(image1.shape[0] * image1.shape[1])
    return err


def mse_cupy(image1: cp.ndarray, image2: cp.ndarray) -> np.float:
    """
    Mean squared error between two cupy ndarrays
    Input:
     - image1: fist array
     - image2: second numpy ndarray
    Ouput:
     - Mean squared error numpy.float
     """
    err = np.float(cp.sum((image1 - image2) ** 2))
    err /= np.float(image1.shape[0] * image1.shape[1])
    return err


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
    if cupy:
        return mse_cupy(cp.asarray(image1), cp.asarray(image2))
    else:
        return mse_numpy(image1, image2)
