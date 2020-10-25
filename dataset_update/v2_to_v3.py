import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import os
from typing import List, Sized, Iterable
import argparse
import logging
import cv2
import math
import multiprocessing as mp
from functools import partial
import sys


def batch(iterable: Sized, n: int = 1) -> Iterable:
    l = len(iterable)
    s = math.ceil(l / n)
    for ndx in range(0, l, s):
        yield iterable[ndx : min(ndx + s, l)]


def get_last_file_num(dir_path: str) -> int:
    """
    Given a directory with files in the format [number].jpeg return the higher number
    Input:
     - dir_path path of the directory where the files are stored
    Output:
     - int max number in the directory. -1 if no file exits
     """

    files = [
        int(f.split(".")[0])
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(".jpeg")
    ]

    return -1 if len(files) == 0 else max(files)


def process_files(output_dir: str, data):
    global last_number
    process_id: int
    file_names: int
    process_id, file_names = data

    for file_path in tqdm(file_names, desc=f"[Process {process_id}] Converting files"):
        try:
            data = np.load(file_path, allow_pickle=True)["arr_0"]
            h: int = data[0][0].shape[0]
            l: int = data[0][0].shape[1]
            c: int = data[0][0].shape[2]
            concat_image: np.ndarray = np.zeros((h, l * 5, c), dtype=data[0][0].dtype)
            for i in range(len(data)):
                for x in range(5):
                    concat_image[:, l * x : l * x + l, :] = data[i][x]

                with last_number.get_lock():
                    last_number.value += 1

                    filename = os.path.join(
                        output_dir, f"{last_number.value}_{y_format(data[i][-1])}.jpeg"
                    )

                if os.path.exists(filename):
                    logging.warning(f"Overwriting file {filename}")

                Image.fromarray(cv2.cvtColor(concat_image, cv2.COLOR_BGR2RGB)).save(
                    filename
                )

        except (IOError, ValueError) as err:
            logging.warning(f"[{err}] Error in file: {file_path}, ignoring the file.")
            continue
        except Exception as err:
            logging.warning(
                f"[Unknown exception, probably corrupted file] Error in file: {file_path}, ignoring the file."
                f"Exception error message: \n {err}"
            )
            continue


def y_format(y: np.ndarray) -> int:
    """
    multi-hot vector to one-hot vector
    Input:
     - y: multi_hot vectors (4 dims)
    Output:
     - one-hot encoding
     """
    if np.array_equal(y, [0, 0, 0, 0]):
        return 0
    elif np.array_equal(y, [1, 0, 0, 0]):
        return 1
    elif np.array_equal(y, [0, 1, 0, 0]):
        return 2
    elif np.array_equal(y, [0, 0, 1, 0]):
        return 3
    elif np.array_equal(y, [0, 0, 0, 1]):
        return 4
    elif np.array_equal(y, [1, 0, 1, 0]):
        return 5
    elif np.array_equal(y, [1, 0, 0, 1]):
        return 6
    elif np.array_equal(y, [0, 1, 1, 0]):
        return 7
    elif np.array_equal(y, [0, 1, 0, 1]):
        return 8

    raise ValueError(f"Unknown tag {y}")


def init(args):
    """ store the counter for later use """
    global last_number
    last_number = args


def v2_to_v3(input_dir: str, output_dir: str, max_cores: int):
    """
    Given a directory with files in the v2 dataset format converts it to the v3 dataset format
    Input:
     - input_dir: path of the directory where the v2 dataset files are stored
    Output:
      output_dir: path of the directory where the v3 dataset files are going to be stored
      if v3 files already exists in the output_dir directory the script will not overwrite them
     """

    if not os.path.exists(output_dir):
        print(f"{output_dir} does not exits. We will create it.")
        os.makedirs(output_dir)

    v2_files: List[str] = glob.glob(os.path.join(input_dir, "*.npz"))

    print(f">>> v2 to v3 converter: {len(v2_files)} files found in {input_dir} ")

    num_cores: int = min(os.cpu_count(), len(v2_files), max_cores)
    process_ids: List[int] = list(range(num_cores))
    batches: List[List[str]] = list(batch(v2_files, n=num_cores))

    last_number = mp.Value("l", get_last_file_num(output_dir))

    func = partial(process_files, output_dir)

    with mp.Pool(
        processes=num_cores, initializer=init, initargs=(last_number,)
    ) as pool:
        _ = pool.map(func, zip(process_ids, batches))

    print("Done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory where the v2 data is stored",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the v3 data is going to be stored",
    )

    parser.add_argument(
        "--max_cores",
        type=int,
        default=sys.maxsize,
        help="Max cores to use. By default we will use all cores available. If you get out of memory errors decrease "
        "this value. NOTE: If you want  your data to be stored in the same order that it is stored in the v2 files "
        "use --max_cores 1, using multiple cores will shuffle the training examples.",
    )

    args = parser.parse_args()

    v2_to_v3(args.input_dir, args.output_dir, args.max_cores)
