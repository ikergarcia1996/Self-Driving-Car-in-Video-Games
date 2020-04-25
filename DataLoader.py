from numpy import ndarray
from utils import load_and_shuffle_datasets
from typing import Sized, Iterable, List, Tuple
import glob
import os
import threading
import time
import random
import numpy as np

global data
global file_chunks
global next_file


def chunks(iterable: Sized, n: int = 1) -> Iterable:
    """
    Given a iterable generate chunks of size n
    Input:
     - Sized that will be chunked
     - n: Integer batch size
    Output:
    - Iterable
    """
    l: int = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def dataLoaderThread(
    hide_map_prob: float,
    fp: int,
    load_next_event: threading.Event,
    end_thread_event: threading.Event,
):
    """
    When load_next_event is set the thread will load file_chunks[next_file] and store it in the data global variable
    Input:
     - hide_map_prob: Probability for removing the minimap (put a black square)
       from a training example (0<=hide_map_prob<=1)
     - fp: floating-point precision: Available values: 16, 32, 64
     - load_next_event: Event to load the next chunk of data
     - end_thread_event: Event to end the thread that loads the data
    Output:
    """
    global data
    global file_chunks
    global next_file
    while not end_thread_event.isSet():
        load_next_event.wait(timeout=10)
        if load_next_event.isSet():
            load_next_event.clear()

            data = load_and_shuffle_datasets(
                paths=file_chunks[next_file],
                hide_map_prob=hide_map_prob,
                fp=fp,
                force_cpu=True,
            )

            next_file += 1


class DataLoaderTEDD:
    """
    Class for loading the dataset during training. The class will spawn a thread that will buffer the next
    chunk of data so the next chunk will be loaded using the CPU while the GPU is being used for training
    the model. Once all the data has been used the DataLoader will return "None".
    """

    def __init__(
        self, dataset_dir: str, nfiles2load: int, hide_map_prob: float, fp: int
    ):
        """
        Dataloader initialization
        Input:
         - dataset_dir: Directory where the training files (training_file*.npz are stored)
         - nfiles2load: Number of files to load each chunk. The examples in the files will be shuffled
         - hide_map_prob: Probability for removing the minimap (put a black square)
           from a training example (0<=hide_map_prob<=1)
         - fp: floating-point precision: Available values: 16, 32, 64
         Output:
        """
        global data
        global file_chunks
        global next_file

        files: List[str] = glob.glob(os.path.join(dataset_dir, "*.npz"))
        random.seed()
        random.shuffle(files)
        self.total_files: int = len(files)
        file_chunks = list(chunks(files, n=nfiles2load))
        if len(file_chunks) > 0:
            self.load_next_event = threading.Event()
            self.end_thread_event = threading.Event()

            th_data: threading.Thread = threading.Thread(
                target=dataLoaderThread,
                args=[hide_map_prob, fp, self.load_next_event, self.end_thread_event],
            )
            th_data.setDaemon(True)
            th_data.start()
            self.load_next_event.set()

            next_file = 0
            self.actual_file = 0
        else:
            raise ValueError(f"No .npz file found in {dataset_dir}")

    def get_next(self) -> (np.ndarray, np.ndarray):
        """
        Return the next chunk of data and sets the event to buffering the next chunk. Note: The last chunk
        will load num_files%nfiles2load files (the remaining ones).
        Input:
        Output:
         If there are files remaining
          - X: input examples [nfiles2load * num_examples_per_file, 5, 3, H, W]
          - y: golds for the input examples [nfiles2load * num_examples_per_file]
         else (if all files have been already loaded):
          - None
        """
        global next_file
        global data

        if self.actual_file >= len(file_chunks):
            print(f"All files in the DataLoader used")
            self.end_thread_event.set()
            return None

        while self.actual_file == next_file:
            time.sleep(0.1)

        rdata: Tuple[ndarray, ndarray] = data
        self.actual_file += 1
        if not next_file >= len(file_chunks):
            self.load_next_event.set()
        else:
            self.end_thread_event.set()
        return rdata

    def __len__(self) -> int:
        """
        Return the number of files in the dataset
        Input
        Output:
        -int: Number of files in the dataset
        """
        return self.total_files

    def close(self):
        """
        Set the event to stop the thread that buffers the data
        Input
        Output:
        -int: 0
        """
        self.end_thread_event.set()
        return 0
