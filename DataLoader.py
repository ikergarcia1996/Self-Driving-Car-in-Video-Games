from numpy import ndarray
from utils import load_and_shuffle_datasets
from typing import Sized, Iterable, List, Tuple
import glob
import os
import threading
import time
import random

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
    def __init__(
        self, dataset_dir: str, nfiles2load: int, hide_map_prob: float, fp: int
    ):
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

    def get_next(self):
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

    def __len__(self):
        return self.total_files

    def close(self):
        self.end_thread_event.set()
        return 0
