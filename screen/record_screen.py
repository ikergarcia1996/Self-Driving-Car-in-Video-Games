from screen.grabber import Grabber
import numpy as np
import time
import cv2
from keyboard.getkeys import key_check
import threading
import logging
import math

global fps
global grb
global front_buffer
global back_buffer
global num
global seq
global key_out


def keys_to_output(keys: np.ndarray) -> np.ndarray:
    """
    Convert keys to a ...multi-hot... array
    Input:
     - np.ndarray of strings ["A","W"]
    Ouput:
     - multi-hot array of integers (0,1) representing which keys are pressed (1). Array: [A,D,W,S]
    """
    output = np.asarray([0, 0, 0, 0])

    if "A" in keys:
        output[0] = 1
    if "D" in keys:
        output[1] = 1
    if "W" in keys:
        output[2] = 1
    if "S" in keys:
        output[3] = 1

    return output


def screen_record():
    """
    Do a screenshot using ImageGRAB from PIL
    Input:

    Output:
    - Screenshot in the (1, 26, 1601, 926) coordinates of the screen: size [1600,900,3]
    """
    global grb
    general_img = grb.grab(None)
    return general_img


def img_thread(stop_event: threading.Event):
    """
    A thread that will continuously do screenshots and will store the last one in the global back_back_buffer variable
    Input:
    - stop_event: threading.Event that will stop the thread
    Output:

    """
    global front_buffer
    global back_buffer
    global fps

    last_time = time.time()
    while not stop_event.is_set():
        front_buffer = screen_record()
        # Swap buffers
        front_buffer, back_buffer = back_buffer, front_buffer
        fps = int(1.0 / (time.time() - last_time))
        last_time = time.time()


def preprocess_image(image):
    """
    Given an image resize it and convert it to a numpy array
    Input:
    - image: PIL image
    Output:
    - numpy ndarray: [480,270,3]
    """
    processed_image = cv2.resize(image, (480, 270))
    return np.asarray(processed_image, dtype=np.uint8,)


def image_sequencer_thread(stop_event: threading.Event) -> None:
    """
    Get the images from img_thread and maintain an updated array seq of the last 5 captured images with a 1/10 secs
    span between them.
    Input:
    - stop_event: threading.Event that will stop the thread
    Output:

    """
    global back_buffer
    global seq
    global key_out
    global num

    # Frames per second capture rate
    capturerate = 10.0
    while not stop_event.is_set():
        last_time = time.time()
        seq, num, key_out = (
            np.concatenate(
                (seq[1:], [preprocess_image(np.copy(back_buffer))]), axis=0,
            ),
            num + 1,
            keys_to_output(key_check()),
        )
        waittime = (1.0 / capturerate) - (time.time() - last_time)
        if waittime > 0.0:
            time.sleep(waittime)


def multi_image_sequencer_thread(
    stop_event: threading.Event, num_sequences: int
) -> None:
    """
    Get the images from img_thread and maintain an updated array seq of the last 5 captured images with a 1/10 secs
    span between them.
    Input:
    - stop_event: threading.Event that will stop the thread
    Output:

    """
    global back_buffer
    global seq
    global key_out
    global num

    # Frames per second capture rate
    capturerate: float = 10.0
    sequence_delay: float = 1.0 / capturerate / num_sequences
    sequences: np.ndarray = np.repeat(
        np.expand_dims(
            np.asarray(
                [
                    np.zeros((270, 480, 3)),
                    np.zeros((270, 480, 3)),
                    np.zeros((270, 480, 3)),
                    np.zeros((270, 480, 3)),
                    np.zeros((270, 480, 3)),
                ],
                dtype=np.uint8,
            ),
            0,
        ),
        num_sequences,
        axis=0,
    )

    while not stop_event.is_set():
        for i in range(num_sequences):
            start_time: float = time.time()
            sequences[i][0] = preprocess_image(np.copy(back_buffer))
            sequences[i] = sequences[i][[1, 2, 3, 4, 0]]
            seq, num, key_out = sequences[i], num + 1, keys_to_output(key_check())
            waittime: float = sequence_delay - (time.time() - start_time)
            if waittime > 0:
                time.sleep(waittime)
            else:
                logging.warning(
                    f"{math.fabs(waittime)} delay in the sequence capture, consider reducing num_sequences"
                )


def initialize_global_variables() -> None:
    """
    Initialize global variables
    Input:
    Output:
    """
    global fps
    global grb
    global front_buffer
    global back_buffer
    global num
    global seq
    global key_out

    fps = 10
    grb = Grabber(bbox=(1, 26, 1601, 926))
    front_buffer = np.zeros((1600, 900), dtype=np.int8)
    back_buffer = np.zeros((1600, 900), dtype=np.int8)
    num = 0
    seq = np.asarray(
        [
            np.zeros((270, 480, 3)),
            np.zeros((270, 480, 3)),
            np.zeros((270, 480, 3)),
            np.zeros((270, 480, 3)),
            np.zeros((270, 480, 3)),
        ],
        dtype=np.uint8,
    )
    key_out = np.array([0, 0, 0, 0])
