import os
import time

import numpy as np
import argparse
import threading
import screen.record_screen as screen_recorder
from keyboard.getkeys import key_check
import cv2
from PIL import Image


def save_data(dir_path: str, images: np.ndarray, y: int, number: int):
    """
    Save a trainign example
    Input:
     - dir_path path of the directory where the files are going to be stored
     - data numpy ndarray
     - number integer used to name the file
    Output:

    """

    Image.fromarray(
        cv2.cvtColor(np.concatenate(images, axis=1), cv2.COLOR_BGR2RGB)
    ).save(os.path.join(dir_path, f"{number}_{y}.jpeg"))


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


def counter_keys(key: np.ndarray) -> int:
    """
    Multi-hot vector to one hot vector (represented as an integer)
    Input:
     - key numpy array of integers (1,0) of size 4
    Output:
    - One hot vector encoding represented as an index (int). If the vector does not represent any valid key
    input the returned value will be -1

    """
    if np.array_equal(key, [0, 0, 0, 0]):
        return 0
    elif np.array_equal(key, [1, 0, 0, 0]):
        return 1
    elif np.array_equal(key, [0, 1, 0, 0]):
        return 2
    elif np.array_equal(key, [0, 0, 1, 0]):
        return 3
    elif np.array_equal(key, [0, 0, 0, 1]):
        return 4
    elif np.array_equal(key, [1, 0, 1, 0]):
        return 5
    elif np.array_equal(key, [1, 0, 0, 1]):
        return 6
    elif np.array_equal(key, [0, 1, 1, 0]):
        return 7
    elif np.array_equal(key, [0, 1, 0, 1]):
        return 8
    else:
        return -1


def generate_dataset(output_dir: str, use_probability: bool = True) -> None:
    """
    Generate dataset exampled from a human playing a videogame
    HOWTO:
        Set your game in windowed mode
        Set your game to 1600x900 resolution
        Move the game window to the top left corner, there should be a blue line of 1 pixel in the left bezel of your
         screen and the window top bar should start in the top bezel of your screen.
        Play the game! The program will capture your screen and generate the training examples. There will be saved
         as files named "training_dataX.npz" (numpy compressed array). Don't worry if you re-launch this script,
          the program will search for already existing dataset files in the directory and it won't overwrite them.

    Input:
    - output_dir: Directory where the training files will be saved
    - num_training_examples_per_file: Number of training examples per output file
    - use_probability: Use probability to generate a balanced dataset. Each example will have a probability that
      depends on the number of instances with the same key combination in the dataset.

    Output:

    """
    if not os.path.exists(output_dir):
        print(f"{output_dir} does not exits. We will create it.")
        os.makedirs(output_dir)

    training_data: list = []
    stop_recording: threading.Event = threading.Event()

    th_img: threading.Thread = threading.Thread(
        target=screen_recorder.img_thread, args=[stop_recording]
    )

    th_seq: threading.Thread = threading.Thread(
        target=screen_recorder.image_sequencer_thread, args=[stop_recording]
    )
    th_img.setDaemon(True)
    th_seq.setDaemon(True)
    th_img.start()
    # Wait to launch the image_sequencer_thread, it needs the img_thread to be running
    time.sleep(2)
    th_seq.start()
    number_of_files: int = get_last_file_num(output_dir) + 1
    time.sleep(6)
    last_num: int = 5  # The image sequence starts with images containing zeros, wait until it is filled

    number_of_keys = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0])

    while True:

        while last_num == screen_recorder.num:
            time.sleep(0.01)

        last_num = screen_recorder.num
        img_seq, output = screen_recorder.seq.copy(), screen_recorder.key_out.copy()

        print(
            f"Recording at {screen_recorder.fps} FPS\n"
            f"Images in sequence {len(img_seq)}\n"
            f"Training data len {number_of_files} sequences\n"
            f"Number of archives {number_of_files}\n"
            f"Keys pressed: {output}\n"
            f"Keys samples recorded: "
            f"None: {str(number_of_keys[0])} "
            f"A: {str(number_of_keys[1])} "
            f"D {str(number_of_keys[2])} "
            f"W {str(number_of_keys[3])} "
            f"S {str(number_of_keys[4])} "
            f"AW {str(number_of_keys[5])} "
            f"AS {str(number_of_keys[6])} "
            f"WD {str(number_of_keys[7])} "
            f"SD {str(number_of_keys[8])}\n"
            f"Push QE to exit\n",
            end="\r",
        )

        key = counter_keys(output)

        if key != -1:
            if use_probability:
                total = np.sum(number_of_keys)
                key_num = number_of_keys[key]
                if total != 0:
                    prop = ((total - key_num) / total) ** 2
                    if prop < 0.5:
                        prop = 0.1

                else:
                    prop = 1.0
                if np.random.rand() <= prop:
                    number_of_keys[key] += 1

                    save_data(
                        dir_path=output_dir,
                        images=img_seq,
                        y=key,
                        number=number_of_files,
                    )
                    number_of_files += 1

            else:
                number_of_keys[key] += 1
                save_data(
                    dir_path=output_dir, images=img_seq, y=key, number=number_of_files
                )
                number_of_files += 1

        keys = key_check()
        if "Q" in keys and "E" in keys:
            print("\nStopping...")
            stop_recording.set()
            th_seq.join()
            th_img.join()
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.getcwd(),
        help="Directory where the training data will be saved",
    )

    parser.add_argument(
        "--save_everything",
        action="store_true",
        help="If this flag is added we will save every recorded sequence,"
        " it will result in a very unbalanced dataset. If this flag "
        "is not added we will use probability to try to generate a balanced "
        "dataset",
    )

    args = parser.parse_args()

    screen_recorder.initialize_global_variables()

    generate_dataset(
        output_dir=args.save_dir, use_probability=not args.save_everything,
    )
