import os
import time
import numpy as np
import argparse
from screen.screen_recorder import ImageSequencer
import cv2
from PIL import Image
from typing import Union


class BalancedDataset:
    """
    Divides the continuous input into classes and ensures that we don't get to many
    examples of one class to produce a balanced dataset.

    """

    class_matrix: np.ndarray
    lx_ranges: np.ndarray
    lt_ranges: np.ndarray
    rt_ranges: np.ndarray
    total: int

    def __init__(self, lx_classes: int = 6, lt_classes: int = 3, rt_classes: int = 3):

        """
        INIT
        Input:
         -lx_classes: integer, number of subranges into we will divide the lx range of values
         -lr_classes: integer, number of subranges into we will divide the lt range of values
         -rt_classes: integer, number of subranges into we will divide the rt range of values
        Output:
        """
        self.class_matrix = np.zeros(
            (lx_classes, lt_classes, rt_classes), dtype=np.int32
        )

        self.lx_ranges = np.asarray(
            [(i * (2 / lx_classes) - 1) for i in range(lx_classes + 1)],
            dtype=np.float32,
        )

        self.lt_ranges = np.asarray(
            [(i * (2 / lt_classes) - 1) for i in range(lt_classes + 1)],
            dtype=np.float32,
        )

        self.rt_ranges = np.asarray(
            [(i * (2 / rt_classes) - 1) for i in range(rt_classes + 1)],
            dtype=np.float32,
        )

        self.total = 0

    def get_class(self, controller_input: np.ndarray) -> np.ndarray:
        """
        Given a controller inputs get its class
        Input:
         -controller input: np.ndarray [3]
        Output:
         -class of the input: np.ndarray [3]
        """

        lx_class: int = 0
        lt_class: int = 0
        rt_class: int = 0
        for lx in range(len(self.lx_ranges) - 1):
            if self.lx_ranges[lx] <= controller_input[0] <= self.lx_ranges[lx + 1]:
                lx_class = lx
                break

        for lt in range(len(self.lt_ranges) - 1):
            if self.lt_ranges[lt] <= controller_input[1] <= self.lt_ranges[lt + 1]:
                lt_class = lt
                break

        for rt in range(len(self.rt_ranges) - 1):
            if self.rt_ranges[rt] <= controller_input[2] <= self.rt_ranges[rt + 1]:
                rt_class = rt
                break

        return np.asarray([lx_class, lt_class, rt_class], dtype=np.int32)

    def balance_dataset(self, controller_input: np.ndarray) -> bool:
        """
        Given a controller inputs decide if we will add this example to the dataset or no depending of how
        many example of the class of the input are already in the dataset. Fewer examples of the same class
        increases the probability of adding it to the dataset.
        Input:
         -controller input: np.ndarray [3]
        Output:
         -bool: True if we should add the example, False if there are already to many examples of this class
        """
        example_class = self.get_class(controller_input=controller_input)
        class_num_examples = self.class_matrix[
            example_class[0], example_class[1], example_class[2]
        ]
        if self.total != 0:
            prop: float = ((self.total - class_num_examples) / self.total) ** 2
            if prop <= 0.7:
                prop = 0.1

            if np.random.rand() <= prop:
                self.class_matrix[
                    example_class[0], example_class[1], example_class[2]
                ] += 1
                self.total += 1
                return True
            else:
                return False
        else:
            self.class_matrix[example_class[0], example_class[1], example_class[2]] += 1
            self.total += 1
            return True

    @property
    def get_matrix(self) -> np.ndarray:
        """
        Return the matrix containing the number of examples per class in the dataset
        Input:
        Output:
         -matrix: np.ndarray [lx_classes, lt_classes, rt_classes]
        """
        return self.class_matrix


def save_data(dir_path: str, images: np.ndarray, y: np.ndarray, number: int):
    """
    Save a training example
    Input:
     - dir_path path of the directory where the files are going to be stored
     - data numpy ndarray
     - number integer used to name the file
    Output:

    """

    filename = (
        str(number)
        + "%"
        + "_".join([",".join([str(e) for e in elem]) for elem in y])
        + ".jpeg"
    )

    Image.fromarray(
        cv2.cvtColor(np.concatenate(images, axis=1), cv2.COLOR_BGR2RGB)
    ).save(os.path.join(dir_path, filename))


def get_last_file_num(dir_path: str) -> int:
    """
    Given a directory with files in the format [number].jpeg return the higher number
    Input:
     - dir_path path of the directory where the files are stored
    Output:
     - int max number in the directory. -1 if no file exits
     """

    files = [
        int(f.split("%")[0])
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(".jpeg")
    ]

    return -1 if len(files) == 0 else max(files)


def generate_dataset(
    output_dir: str,
    width: int = 1600,
    height: int = 900,
    full_screen: bool = False,
    max_examples_per_second: int = 4,
    use_probability: bool = True,
) -> None:
    """
    Generate dataset exampled from a human playing a videogame
    HOWTO:
        Set your game in windowed mode
        Set your game to width x height resolution
        Move the game window to the top left corner, there should be a blue line of 1 pixel in the left bezel of your
         screen and the window top bar should start in the top bezel of your screen.
        Play the game! The program will capture your screen and generate the training examples. There will be saved
         as files named "training_dataX.npz" (numpy compressed array). Don't worry if you re-launch this script,
          the program will search for already existing dataset files in the directory and it won't overwrite them.

    Input:
    - output_dir: Directory where the training files will be saved
    - width: Game window width
    - height: Game window height
    - full_screen: If you are playing in full screen (no window border on top) enable this
    - examples_per_second: Number of training examples per second to capture
    Output:

    """

    if not os.path.exists(output_dir):
        print(f"{output_dir} does not exits. We will create it.")
        os.makedirs(output_dir)

    img_sequencer = ImageSequencer(
        width=width, height=height, get_controller_input=True, full_screen=full_screen
    )

    data_balancer: Union[BalancedDataset, None]
    if use_probability:
        data_balancer = BalancedDataset()
    else:
        data_balancer = None

    number_of_files: int = get_last_file_num(output_dir) + 1
    last_num: int = 5  # The image sequence starts with images containing zeros, wait until it is filled

    close_app: bool = False

    while not close_app:
        try:
            start_time: float = time.time()
            while last_num == img_sequencer.actual_sequence:
                time.sleep(0.01)

            last_num = img_sequencer.actual_sequence
            img_seq, controller_input = img_sequencer.get_sequence()

            if not use_probability or data_balancer.balance_dataset(
                controller_input=controller_input[-1]
            ):
                save_data(
                    dir_path=output_dir,
                    images=img_seq,
                    y=controller_input,
                    number=number_of_files,
                )

                number_of_files += 1

            wait_time: float = (start_time + 1 / max_examples_per_second) - time.time()
            if wait_time > 0:
                time.sleep(wait_time)

            print(
                f"Recording at {img_sequencer.screen_recorder.fps} FPS\n"
                f"Examples per second: {round(1/(time.time()-start_time),1)} \n"
                f"Images in sequence {len(img_seq)}\n"
                f"Training data len {number_of_files} sequences\n"
                f"LX: {int(controller_input[-1][0]*100)}%\n"
                f"LT: {int(controller_input[-1][1]*100)}%\n"
                f"LX: {int(controller_input[-1][2]*100)}%\n"
                f"Examples per class matrix:\n"
                f"{None if not use_probability else data_balancer.get_matrix.T}\n"
                f"Push Ctrl + C to exit",
                end="\r",
            )

        except KeyboardInterrupt:
            print()
            img_sequencer.stop()
            close_app: bool = True


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.getcwd(),
        help="Directory where the training data will be saved",
    )

    parser.add_argument("--width", type=int, default=1600, help="Game window width")
    parser.add_argument("--height", type=int, default=900, help="Game window height")

    parser.add_argument(
        "--full_screen",
        action="store_true",
        help="full_screen: If you are playing in full screen (no window border on top) set this flag",
    )

    parser.add_argument(
        "--examples_per_second",
        type=int,
        default=8,
        help="Number of training examples per second to capture",
    )

    parser.add_argument(
        "--save_everything",
        action="store_true",
        help="If this flag is added we will save every recorded sequence, "
        "it will result in a very unbalanced dataset. If this flag "
        "is not added we will use probability to try to generate a balanced dataset",
    )

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.save_dir,
        width=args.width,
        height=args.height,
        full_screen=args.full_screen,
        max_examples_per_second=args.examples_per_second,
        use_probability=not args.save_everything,
    )
