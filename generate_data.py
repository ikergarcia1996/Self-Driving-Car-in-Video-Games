import os
import time
import numpy as np
import argparse
from screen.screen_recorder import ImageSequencer
import cv2
from PIL import Image
from typing import Union
from utils import IOHandler


class BalancedDataset:
    """
    Generate a dataset of images with balanced classes.
    """

    class_matrix: np.ndarray
    io_handler: IOHandler
    total: int

    def __init__(self):
        """
        INIT
        """
        self.class_matrix = np.zeros(9, dtype=np.int32)

        self.io_handler = IOHandler()

        self.total = 0

    def balance_dataset(self, input_value: Union[np.ndarray, int]) -> bool:
        """
        Decide if a given input value is to be added to the dataset or not.
        The probability of returning True is proportional to the number of examples per class.
        The higher the number of examples of a given class, the lower the probability of returning True
        for examples of that class. Xbox controller inputs are mapped to keys.

        :param int input_value: The controller input value to decide if the example is to be added to the dataset or not.
        :return: True if the example is to be added to the dataset, False otherwise.
        """

        example_class = self.io_handler.input_conversion(
            input_value=input_value, output_type="keyboard"
        )

        if self.total != 0:
            prop: float = (
                (self.total - self.class_matrix[example_class]) / self.total
            ) ** 2
            if prop <= 0.7:
                prop = 0.1

            if np.random.rand() <= prop:
                self.class_matrix[example_class] += 1
                self.total += 1
                return True
            else:
                return False
        else:
            self.class_matrix[example_class] += 1
            self.total += 1
            return True

    @property
    def get_matrix(self) -> np.ndarray:
        """
        Get the class matrix.

        :return: The class matrix.
        """
        return self.class_matrix


def save_data(
    dir_path: str,
    images: np.ndarray,
    y: np.ndarray,
    number: int,
    control_mode: str = "keyboard",
):
    """
    Save a training example (the images and the labels) in the given directory.

    :param str dir_path: The directory where the example will be saved.
    :param np.ndarray images: The images to be saved.
    :param np.ndarray y: The labels to be saved.
    :param int number: The number of the example.
    :param str control_mode: Type of the user input: "keyboard" or "controller"


    """
    assert control_mode in [
        "keyboard",
        "controller",
    ], f"Control mode: {control_mode} not supported. Available modes: [keyboard,controller]"

    filename = (
        ("K" if control_mode == "keyboard" else "C")
        + str(number)
        + "%"
        + "_".join([",".join([str(e) for e in elem]) for elem in y])
        + ".jpeg"
    )

    Image.fromarray(
        cv2.cvtColor(np.concatenate(images, axis=1), cv2.COLOR_BGR2RGB)
    ).save(os.path.join(dir_path, filename))


def get_last_file_num(dir_path: str) -> int:
    """
    Get the number of the last file in the given directory.

    :param str dir_path: The directory where the files are.
    :return: int - The number of the last file.
    """

    files = [
        int(f.split("%")[0][1:])
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
    control_mode: str = "keyboard",
) -> None:
    """
    Generate dataset exampled from a human playing a videogame

    HOWTO:
       - If you play in windowed mode move the game window to the top left corner of the primary screen.
       - If you play in full screen mode, set the full_screen parameter to True.
       - Set your game to width x height resolution specified in the parameters.
       - If you want to record the input from the keyboard set the control_mode parameter to "keyboard".
       - If you want to record the input from an xbox controller set the control_mode parameter to "controller".
       - Play the game! The program will capture your screen and generate the training examples.
       - The program will save the training examples in the output_dir directory.
       - You can call this function again to generate more examples.
       - Detailed instructions can be found in the README.md file.

    :param str output_dir: The directory where the examples will be saved.
    :param int width: The width of the game window.
    :param int height: The height of the game window.
    :param bool full_screen: If the game is played in full screen mode.
    :param int max_examples_per_second: The maximum number of examples per second to capture.
    :param bool use_probability: We will try to balance the number of examples per class recorded.
    :param str control_mode: Type of the user input: "keyboard" or "controller"
    """

    assert control_mode in [
        "keyboard",
        "controller",
    ], f"Control mode: {control_mode} not supported. Available modes: [keyboard,controller]"

    if not os.path.exists(output_dir):
        print(f"{output_dir} does not exits. We will create it.")
        os.makedirs(output_dir)

    img_sequencer = ImageSequencer(
        width=width,
        height=height,
        get_controller_input=True,
        control_mode=control_mode,
        full_screen=full_screen,
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
            while last_num == img_sequencer.num_sequence:
                time.sleep(0.01)

            last_num = img_sequencer.num_sequence
            img_seq, controller_input = img_sequencer.get_sequence()

            if not use_probability or data_balancer.balance_dataset(
                input_value=controller_input[-1]
            ):
                save_data(
                    dir_path=output_dir,
                    images=img_seq,
                    y=controller_input,
                    number=number_of_files,
                    control_mode=control_mode,
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
                f"User input: {controller_input[-1]}\n"
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

    parser = argparse.ArgumentParser(
        description="Generate training data from the game. See the README.md file for more info."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.getcwd(),
        help="The directory where the examples will be saved.",
    )

    parser.add_argument(
        "--width", type=int, default=1600, help="The width of the game window."
    )
    parser.add_argument(
        "--height", type=int, default=900, help="The height of the game window."
    )

    parser.add_argument(
        "--full_screen",
        action="store_true",
        help="If the game is played in full screen mode.",
    )

    parser.add_argument(
        "--examples_per_second",
        type=int,
        default=8,
        help="The maximum number of examples per second to capture.",
    )

    parser.add_argument(
        "--save_everything",
        action="store_true",
        help="Do not try to balance the number of examples per class recorded. "
        "Not recommended you will end up with a huge amount of examples, "
        "specially if you set the examples_per_second to a high value.",
    )

    parser.add_argument(
        "--control_mode",
        type=str,
        default="keyboard",
        choices=["keyboard", "controller"],
        help='Type of the user input: "keyboard" or "controller"',
    )

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.save_dir,
        width=args.width,
        height=args.height,
        full_screen=args.full_screen,
        max_examples_per_second=args.examples_per_second,
        use_probability=not args.save_everything,
        control_mode=args.control_mode,
    )
