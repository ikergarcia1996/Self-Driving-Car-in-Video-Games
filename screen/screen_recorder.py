from screen.grabber import Grabber
from controller.xbox_controller_reader import XboxControllerReader
import numpy as np
import time
import cv2
import threading
import logging
import math
from typing import Union
from keyboard.getkeys import key_check, keys_to_id


def preprocess_image(image):
    """
    Given an image resize it and convert it to a numpy array
    Input:
    - image: PIL image
    Output:
    - numpy ndarray: [480,270,3]
    """
    processed_image = cv2.resize(image, (480, 270))
    return np.asarray(
        processed_image,
        dtype=np.uint8,
    )


class ScreenRecorder:
    """
    Captures screenshots using ImageGRAB from PIL
    """

    fps: int
    width: int
    height: int
    screen_grabber: Grabber
    front_buffer: np.ndarray
    back_buffer: np.ndarray
    get_controller_input: bool
    controller_input: np.ndarray
    controller_reader: XboxControllerReader
    img_thread: threading.Thread

    def __init__(
        self,
        width: int = 1600,
        height: int = 900,
        full_screen: bool = False,
        get_controller_input: bool = False,
        control_mode: str = "keyboard",
        total_wait_secs: int = 5,
    ):
        """
        INIT

        Input:
         - width: Game window width
         - height: Game window height
         - full_screen: If you are playing in full screen (no window border on top) enable this
         - get_controller_input: If true a xbox controller input will be recorded together with the images
         - total_wait_secs: Total secs to wait to prevent false readings

        """
        print(f"We will capture a window of W:{width} x H:{height} size")

        assert control_mode in [
            "keyboard",
            "controller",
        ], f"Control mode: {control_mode} not supported. Available modes: [keyboard,controller]"

        self.control_mode = control_mode
        self.width = width
        self.height = height
        self.get_controller_input = get_controller_input

        if full_screen:
            self.screen_grabber = Grabber(bbox=(0, 0, width, height))
        else:
            self.screen_grabber = Grabber(bbox=(1, 26, width + 1, height + 26))

        self.front_buffer = np.zeros((width, height, 3), dtype=np.int8)
        self.back_buffer = np.zeros((width, height, 3), dtype=np.int8)

        if get_controller_input:
            if control_mode == "keyboard":
                self.controller_input = np.zeros(1, dtype=np.int)
            else:
                self.controller_input = np.zeros(3, dtype=np.float32)

        self.stop_recording: threading.Event = threading.Event()
        self.img_thread: threading.Thread = threading.Thread(
            target=self._img_thread, args=[self.stop_recording]
        )
        self.img_thread.setDaemon(True)
        self.img_thread.start()

        for delay in range(int(total_wait_secs), 0, -1):
            print(
                f"Initializing image recorder, waiting {delay} seconds to prevent wrong readings...",
                end="\r",
            )
            time.sleep(1)

    def _img_thread(self, stop_event: threading.Event):
        """
        Thread that continuously captures the screen
        Input:
         - stop_event: threading.Event that will stop the infinite loop when set
        """
        if self.get_controller_input and self.control_mode == "controller":
            self.controller_reader = XboxControllerReader(total_wait_secs=2)

        while not stop_event.is_set():
            last_time = time.time()
            self.front_buffer = self.screen_grabber.grab(None)

            # Swap buffers
            self.front_buffer, self.back_buffer, self.controller_input = (
                self.back_buffer,
                self.front_buffer,
                None
                if not self.get_controller_input
                else keys_to_id(key_check())
                if self.control_mode == "keyboard"
                else self.controller_reader.read(),
            )

            self.fps = int(1.0 / (time.time() - last_time))

        print("Image capture thread stopped")

    def get_image(self) -> (np.ndarray, Union[np.ndarray, None]):
        """
        Return the last image captured and the xbox controller input when it was captured
        Input:
        Output:
        - np.ndarray  [width x height x 3]: Last image captured
        - np.ndarray [3] if get_controller_input:  xbox controller input when image was captured
          None if not get_controller_input
        """
        return (
            self.back_buffer,
            None if not self.get_controller_input else self.controller_input,
        )

    def stop(self):
        """
        Stops the screen recording and the sequence thread
        Input:
        Output:
        """
        self.stop_recording.set()


class ImageSequencer:

    screen_recorder: ScreenRecorder
    num_sequences: int
    image_sequences: np.ndarray
    controller_sequences: np.ndarray
    get_controller_input: float
    capture_rate: float
    sequence_delay: float
    num_sequence: int
    actual_sequence: int

    def __init__(
        self,
        width: int = 1600,
        height: int = 900,
        full_screen: bool = False,
        get_controller_input: bool = False,
        capturerate: float = 10.0,
        num_sequences: int = 2,
        total_wait_secs: int = 10,
        control_mode: str = "keyboard",
    ):
        """
        INIT

        Input:
         - width: Game window width
         - height: Game window height
         - full_screen: If you are playing in full screen (no window border on top) enable this
         - get_controller_input: If true a xbox controller input will be recorded together with the images
         - capturerate: Number of images to capture per second.
                        The delay between each image in the sequence will be 1/capturerate seconds
         - num_sequences: Number of simultaneous sequences to store. Thread safe if num_sequences > 1
                          More sequences if num_sequences is larger the recorded sequence of images will be
                          updated faster and the model  will use more recent images as well as being able to
                          do more iterations per second but the memory and CPU usage will increase.
                          1-2 recommended for generating the dataset
                          2-6 recommended for inference

         - total_wait_secs: Total secs to wait to prevent false readings
        """

        assert control_mode in [
            "keyboard",
            "controller",
        ], f"Control mode: {control_mode} not supported. Available modes: [keyboard,controller]"

        self.screen_recorder = ScreenRecorder(
            width=width,
            height=height,
            get_controller_input=get_controller_input,
            control_mode=control_mode,
            total_wait_secs=5,
            full_screen=full_screen,
        )  # We will wait after the initialization of this class

        self.num_sequences = num_sequences
        self.image_sequences = np.repeat(
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

        self.get_controller_input = get_controller_input

        if get_controller_input:
            if control_mode == "keyboard":
                self.input_sequences = np.repeat(
                    np.expand_dims(
                        np.asarray(
                            [
                                np.zeros(1),
                            ],
                            dtype=np.int,
                        ),
                        0,
                    ),
                    num_sequences,
                    axis=0,
                )
            else:

                self.input_sequences = np.repeat(
                    np.expand_dims(
                        np.asarray(
                            [
                                np.zeros(3),
                                np.zeros(3),
                                np.zeros(3),
                                np.zeros(3),
                                np.zeros(3),
                            ],
                            dtype=np.float32,
                        ),
                        0,
                    ),
                    num_sequences,
                    axis=0,
                )

        self.capture_rate = capturerate
        self.sequence_delay: float = 1.0 / capturerate / num_sequences

        self.num_sequence = 0
        self.actual_sequence = 0

        self.stop_recording: threading.Event = threading.Event()
        self.sequence_thread: threading.Thread = threading.Thread(
            target=self._sequence_thread, args=[self.stop_recording]
        )
        self.sequence_thread.setDaemon(True)
        self.sequence_thread.start()

        for delay in range(int(total_wait_secs), 0, -1):
            print(
                f"Initializing image sequencer, waiting {delay} seconds to prevent wrong readings...",
                end="\r",
            )
            time.sleep(1)

    def _sequence_thread(self, stop_event: threading.Event):
        """
        Thread that continuously captures sequences of images
        Input:
         - stop_event: threading.Event that will stop the infinite loop when set
        """

        while not stop_event.is_set():
            for i in range(self.num_sequences):
                start_time: float = time.time()

                image, user_input = np.copy(self.screen_recorder.get_image())

                self.image_sequences[i][0] = preprocess_image(image)
                self.image_sequences[i] = self.image_sequences[i][[1, 2, 3, 4, 0]]

                if self.get_controller_input:
                    self.input_sequences[i][0] = user_input
                    self.input_sequences[i] = self.input_sequences[i][[1, 2, 3, 4, 0]]

                self.actual_sequence = i
                self.num_sequence += 1

                wait_time: float = self.sequence_delay - (time.time() - start_time)
                if wait_time > 0:
                    time.sleep(wait_time)
                else:
                    logging.warning(
                        f"{math.fabs(wait_time)} delay in the sequence capture, consider reducing num_sequences"
                    )

        print("Image sequence thread stopped")

    @property
    def sequence_number(self) -> int:
        """
        Get the number of the last sequence captured
        Input:
        Output:
         -Integer: Last sequence captured
        """
        return self.num_sequence

    def stop(self):
        """
        Stops the screen recording and the sequence thread
        Input:
        Output:
        """
        self.stop_recording.set()
        self.screen_recorder.stop()

    def get_sequence(self) -> (np.ndarray, Union[np.ndarray, None]):
        """
        Return the last sequence and the xbox controller input when it was captured
        Input:
        Output:
        - np.ndarray  [5, width x height x 3]: Last image captured
        - np.ndarray [5, 3] if get_controller_input:  xbox controller input when image was captured
          None if not get_controller_input
        """

        return (
            self.image_sequences[self.actual_sequence],
            None
            if not self.get_controller_input
            else self.controller_sequences[self.actual_sequence],
        )
