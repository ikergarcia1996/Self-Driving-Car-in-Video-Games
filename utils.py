import datetime
from typing import Union
import numpy as np
import os


def print_message(message: str) -> None:
    """
    Print a message in the <date> : message format
    Input:
     - message: string to print
    Output:
    """
    print(f"<{str(datetime.datetime.now()).split('.')[0]}> {message}")


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
    err = np.float(np.sum((np.asarray(image1) - np.asarray(image2)) ** 2))
    err /= np.float(image1.shape[0] * image1.shape[1])
    return err


def length_normalize(
    matrix: np.ndarray,
) -> np.ndarray:

    norms = np.sqrt(np.sum(matrix ** 2, axis=1))
    norms[norms == 0] = 1
    return matrix / norms[:, np.newaxis]


class IOHandler:
    def __init__(self):

        self.keys2controllerMatrix = np.array(
            [
                [0.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [0.0, -1.0, 1.0],
                [0.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
            ]
        )

        # self.keys2controllerMatrix_norm = length_normalize(self.keys2controllerMatrix)

    def keys2controller(self, keys: int) -> np.ndarray:
        return self.keys2controller[keys]

    def controller2keys(self, controller_vector: np.ndarray) -> int:
        """
        return int(
            np.argmax(
                length_normalize(
                    controller_vector[np.newaxis, :].dot(
                        self.keys2controllerMatrix_norm.T
                    )
                )
            )
        )
        """
        return int(
            np.argmin(
                np.sum(
                    (
                        self.keys2controllerMatrix[np.newaxis, :]
                        - controller_vector[np.newaxis, :][:, np.newaxis]
                    )
                    ** 2,
                    -1,
                )
            )
        )

    def input_conversion(
        self, image_name: str, input_type: str, output_type: str
    ) -> Union[int, np.ndarray]:
        if input_type == "controller":
            input_value: np.ndarray = np.asarray(
                [
                    float(x)
                    for x in os.path.basename(image_name)[:-5].split("_")[-1].split(",")
                ],
                dtype=np.float32,
            )

            if output_type == "controller":
                return input_value
            elif output_type == "keyboard":
                return self.controller2keys(controller_vector=input_value)
            else:
                raise ValueError(
                    f"{output_type} output type not supported. Supported outputs: [keyboard,controller]"
                )
        elif input_type == "keyboard":
            input_value: int = int(os.path.basename(image_name)[-6])

            if output_type == "controller":
                return self.keys2controller(input_value)
            elif output_type == "keyboard":
                return input_value
            else:
                raise ValueError(
                    f"{output_type} output type not supported. Supported outputs: [keyboard,controller]"
                )

        else:
            raise ValueError(
                f"{input_type} input type not supported. Supported inputs: [keyboard,controller]"
            )
