from typing import Union, Tuple
import numpy as np
import os
from transformers import PreTrainedModel
import logging


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def mse(image1: np.ndarray, image2: np.ndarray) -> np.float:
    """
    Mean squared error between two images (np.ndarrays).

    :param np.ndarray image1: First image
    :param np.ndarray image2: Second image
    :return: Float - Mean squared error
    """
    return np.sum((np.asarray(image1) - np.asarray(image2)) ** 2).item()


def length_normalize(
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Normalizes the length of a matrix.

    :param np.ndarray matrix: Matrix to normalize
    :return: np.ndarray - Normalized matrix
    """
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    return matrix / norms[:, np.newaxis]


class IOHandler:
    """
    Class for handling input and output formats. It is used to convert between keyboard input and controller input.
    It also handles the saving and loading of the data.
    """

    def __init__(self):
        """
        INIT
        """
        self.keys2controllerMatrix = np.array(
            [
                [0.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
                [1.0, 1.0],
                [1.0, -1.0],
            ]
        )

        # self.keys2controllerMatrix_norm = length_normalize(self.keys2controllerMatrix)

    def keys2controller(self, keys: int) -> np.ndarray:
        """
        Converts a keyboard input to a controller input.

        :param int keys: Keyboard input
        :return: np.ndarray [2] - Controller input
        """
        return self.keys2controllerMatrix[keys]

    def controller2keys(self, controller_vector: np.ndarray) -> int:
        """
        Converts a controller input to a keyboard input.
        :param np.ndarray controller_vector: Controller input [2]
        :return: int - Keyboard input
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

    def imagename_input_conversion(
        self, image_name: str, output_type: str
    ) -> Union[int, np.ndarray]:
        """
        Converts an image name to an 'output_type' input

        :param str image_name: Image name
        :param str output_type: Output type: keyboard or controller
        :return: Union[int, np.ndarray] - Output in the specified format
        """
        metadata = os.path.basename(image_name)[:-5]
        header, values = metadata.split("%")
        control_mode = header[0]
        values = values.split("_")

        if control_mode == "controller":
            input_value: np.ndarray = np.asarray(
                [float(x) for x in values[-1].split(",")],
                dtype=np.float32,
            )

            input_value = np.asarray(
                [input_value[0], (input_value[2] - input_value[1]) / 2]
            )

            if output_type == "controller":
                return input_value
            elif output_type == "keyboard":
                return self.controller2keys(controller_vector=input_value)
            else:
                raise ValueError(
                    f"{output_type} output type not supported. Supported outputs: [keyboard,controller]"
                )
        else:
            input_value: int = int(values[-1])

            if output_type == "controller":
                return self.keys2controller(input_value)
            elif output_type == "keyboard":
                return input_value
            else:
                raise ValueError(
                    f"{output_type} output type not supported. Supported outputs: [keyboard,controller]"
                )

    def input_conversion(
        self, input_value: Union[int, np.ndarray], output_type: str
    ) -> Union[int, np.ndarray]:
        """
        Converts an input to an 'output_type' input

        :param Union[int, np.ndarray] input_value: Input value
        :param str output_type: Output type: keyboard or controller
        :return: Union[int, np.ndarray] - Output in the specified format
        """

        if isinstance(input_value, int) or input_value.size == 1:
            if output_type == "controller":
                return self.keys2controller(int(input_value))
            elif output_type == "keyboard":
                return int(input_value)
            else:
                raise ValueError(
                    f"{output_type} output type not supported. Supported outputs: [keyboard,controller]"
                )
        else:
            if output_type == "controller":
                return input_value
            elif output_type == "keyboard":
                return self.controller2keys(controller_vector=input_value)
            else:
                raise ValueError(
                    f"{output_type} output type not supported. Supported outputs: [keyboard,controller]"
                )


def get_trainable_parameters(model: PreTrainedModel) -> Tuple[int, int, float]:
    """
    Prints the number of trainable parameters in the model.

    Args:
        model (`PreTrainedModel`):
            The model to print the number of trainable parameters for.

    Returns:
        `Tuple[int, int, float]`:
            The number of trainable parameters, the total number of parameters and the
            percentage of trainable parameters.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param, 100 * trainable_params / all_param
