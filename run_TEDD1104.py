from modeling_videomae import VideoMAEForVideoClassification
from keyboard.getkeys import key_check
import argparse
from screen.screen_recorder import ImageSequencer
import torch
import logging
import time
from tkinter import *
import numpy as np
import cv2
from utils import mse
from keyboard.inputsHandler import select_key
from keyboard.getkeys import id_to_key
import math
from typing import Optional
from image_processing_videomae import VideoMAEImageProcessor
from constants import IMAGE_MEAN, IMAGE_STD


from utils import IOHandler, get_trainable_parameters

try:
    from controller.xbox_controller_emulator import XboxControllerEmulator

    _controller_available = True
except ImportError:
    _controller_available = False
    XboxControllerEmulator = None
    print(
        "[WARNING!] Controller emulation unavailable, see controller/setup.md for more info. "
        "You can ignore this warning if you will use the keyboard as controller for TEDD1104."
    )


def load_model(
    model_name_or_path, precision: int = 16
) -> VideoMAEForVideoClassification:
    """
    Load the model and set the precision for inference.

    Args:
        model_name_or_path (str): Path to the model directory or HuggingFace model name.
        precision (int): Precision for inference. Choose from 4, 8, 16 or 32.

    Returns:
        VideoMAEForVideoClassification: The model.
    """
    if precision == 32:
        model_dtype = torch.float32
        bnb_config = None
        quant_args = {}
        logging.info(
            f"Loading model with using float32. This is the slowest option. "
            f"Use precision 16 for faster inference."
        )
    elif precision == 16:
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            model_dtype = torch.float16

            logging.warning(
                f"Your GPU does not support bfloat16, using float16 instead. "
                f"Models were trained with bfloat16, so you might encounter worse accuracy. "
                f"If this is the case, you can use precision 32 for better accuracy, "
                f"although inference will be slower and the model will use more memory."
            )
        else:
            model_dtype = torch.bfloat16
            logging.info("We will load the model using bfloat16.")
        bnb_config = None
        quant_args = {}
    elif precision == 8:
        logging.info("We will load the model using 8 bit quantization.")

        from transformers import BitsAndBytesConfig

        quant_args = {"load_in_8bit": True}
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        model_dtype = (
            torch.float32
        )  # We will load the model in FP32 and quantize it to 8 bit

    elif precision == 4:
        logging.info("We will load the model using 4 bit quantization.")

        from transformers import BitsAndBytesConfig

        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            model_dtype = torch.float16
            logging.warning(
                "Your GPU does not support bfloat16, using float16 instead. "
                "Models were trained with bfloat16, so you might encounter worse accuracy. "
                "If you have enough memory, use 8-bit quantization instead of 4-bit quantization."
            )
        else:
            model_dtype = torch.bfloat16

        quant_args = {"load_in_4bit": True}
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype,
        )

    else:
        raise ValueError(
            f"Precision {precision} not supported. Choose from 4, 8, 16 or 32"
        )

    model = VideoMAEForVideoClassification.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        torch_dtype=model_dtype,
        quantization_config=bnb_config,
        num_labels=9,
        **quant_args,
    )

    if torch.cuda.is_available() and precision not in [
        4,
        8,
    ]:  # Quantified models are already loaded in Cuda
        model = model.cuda()

    model.eval()

    _, all_param, _ = get_trainable_parameters(model=model)
    logging.info(
        f"Model loaded from {model_name_or_path}. Dtype: {model.dtype}. "
        f"Device: {model.device}. Model params: {all_param}"
    )

    return model


def run_ted1104(
    model_name_or_path: str,
    enable_evasion: bool,
    show_current_control: bool,
    num_parallel_sequences: int = 2,
    width: int = 1600,
    height: int = 900,
    full_screen: bool = False,
    evasion_score=1000,
    control_mode: str = "keyboard",
    precision: int = 16,
) -> None:
    """
    Run TEDD1104 model in Real-Time inference

    HOWTO:
       - If you play in windowed mode move the game window to the top left corner of the primary screen.
       - If you play in full screen mode, set the full_screen parameter to True.
       - Set your game to width x height resolution specified in the parameters.
       - If you use the keyboard for controlling the game set the control_mode parameter to "keyboard".
       - If you  use an vXbox Controller for controlling the game set the control_mode parameter to "controller".
       - Run the script and let TEDD1104 Play the game!
       - Detailed instructions can be found in the README.md file.

    Args:
        model_name_or_path (str): Path to the model directory or HuggingFace model name.
        enable_evasion (bool): Enable evasion, if the vehicle gets stuck we will reverse and randomly turn left/right.
        show_current_control (bool): Show if TEDD or the user is driving in the screen .
        num_parallel_sequences (int): Number of sequences to run in parallel.
        width (int): Width of the game window.
        height (int): Height of the game window.
        full_screen (bool): If the game is played in full screen mode.
        evasion_score (int): Threshold to trigger the evasion.
        control_mode (str): Device that TEDD will use from driving "keyboard" or "controller" (xbox controller).
        precision (int): Precision for inference. Choose from 4, 8, 16 or 32.
    """

    assert control_mode in [
        "keyboard",
        "controller",
    ], f"{control_mode} control mode not supported. Supported dataset types: [keyboard, controller].  "

    if control_mode == "controller" and not _controller_available:
        raise ModuleNotFoundError(
            "Controller emulation not available see controller/setup.md for more info."
        )

    show_what_ai_sees: bool = False
    fp16: bool

    model = load_model(
        model_name_or_path=model_name_or_path,
        precision=precision,
    )

    image_processor = VideoMAEImageProcessor(
        do_resize=False,
        do_center_crop=False,
        do_rescale=True,
        do_normalize=True,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
    )

    io_handler = IOHandler()

    if control_mode == "controller":
        xbox_controller: Optional[XboxControllerEmulator] = XboxControllerEmulator()
    else:
        xbox_controller = None

    img_sequencer = ImageSequencer(
        width=width,
        height=height,
        full_screen=full_screen,
        get_controller_input=False,
        num_sequences=num_parallel_sequences,
        total_wait_secs=5,
    )

    if show_current_control:
        root = Tk()
        var = StringVar()
        var.set("T.E.D.D. 1104 Driving")
        text_label = Label(root, textvariable=var, fg="green", font=("Courier", 44))
        text_label.pack()
    else:
        root = None
        var = None
        text_label = None

    last_time: float = time.time()
    score: float = 0.0
    last_num: int = 5  # The image sequence starts with images containing zeros, wait until it is filled

    close_app: bool = False
    model_prediction = np.zeros(3 if control_mode == "controller" else 1)

    lt: float = 0
    rt: float = 0

    while not close_app:
        try:
            # start = time.time()
            while last_num == img_sequencer.num_sequence:
                time.sleep(0.01)

            last_num = img_sequencer.num_sequence
            img_seq, _ = img_sequencer.get_sequence()
            # print(f"Time to get sequence: {time.time() - start}")

            init_copy_time: float = time.time()
            keys = key_check()
            if "J" not in keys:
                with torch.no_grad():
                    # start = time.time()
                    images = list(img_seq)
                    # print(f"Time to list images: {time.time() - start}")
                    start = time.time()
                    try:
                        model_inputs: torch.tensor = image_processor(
                            images=images,
                            input_data_format="channels_last",
                            return_tensors="pt",
                        )

                    except ValueError:
                        print()
                        img_sequencer.stop()
                        if control_mode == "controller":
                            xbox_controller.stop()
                        exit()

                    # print(f"Time to process images: {time.time() - start}")
                    # start = time.time()
                    model_prediction: torch.tensor = model(
                        **model_inputs.to(device=model.device, dtype=model.dtype)
                    ).logits[0]
                    # print(f"Time to inference: {time.time() - start}")
                    # print(model_prediction)
                    # start = time.time()
                    model_prediction = torch.argmax(model_prediction).item()

                    model_prediction = io_handler.input_conversion(
                        input_value=model_prediction,
                        output_type=control_mode,
                    )
                    # print(f"Time to convert input: {time.time() - start}")

                # start = time.time()

                if control_mode == "controller":
                    if model_prediction[1] > 0:
                        rt = min(1.0, float(model_prediction[1])) * 2 - 1
                        lt = -1
                    else:
                        rt = -1
                        lt = min(1.0, math.fabs(float(model_prediction[1]))) * 2 - 1

                    lx = max(-1.0, min(1.0, float(model_prediction[0])))

                    xbox_controller.set_controller_state(
                        lx=lx,
                        lt=lt,
                        rt=rt,
                    )
                else:
                    select_key(model_prediction)

                key_push_time: float = time.time()
                # print(f"Time to push key: {time.time() - start}")
                # start = time.time()
                if show_current_control:
                    var.set("T.E.D.D. 1104 Driving")
                    text_label.config(fg="green")
                    root.update()

                if enable_evasion:
                    score = mse(img_seq[0], img_seq[4])
                    if score < evasion_score:
                        if show_current_control:
                            var.set("Evasion maneuver")
                            text_label.config(fg="blue")
                            root.update()
                        if control_mode == "controller":
                            xbox_controller.set_controller_state(lx=0, lt=1.0, rt=-1.0)
                            time.sleep(1)
                            if np.random.rand() > 0.5:
                                xbox_controller.set_controller_state(
                                    lx=1.0, lt=0.0, rt=-1.0
                                )
                            else:
                                xbox_controller.set_controller_state(
                                    lx=-1.0, lt=0.0, rt=-1.0
                                )
                            time.sleep(0.2)
                        else:
                            select_key(4)
                            time.sleep(1)
                            if np.random.rand() > 0.5:
                                select_key(6)
                            else:
                                select_key(8)
                            time.sleep(0.2)

                        if show_current_control:
                            var.set("T.E.D.D. 1104 Driving")
                            text_label.config(fg="green")
                            root.update()

            else:
                if show_current_control:
                    var.set("Manual Control")
                    text_label.config(fg="red")
                    root.update()

                if control_mode == "controller":
                    xbox_controller.set_controller_state(lx=0.0, lt=-1, rt=-1.0)

                key_push_time: float = 0.0

            if show_what_ai_sees:
                cv2.imshow("window1", img_seq[0])
                cv2.waitKey(1)
                cv2.imshow("window2", img_seq[1])
                cv2.waitKey(1)
                cv2.imshow("window3", img_seq[2])
                cv2.waitKey(1)
                cv2.imshow("window4", img_seq[3])
                cv2.waitKey(1)
                cv2.imshow("window5", img_seq[4])
                cv2.waitKey(1)

            if "L" in keys:
                time.sleep(0.1)  # Wait for key release
                if show_what_ai_sees:
                    cv2.destroyAllWindows()
                    show_what_ai_sees = False
                else:
                    show_what_ai_sees = True

            time_it: float = time.time() - last_time

            if control_mode == "controller":
                info_message = (
                    f"LX: {int(model_prediction[0] * 100)}%"
                    f"\n LT: {int(lt * 100)}%\n"
                    f"RT: {int(rt * 100)}%"
                )
            else:
                info_message = f"Predicted Key: {id_to_key(model_prediction)}"

            print(
                f"Recording at {img_sequencer.screen_recorder.fps} FPS\n"
                f"Actions per second {None if time_it == 0 else 1 / time_it}\n"
                f"Reaction time: {round(key_push_time - init_copy_time, 3) if key_push_time > 0 else 0} secs\n"
                f"{info_message}\n"
                f"Difference from img 1 to img 5 {None if not enable_evasion else score}\n"
                f"Push Ctrl + C to exit\n"
                f"Push L to see the input images\n"
                f"Push J to use to use manual control\n",
                end="\r",
            )

            # print(f"Time to show info: {time.time() - start}")

            last_time = time.time()

        except KeyboardInterrupt:
            print()
            img_sequencer.stop()
            if control_mode == "controller":
                xbox_controller.stop()
            exit()


if __name__ == "__main__":
    # Set logging level
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model directory or HuggingFace model name.",
    )

    parser.add_argument("--width", type=int, default=1600, help="Game window width")
    parser.add_argument("--height", type=int, default=900, help="Game window height")

    parser.add_argument(
        "--enable_evasion",
        action="store_true",
        help="Enable evasion, if the vehicle gets stuck we will reverse and randomly turn left/right. "
        "Usefull if you want 0 human intervention. But it has a high computational cost.",
    )

    parser.add_argument(
        "--show_current_control",
        action="store_true",
        help="Show if TEDD or the user is driving in the screen .",
    )

    parser.add_argument(
        "--num_parallel_sequences",
        type=int,
        default=3,
        help="number of parallel sequences to record, if the number is higher the model will do more "
        "iterations per second (will push keys more often) provided your GPU is fast enough. "
        "This improves the performance of the model but increases the CPU and RAM usage.",
    )

    parser.add_argument(
        "--evasion_score",
        type=float,
        default=200,
        help="Threshold to trigger the evasion.",
    )

    parser.add_argument(
        "--control_mode",
        type=str,
        choices=["keyboard", "controller"],
        default="keyboard",
        help="Device that TEDD will use from driving 'keyboard' or 'controller' (xbox controller).",
    )

    parser.add_argument(
        "--full_screen",
        action="store_true",
        help="If you are playing in full screen (no window border on top) set this flag",
    )

    parser.add_argument(
        "--precision",
        type=int,
        choices=[32, 16, 8, 4],
        default=16,
        help="Precision for inference. "
        "Choose from 4, 8, 16 or 32. 16 is the default and fastest option. For 8 and 4 bit "
        "quantization you need to install the bitsandbytes library ('pip install bitsandbytes'). Quantization will "
        "significantly reduce the model size and inference so it will use much less memory. Use it if your GPU has "
        "less than 8GB of memory. 32 is the slowest option, only recommended if you have compatibility issues with "
        "16 bit precision.",
    )

    args = parser.parse_args()

    run_ted1104(
        model_name_or_path=args.model,
        width=args.width,
        height=args.height,
        full_screen=args.full_screen,
        enable_evasion=args.enable_evasion,
        show_current_control=args.show_current_control,
        num_parallel_sequences=args.num_parallel_sequences,
        evasion_score=args.evasion_score,
        control_mode=args.control_mode,
        precision=args.precision,
    )
