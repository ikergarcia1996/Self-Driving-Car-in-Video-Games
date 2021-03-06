from model import load_model, TEDD1104
from keyboard.inputsHandler import select_key
from keyboard.getkeys import key_check, key_press
import argparse
import threading
import screen.record_screen as screen_recorder
import torch
import logging
import time
from tkinter import *
import numpy as np
import cv2
from segmentation.segmentation_coco import ImageSegmentation
from torch.cuda.amp import autocast
from torchvision import transforms
from utils import mse

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    logging.warning(
        "GPU not found, using CPU, inference will be very slow. CPU NOT COMPATIBLE WITH FP16"
    )


def run_ted1104(
    model_dir,
    enable_evasion: bool,
    show_current_control: bool,
    num_parallel_sequences: int = 1,
    evasion_score=1000,
    enable_segmentation: bool = False,
) -> None:
    """
    Generate dataset exampled from a human playing a videogame
    HOWTO:
        Set your game in windowed mode
        Set your game to 1600x900 resolution
        Move the game window to the top left corner, there should be a blue line of 1 pixel in the left bezel of your
         screen and the window top bar should start in the top bezel of your screen.
        Let the AI play the game!
    Controls:
        Push QE to exit
        Push L to see the input images
        Push and hold J to use to use manual control

    Input:
    - model_dir: Directory where the model to use is stored (model.bin and model_hyperparameters.json files)
    - enable_evasion: automatic evasion maneuvers when the car gets stuck somewhere. Note: It adds computation time
    - show_current_control: Show a window with text that indicates if the car is currently being driven by
      the AI or a human
    - num_parallel_sequences: num_parallel_sequences to record, is the number is larger the recorded sequence of images
      will be updated faster and the model  will use more recent images as well as being able to do more iterations
      per second. However if num_parallel_sequences is too high it wont be able to update the sequences with 1/10 secs
      between images (default capturate to generate training examples).
    -evasion_score: Mean squared error value between images to activate the evasion maneuvers
    -enable_segmentation: Image segmentation will be performed using a pretrained model. Cars, persons, bikes.. will be
     highlighted to help the model to identify them.

    Output:

    """

    show_what_ai_sees: bool = False
    fp16: bool
    model: TEDD1104
    model, fp16 = load_model(save_dir=model_dir, device=device)

    if enable_segmentation:
        image_segmentation = ImageSegmentation(
            model_name="fcn_resnet101", device=device, fp16=fp16
        )
    else:
        image_segmentation = None

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model.eval()
    stop_recording: threading.Event = threading.Event()

    th_img: threading.Thread = threading.Thread(
        target=screen_recorder.img_thread, args=[stop_recording]
    )
    th_seq: threading.Thread = threading.Thread(
        target=screen_recorder.multi_image_sequencer_thread,
        args=[stop_recording, num_parallel_sequences],
    )
    th_img.setDaemon(True)
    th_seq.setDaemon(True)
    th_img.start()
    # Wait to launch the image_sequencer_thread, it needs the img_thread to be running
    time.sleep(5)
    th_seq.start()

    if show_current_control:
        root = Tk()
        var = StringVar()
        var.set("T.E.D.D. 1104 Driving")
        l = Label(root, textvariable=var, fg="green", font=("Courier", 44))
        l.pack()

    last_time: float = time.time()
    model_prediction: np.ndarray = np.asarray([0])
    score: np.float = np.float(0)
    last_num: int = 0
    while True:
        while (
            last_num == screen_recorder.num
        ):  # Don't run the same sequence again, the resulted key will be the same
            time.sleep(0.0001)
        last_num = screen_recorder.num

        init_copy_time: float = time.time()
        if enable_segmentation:
            img_seq: np.ndarray = image_segmentation.add_segmentation(
                np.copy(screen_recorder.seq)
            )
        else:
            img_seq: np.ndarray = np.copy(screen_recorder.seq)

        keys = key_check()
        if "J" not in keys:

            X = torch.stack(
                (
                    transform(img_seq[0] / 255.0).half(),
                    transform(img_seq[1] / 255.0).half(),
                    transform(img_seq[2] / 255.0).half(),
                    transform(img_seq[3] / 255.0).half(),
                    transform(img_seq[4] / 255.0).half(),
                ),
                dim=0,
            ).to(device)

            if fp16:
                with autocast():
                    model_prediction: torch.tensor = model.predict(X).cpu().numpy()
            else:
                model_prediction: torch.tensor = model.predict(X).cpu().numpy()

            select_key(int(model_prediction[0]))
            key_push_time: float = time.time()

            if show_current_control:
                var.set("T.E.D.D. 1104 Driving")
                l.config(fg="green")
                root.update()

            if enable_evasion:
                score = mse(img_seq[0], img_seq[4])
                if score < evasion_score:
                    if show_current_control:
                        var.set("Evasion maneuver")
                        l.config(fg="blue")
                        root.update()
                    select_key(4)
                    time.sleep(1)
                    if np.random.rand() > 0.5:
                        select_key(6)
                    else:
                        select_key(8)
                    time.sleep(0.2)
                    if show_current_control:
                        var.set("T.E.D.D. 1104 Driving")
                        l.config(fg="green")
                        root.update()

        else:
            if show_current_control:
                var.set("Manual Control")
                l.config(fg="red")
                root.update()

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

        if "Q" in keys and "E" in keys:
            print("\nStopping...")
            stop_recording.set()
            th_seq.join()
            th_img.join()
            if show_what_ai_sees:
                cv2.destroyAllWindows()

            break

        if "L" in keys:
            time.sleep(0.1)  # Wait for key release
            if show_what_ai_sees:
                cv2.destroyAllWindows()
                show_what_ai_sees = False
            else:
                show_what_ai_sees = True

        time_it: float = time.time() - last_time
        print(
            f"Recording at {screen_recorder.fps} FPS\n"
            f"Actions per second {None if time_it==0 else 1/time_it}\n"
            f"Reaction time: {round(key_push_time-init_copy_time,3) if key_push_time>0 else 0} secs\n"
            f"Key predicted by nn: {key_press(int(model_prediction[0]))}\n"
            f"Difference from img 1 to img 5 {None if not enable_evasion else score}\n"
            f"Push QE to exit\n"
            f"Push L to see the input images\n"
            f"Push J to use to use manual control\n",
            end="\r",
        )

        last_time = time.time()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory where the model to use is stored",
    )

    parser.add_argument(
        "--enable_evasion",
        action="store_true",
        help="Enable automatic evasion maneuvers when the car gets stuck somewhere. Note: It adds computation time",
    )

    parser.add_argument(
        "--show_current_control",
        action="store_true",
        help="Show a window with text that indicates if the car is currently being driven by the AI or a human",
    )

    parser.add_argument(
        "--num_parallel_sequences",
        type=int,
        default=1,
        help="num_parallel_sequences to record, is the number is larger the recorded sequence of images will be "
        "updated faster and the model  will use more recent images as well as being able to do more iterations "
        "per second. However if num_parallel_sequences is too high it wont be able to update the sequences with "
        "1/10 secs between images (default capturate to generate training examples). ",
    )

    parser.add_argument(
        "--evasion_score",
        type=float,
        default=200,
        help="Mean squared error value between images to activate the evasion maneuvers",
    )

    parser.add_argument(
        "--enable_segmentation",
        action="store_true",
        help="Image segmentation will be performed using a pretrained model. "
        "Cars, persons, bikes.. will be highlighted to help the model to identify them. "
        "Note: Segmentation will very significantly increase compuation time",
    )

    args = parser.parse_args()

    screen_recorder.initialize_global_variables()

    run_ted1104(
        model_dir=args.model_dir,
        enable_evasion=args.enable_evasion,
        show_current_control=args.show_current_control,
        num_parallel_sequences=args.num_parallel_sequences,
        evasion_score=args.evasion_score,
        enable_segmentation=args.enable_segmentation,
    )
