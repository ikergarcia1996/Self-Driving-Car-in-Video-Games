import glob
import os
from skimage import io
from utils import IOHandler
import json
from tqdm import tqdm
import argparse
from skimage import img_as_ubyte
import cv2


def keys_to_id(keys: str) -> int:
    keys = keys.lower()
    if keys == "a":
        return 1
    if keys == "d":
        return 2
    if keys == "w":
        return 3
    if keys == "s":
        return 4
    if keys == "aw" or keys == "wa":
        return 5
    if keys == "as" or keys == "sa":
        return 6
    if keys == "dw" or keys == "wd":
        return 7
    if keys == "ds" or keys == "ds":
        return 8

    return 0


def restore_dictionary(annotation_path: str):
    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as index_file:
            return json.load(index_file)
    else:
        return {"total": 0, "correct": 0, "human_predictions": {}}


def human_baseline(gold_dataset_dir: str, annotation_path: str):
    files = glob.glob(os.path.join(gold_dataset_dir, "*.jpeg"))
    io_handler = IOHandler()
    input_dictionary = restore_dictionary(annotation_path=annotation_path)

    try:
        pbar_desc = (
            "-1"
            if input_dictionary["total"] == 0
            else f"Current human accuracy: {round((input_dictionary['correct']/input_dictionary['total'])*100,2)}%"
        )
        with tqdm(total=len(files) - input_dictionary["total"], desc=pbar_desc) as pbar:
            for image_name in files:
                metadata = os.path.basename(image_name)[:-5]
                header, values = metadata.split("%")
                image_no = int(header[1:])
                if image_no not in input_dictionary["human_predictions"]:
                    gold_key = io_handler.imagename_input_conversion(
                        image_name=image_name, output_type="keyboard"
                    )

                    # image = io.imread(image_name)
                    os.system(f"xv {image_name} &")
                    # cv2.imshow("window1", img_as_ubyte(image))
                    # cv2.waitKey(1)
                    user_key = keys_to_id(input("Push the keys: "))

                    input_dictionary["human_predictions"][image_no] = user_key
                    input_dictionary["total"] += 1
                    if user_key == gold_key:
                        input_dictionary["correct"] += 1

                    pbar.update(1)

                    pbar.set_description(
                        f"Current human accuracy: {round((input_dictionary['correct']/input_dictionary['total'])*100,2)}%"
                    )

                    if input_dictionary["total"] % 20 == 0:
                        with open(
                            annotation_path, "w+", encoding="utf8"
                        ) as annotation_file:
                            json.dump(input_dictionary, annotation_file)

    except KeyboardInterrupt:
        with open(annotation_path, "w+", encoding="utf8") as annotation_file:
            json.dump(input_dictionary, annotation_file)

    with open(annotation_path, "w+", encoding="utf8") as annotation_file:
        json.dump(input_dictionary, annotation_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gold_dataset_dir",
        type=str,
        help="Path of the gold dataset directory",
    )

    parser.add_argument(
        "--annotation_path",
        type=str,
        help="Path where the human predictions will we stored, if it exists we will reload the progress",
    )

    args = parser.parse_args()

    human_baseline(
        gold_dataset_dir=args.gold_dataset_dir, annotation_path=args.annotation_path
    )
