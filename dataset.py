import os
import torch
from torch.utils.data import Dataset
import glob
from typing import List
from utils import IOHandler
import numpy as np
from PIL import Image
import logging
from transformers import VideoMAEImageProcessor
import random
import torch.multiprocessing
from constants import IMAGE_MEAN, IMAGE_STD

torch.multiprocessing.set_sharing_strategy("file_system")


def count_examples(dataset_dir: str) -> int:
    return len(glob.glob(os.path.join(dataset_dir, "*.jpeg")))


def pil_to_numpy(im):
    """
    Converts a PIL Image object to a NumPy array.
    Source : Fast import of Pillow images to NumPy / OpenCV arrays Written by Alex Karpinsky

    Args:
        im (PIL.Image.Image): The input PIL Image object.

    Returns:
        numpy.ndarray: The NumPy array representing the image.
    """
    im.load()

    # Unpack data
    encoder = Image._getencoder(im.mode, "raw", im.mode)
    encoder.setimage(im.im)

    # NumPy buffer for the result
    shape, typestr = Image._conv_type_shape(im)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast("B", (data.data.nbytes,))

    bufsize, s, offset = 65536, 0, 0
    while not s:
        _, s, d = encoder.encode(bufsize)

        mem[offset : offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)
    return data


class SplitImages(object):
    """
    Splits a sequence image file into 5 images
    """

    def __call__(self, image: np.array) -> np.array:
        """
        Applies the transformation to the sequence of images.

        Args:
            image (np.array): Sequence of images. Size (270, 2400, 3)

        Returns:
            np.array: Transformed sequence of images. Size (5, 270, 480, 3)
        """

        width: int = int(image.shape[1] / 5)
        image1 = image[:, 0:width, :]
        image2 = image[:, width : width * 2, :]
        image3 = image[:, width * 2 : width * 3, :]
        image4 = image[:, width * 3 : width * 4, :]
        image5 = image[:, width * 4 : width * 5, :]
        return np.asarray([image1, image2, image3, image4, image5])


class RemoveMinimap(object):
    """Remove minimap (black square) from all the images in the sequence"""

    def __init__(self, hide_map_prob: float):
        """
        INIT
        Args:
            hide_map_prob (float): Probability of hiding the minimap
        """
        if not (0 <= hide_map_prob <= 1.0):
            raise ValueError(
                f"hide_map_prob not in 0 <= hide_map_prob <= 1.0 range. "
                f"hide_map_prob: {hide_map_prob}"
            )
        self.hide_map_prob = hide_map_prob

    def __call__(self, images: np.array) -> np.array:
        """
        Applies the transformation to the sequence of images.

        Args:
            images (np.array): Sequence of images. Size (5, 270, 480, 3)

        Returns:
            np.array: Transformed sequence of images. Size (5, 270, 480, 3)
        """

        if self.hide_map_prob > 0 and random.random() <= self.hide_map_prob:
            for j in range(0, 5):
                images[215:, j * 480 : (j * 480) + 80, :] = 0

        return images


class TubeMaskingGenerator(object):
    """
    Generate a tube mask for a sequence of images
    Adapted from: https://github.com/MCG-NJU/VideoMAE/blob/main/masking_generator.py
    """

    def __init__(self, mask_ratio, patch_size, tubelet_size):
        """
        INIT
        Args:
            mask_ratio (float): Ratio of the image to be masked
        """

        seq_length = 5 // tubelet_size * 270 // patch_size * 480 // patch_size
        masked_frames = int(mask_ratio * seq_length)
        unmasked_frames = seq_length - masked_frames
        self.mask = np.hstack(
            [
                np.zeros(unmasked_frames),
                np.ones(masked_frames),
            ]
        )

    def __call__(self):
        """
        Generate a tube mask for a sequence of images

        Returns:
            torch.tensor: Tube mask
        """

        np.random.shuffle(self.mask)
        mask = torch.from_numpy(self.mask).to(dtype=torch.bool)
        mask.requires_grad = False
        return mask


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")


class Tedd1104Dataset(Dataset):
    """TEDD1104 dataset for video classification"""

    def __init__(
        self,
        dataset_dir: str,
        hide_map_prob: float,
        mask_ratio: float,
        patch_size: int,
        tubelet_size: int,
        task: str = "video-classification",
    ):
        """
        INIT
        Args:
            dataset_dir (str): Path to the dataset directory
            hide_map_prob (float): Probability of hiding the minimap
            mask_ratio (float): Ratio of the image to be masked
            patch_size (int): Patch size
            tubelet_size (int): Tubelet size
            task (str): Task to perform. One of: video-classification, video-masking
        """
        if not (0 <= hide_map_prob <= 1.0):
            raise ValueError(
                f"hide_map_prob not in 0 <= hide_map_prob <= 1.0 range. "
                f"hide_map_prob: {hide_map_prob}"
            )

        if not (0 <= mask_ratio <= 1.0):
            raise ValueError(
                f"mask_ratio not in 0 <= mask_ratio <= 1.0 range. "
                f"mask_ratio: {mask_ratio}"
            )

        if task not in ["video-classification", "video-masking"]:
            raise ValueError(
                f"task not in ['video-classification', 'video-masking']. "
                f"task: {task}"
            )

        self.dataset_dir = dataset_dir
        self.hide_map_prob = hide_map_prob
        self.task = task
        self.control_mode = "keyboard".lower()
        self.image_splitter = SplitImages()
        self.mask_generator = TubeMaskingGenerator(
            mask_ratio=mask_ratio, patch_size=patch_size, tubelet_size=tubelet_size
        )

        if self.hide_map_prob > 0:
            self.map_remover = RemoveMinimap(self.hide_map_prob)
        else:
            self.map_remover = None

        self.image_processor = VideoMAEImageProcessor(
            do_resize=False,
            do_center_crop=False,
            do_rescale=True,
            do_normalize=True,
            image_mean=IMAGE_MEAN,
            image_std=IMAGE_STD,
        )

        self.dataset_files = glob.glob(os.path.join(dataset_dir, "*.jpeg"))

        if len(self.dataset_files) == 0:
            raise ValueError(f"No images found in {dataset_dir}")

        self.IOHandler = IOHandler()

        logging.info(
            f"Dataset: {dataset_dir}\n"
            f"   Number of images: {len(self.dataset_files)}\n"
            f"   Hide map probability: {self.hide_map_prob}\n"
            f"   Mask ratio: {mask_ratio}\n"
            f"   Task: {self.task}"
        )

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.dataset_files)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, torch.tensor]: Dictionary containing the model inputs.
        """
        if torch.is_tensor(idx):
            idx = int(idx)

        img_name = self.dataset_files[idx]
        images = []
        while len(images) == 0:
            try:
                # Read image into numpy array
                image: np.array = pil_to_numpy(Image.open(img_name))
                if self.map_remover:
                    image = self.map_remover(image)
                images: List[np.array] = self.image_splitter(image)

            except (ValueError, FileNotFoundError) as err:
                error_message = str(err).split("\n")[-1]
                logging.warning(
                    f"Error reading image: {img_name} probably a corrupted file.\n"
                    f"Exception: {error_message}\n"
                    f"We will load a random image instead."
                )
                img_name = np.random.choice(self.dataset_files)

        y = self.IOHandler.imagename_input_conversion(
            image_name=img_name,
            output_type=self.control_mode,
        )

        model_inputs = self.image_processor(
            images=images, input_data_format="channels_last", return_tensors="pt"
        )
        # Remove the batch dimension
        model_inputs["pixel_values"] = model_inputs["pixel_values"][0]

        if self.task == "video-classification":
            model_inputs["labels"] = torch.tensor(y, dtype=torch.long)
        elif self.task == "video-masking":
            model_inputs["bool_masked_pos"] = self.mask_generator()
        else:
            raise ValueError(
                f"task not in ['video-classification', 'video-masking']. "
                f"task: {self.task}"
            )

        return model_inputs
