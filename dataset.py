import os
import torch
from torch.utils.data import Dataset
import glob
from typing import List
from utils import IOHandler
import numpy as np
from PIL import Image
import logging
from image_processing_videomae import VideoMAEImageProcessor
import random
import torch.multiprocessing
from constants import IMAGE_MEAN, IMAGE_STD
from torchvision import transforms

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

    def __call__(self, image: np.array) -> torch.tensor:
        """
        Applies the transformation to the sequence of images.

        Args:
            image (np.array): Sequence of images. Size (270, 2400, 3)

        Returns:
            torch.tensor: Transformed sequence of images. Size (5, 270, 480, 3)
        """
        print(image.shape)
        width: int = int(image.shape[1] / 5)
        image1 = torch.from_numpy(image[:, 0:width, :])
        print(image1.size())
        image2 = torch.from_numpy(image[:, width : width * 2, :])
        print(image2.size())
        image3 = torch.from_numpy(image[:, width * 2 : width * 3, :])
        print(image3.size())
        image4 = torch.from_numpy(image[:, width * 3 : width * 4, :])
        print(image4.size())
        image5 = torch.from_numpy(image[:, width * 4 : width * 5, :])
        print(image5.size())
        return torch.stack([image1, image2, image3, image4, image5])


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


class ImageMaskingGenerator(object):
    """
    Mask images in a sequence
    """

    def __init__(self, mask_ratio, patch_size, tubelet_size):
        """
        INIT
        Args:
            mask_ratio (float): Ratio of the image to be masked
        """

        self.seq_length = 5 // tubelet_size * 270 // patch_size * 480 // patch_size
        self.image_length = 270 // patch_size * 480 // patch_size
        self.mask_ratio = mask_ratio
        self.tubelet_size = tubelet_size

        if self.tubelet_size == 5:
            logging.warning(
                "You have set mask_ratio > 0, however tubelet_size is 5. "
                "Therefore, sequences are tokenized as a single image. "
                "We cannot mask the images or the whole sequence will be masked. "
                "We will set mask_ratio to 0."
            )
            self.mask_ratio = 0.0

    def __call__(self):
        """
        Generate an image mask for a sequence of images

        Returns:
            torch.tensor: Tube mask
        """
        if self.tubelet_size > 0:
            return torch.zeros(self.seq_length, dtype=torch.bool)

        bernolli_matrix: torch.tensor = torch.cat(
            ((torch.tensor([self.mask_ratio]).float()).repeat(5),),
            0,
        )

        bernolli_distributor = torch.distributions.Bernoulli(bernolli_matrix)
        sample: torch.tensor = bernolli_distributor.sample()
        mask: torch.tensor = sample > 0

        mask = [
            torch.ones(self.image_length, dtype=torch.bool)
            if m
            else torch.zeros(self.image_length, dtype=torch.bool)
            for m in mask
        ]

        mask = torch.vstack(mask).view(1, -1).squeeze(0)
        mask.requires_grad = False
        return mask


def merge_masks(mask1: torch.tensor, mask2: torch.tensor) -> torch.tensor:
    """
    Merge two masks

    Args:
        mask1 (torch.tensor): Mask 1
        mask2 (torch.tensor): Mask 2

    Returns:
        torch.tensor: Merged mask
    """

    mask = mask1 | mask2
    return mask


class SequenceColorJitter(object):
    """
    Randomly change the brightness, contrast and saturation of a sequence of images
    """

    def __init__(self, brightness=0.5, contrast=0.1, saturation=0.1, hue=0.5):
        """
        INIT

        Args:
            brightness (float): Probability of changing brightness (0<=brightness<=1)
            contrast (float): Probability of changing contrast (0<=contrast<=1)
            saturation (float): Probability of changing saturation (0<=saturation<=1)
            hue (float): Probability of changing hue (0<=hue<=1)

        """
        self.jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, images: torch.tensor) -> torch.tensor:
        """
        Applies the transformation to the sequence of images.

        Args:
            images (torch.tensor): Sequence of images. Size (5, 270, 480, 3)

        Returns:
            torch.tensor: Transformed sequence of images. Size (5, 270, 480, 3)
        """

        images = images.permute(0, 3, 1, 2)
        images = self.jitter(images)
        images = images.permute(0, 2, 3, 1)
        return images


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")


class Tedd1104Dataset(Dataset):
    """TEDD1104 dataset for video classification"""

    def __init__(
        self,
        dataset_dir: str,
        hide_map_prob: float,
        tubelet_mask_ratio: float,
        image_mask_ratio: float,
        patch_size: int,
        tubelet_size: int,
        task: str = "video-classification",
        inference: bool = False,
    ):
        """
        INIT
        Args:
            dataset_dir (str): Path to the dataset directory
            hide_map_prob (float): Probability of hiding the minimap
            tubelet_mask_ratio (float): tubelets masking ratio (https://arxiv.org/pdf/2203.12602.pdf)
            image_mask_ratio (float): whole images masking ratio
            patch_size (int): Patch size
            tubelet_size (int): Tubelet size
            task (str): Task to perform. One of: video-classification, video-masking
            inference (bool): If True, we do not apply any transformation to the images
        """
        if not (0 <= hide_map_prob <= 1.0):
            raise ValueError(
                f"hide_map_prob not in 0 <= hide_map_prob <= 1.0 range. "
                f"hide_map_prob: {hide_map_prob}"
            )

        if not (0 <= tubelet_mask_ratio <= 1.0):
            raise ValueError(
                f"tubelet_mask_ratio not in 0 <= tubelet_mask_ratio <= 1.0 range. "
                f"tubelet_mask_ratio: {tubelet_mask_ratio}"
            )

        if not (0 <= image_mask_ratio <= 1.0):
            raise ValueError(
                f"image_mask_ratio not in 0 <= image_mask_ratio <= 1.0 range. "
                f"image_mask_ratio: {image_mask_ratio}"
            )

        if task not in ["video-classification", "video-masking"]:
            raise ValueError(
                f"task not in ['video-classification', 'video-masking']. "
                f"task: {task}"
            )

        self.inference = inference
        self.dataset_dir = dataset_dir
        self.hide_map_prob = hide_map_prob
        self.task = task
        self.control_mode = "keyboard".lower()
        self.image_splitter = SplitImages()
        self.image_color_jitter = SequenceColorJitter()

        self.tubelet_mask_generator = TubeMaskingGenerator(
            mask_ratio=0.0 if inference else tubelet_mask_ratio,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
        )

        self.image_mask_generator = ImageMaskingGenerator(
            mask_ratio=0.0 if inference else image_mask_ratio,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
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
            f"   Tubelet mask ratio: {tubelet_mask_ratio}\n"
            f"   Image mask ratio: {image_mask_ratio}\n"
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
                images: List[torch.tensor] = self.image_splitter(image)
                if not self.inference:
                    images = self.image_color_jitter(images)

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
            images=list(images), input_data_format="channels_last", return_tensors="pt"
        )
        # Remove the batch dimension
        model_inputs["pixel_values"] = model_inputs["pixel_values"][0]

        mask1 = self.tubelet_mask_generator()
        print(mask1.size())
        mask2 = self.image_mask_generator()
        print(mask2.size())
        mask = merge_masks(mask1, mask2)

        model_inputs["bool_masked_pos"] = mask
        if self.task == "video-classification":
            model_inputs["labels"] = torch.tensor(y, dtype=torch.long)

        return model_inputs
