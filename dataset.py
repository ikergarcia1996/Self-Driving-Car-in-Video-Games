from __future__ import print_function, division
import os
import torch
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from typing import List, Optional
from utils import IOHandler
import pytorch_lightning as pl


class RemoveMinimap(object):
    """Remove minimap (black square) from the sequence"""

    def __init__(self, hide_map_prob):
        """
        Init

        Input:
        -hide_map_prob: Probability for removing the minimap (black square)
          from the sequence of images (0<=hide_map_prob<=1)
        """
        self.hide_map_prob = hide_map_prob

    def __call__(self, sample):
        image, y = (
            sample["image"],
            sample["y"],
        )

        width: int = int(image.shape[1] / 5)

        if self.hide_map_prob > 0:
            if torch.rand(1)[0] <= self.hide_map_prob:
                for j in range(0, 5):
                    image[215:, j * width : (j * width) + 80] = np.zeros(
                        (55, 80, 3), dtype=image.dtype
                    )

        return {
            "image": image,
            "y": y,
        }


class RemoveImage(object):
    """Remove images (black square) from the sequence"""

    def __init__(self, dropout_images_prob):
        """
        Init

        Input:
        - dropout_images_prob List of 5 floats or None, probability for removing each input image during training
         (black image) from a training example (0<=dropout_images_prob<=1)
        """
        self.dropout_images_prob = dropout_images_prob

    def __call__(self, sample):
        image, y = (
            sample["image"],
            sample["y"],
        )
        width: int = int(image.shape[1] / 5)

        for j in range(0, 5):
            if self.dropout_images_prob[j] > 0:
                if torch.rand(1)[0] <= self.dropout_images_prob[j]:
                    image[:, j * width : (j + 1) * width] = np.zeros(
                        (image.shape[0], width, image.shape[2]), dtype=image.dtype
                    )

        return {
            "image": image,
            "y": y,
        }


class SplitImages(object):
    """Splits the sequence into 5 images"""

    def __call__(self, sample):
        image, y = sample["image"], sample["y"]
        width: int = int(image.shape[1] / 5)
        return {
            "image1": image[:, 0:width],
            "image2": image[:, width : width * 2],
            "image3": image[:, width * 2 : width * 3],
            "image4": image[:, width * 3 : width * 4],
            "image5": image[:, width * 4 : width * 5],
            "y": y,
        }


class MergeImages(object):
    """Prepares the images for the model, unique dictionary instead of 5"""

    def __call__(self, sample):
        image1, image2, image3, image4, image5, y = (
            sample["image1"],
            sample["image2"],
            sample["image3"],
            sample["image4"],
            sample["image5"],
            sample["y"],
        )

        return {
            "images": torch.stack([image1, image2, image3, image4, image5]),
            "y": y,
        }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image1, image2, image3, image4, image5, y = (
            sample["image1"],
            sample["image2"],
            sample["image3"],
            sample["image4"],
            sample["image5"],
            sample["y"],
        )

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image1 = image1.transpose((2, 0, 1))
        image2 = image2.transpose((2, 0, 1))
        image3 = image3.transpose((2, 0, 1))
        image4 = image4.transpose((2, 0, 1))
        image5 = image5.transpose((2, 0, 1))
        return {
            "image1": torch.from_numpy(image1),
            "image2": torch.from_numpy(image2),
            "image3": torch.from_numpy(image3),
            "image4": torch.from_numpy(image4),
            "image5": torch.from_numpy(image5),
            "y": torch.tensor(y),
        }


class Normalize(object):
    """Normalize image"""

    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    def __call__(self, sample):
        image1, image2, image3, image4, image5, y = (
            sample["image1"],
            sample["image2"],
            sample["image3"],
            sample["image4"],
            sample["image5"],
            sample["y"],
        )
        return {
            "image1": self.transform(image1 / 255.0),
            "image2": self.transform(image2 / 255.0),
            "image3": self.transform(image3 / 255.0),
            "image4": self.transform(image4 / 255.0),
            "image5": self.transform(image5 / 255.0),
            "y": y,
        }


class Tedd1104Dataset(Dataset):
    """TEDD1104 dataset."""

    def __init__(
        self,
        dataset_dir: str,
        hide_map_prob: float,
        dropout_images_prob: List[float],
        control_mode: str = "keyboard",
    ):
        """
        Init

        Input:
        -dataset_dir: Directory containing the dataset files
        -hide_map_prob: Probability for removing the minimap (black square)
          from the sequence of images (0<=hide_map_prob<=1)
        - dropout_images_prob List of 5 floats or None, probability for removing each input image during training
         (black image) from a training example (0<=dropout_images_prob<=1)
        - control_mode: Set if the dataset true values will be keyboard inputs (9 classes)
          or Controller Inputs (2 continuous values)
        """

        self.dataset_dir = dataset_dir
        self.hide_map_prob = hide_map_prob
        self.dropout_images_prob = dropout_images_prob
        self.control_mode = control_mode.lower()

        assert self.control_mode in [
            "keyboard",
            "controller",
        ], f"{self.control_mode} control mode not supported. Supported dataset types: [keyboard, controller].  "

        assert 0 <= hide_map_prob <= 1.0, (
            f"hide_map_prob not in 0 <= hide_map_prob <= 1.0 range. "
            f"hide_map_prob: {hide_map_prob}"
        )

        assert len(dropout_images_prob) == 5, (
            f"dropout_images_prob must have 5 probabilities, one for each image in the sequence. "
            f"dropout_images_prob len: {len(dropout_images_prob)}"
        )

        for dropout_image_prob in dropout_images_prob:
            assert 0 <= dropout_image_prob <= 1.0, (
                f"All probabilities in dropout_image_prob must be in the range 0 <= dropout_image_prob <= 1.0. "
                f"dropout_images_prob: {dropout_images_prob}"
            )

        self.transform = transforms.Compose(
            [
                RemoveMinimap(hide_map_prob=hide_map_prob),
                RemoveImage(dropout_images_prob=dropout_images_prob),
                SplitImages(),
                ToTensor(),
                Normalize(),
                MergeImages(),
            ]
        )
        self.dataset_files = glob.glob(os.path.join(dataset_dir, "*.jpeg"))

        self.IOHandler = IOHandler()

    def __len__(self):
        return len(self.dataset_files)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.dataset_files[idx]
        image = None
        while image is None:
            try:
                image = io.imread(img_name)
            except (ValueError, FileNotFoundError) as err:
                error_message = str(err).split("\n")[-1]
                print(
                    f"Error reading image: {img_name} probably a corrupted file.\n"
                    f"Exception: {error_message}\n"
                    f"We will load a random image instead."
                )
                img_name = self.dataset_files[
                    int(len(self.dataset_files) * torch.rand(1))
                ]

        y = self.IOHandler.imagename_input_conversion(
            image_name=img_name,
            output_type=self.control_mode,
        )

        sample = {"image": image, "y": y}

        return self.transform(sample)


class Tedd1104ataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        test_dir: str,
        batch_size: int,
        hide_map_prob: float,
        dropout_images_prob: List[float],
        control_mode: str = "keyboard",
        num_workers: int = os.cpu_count(),
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size

        self.hide_map_prob = hide_map_prob
        self.dropout_images_prob = dropout_images_prob
        self.control_mode = control_mode

        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = Tedd1104Dataset(
                dataset_dir=self.train_dir,
                hide_map_prob=self.hide_map_prob,
                dropout_images_prob=self.dropout_images_prob,
                control_mode=self.control_mode,
            )

            print(f"Total training samples: {len(self.train_dataset)}.")

            self.val_dataset = Tedd1104Dataset(
                dataset_dir=self.val_dir,
                hide_map_prob=0.0,
                dropout_images_prob=[0.0, 0.0, 0.0, 0.0, 0.0],
                control_mode=self.control_mode,
            )

            print(f"Total validation samples: {len(self.val_dataset)}.")

        if stage in (None, "test"):
            self.test_dataset = Tedd1104Dataset(
                dataset_dir=self.test_dir,
                hide_map_prob=0.0,
                dropout_images_prob=[0.0, 0.0, 0.0, 0.0, 0.0],
                control_mode=self.control_mode,
            )

            print(f"Total test samples: {len(self.test_dataset)}.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
