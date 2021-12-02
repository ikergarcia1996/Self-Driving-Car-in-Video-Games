from __future__ import print_function, division
import os
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from typing import List, Optional
from utils import IOHandler
import pytorch_lightning as pl
from dataset import (
    RemoveMinimap,
    RemoveImage,
    SplitImages,
    MergeImages,
    Normalize,
    SequenceColorJitter,
)


class ReOrderImages(object):
    """Reorders the image given a tensor of positions"""

    def __call__(self, sample):
        images, y = (
            sample["images"],
            sample["y"],
        )

        return {
            "images": images[y],
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
            "y": y,
        }


class Tedd1104Dataset(Dataset):
    """TEDD1104 dataset."""

    def __init__(
        self,
        dataset_dir: str,
        hide_map_prob: float,
        dropout_images_prob: List[float],
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
                SequenceColorJitter(),
                Normalize(),
                MergeImages(),
                ReOrderImages(),
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

        y = torch.randperm(5)

        sample = {"image": image, "y": y}

        return self.transform(sample)


class Tedd1104ataModuleForImageReordering(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train_dir: str = None,
        val_dir: str = None,
        test_dir: str = None,
        hide_map_prob: float = 0.0,
        dropout_images_prob: List[float] = None,
        num_workers: int = os.cpu_count(),
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size

        self.hide_map_prob = hide_map_prob
        self.dropout_images_prob = (
            dropout_images_prob if dropout_images_prob else [0.0, 0.0, 0.0, 0.0, 0.0]
        )

        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = Tedd1104Dataset(
                dataset_dir=self.train_dir,
                hide_map_prob=self.hide_map_prob,
                dropout_images_prob=self.dropout_images_prob,
            )

            print(f"Total training samples: {len(self.train_dataset)}.")

            self.val_dataset = Tedd1104Dataset(
                dataset_dir=self.val_dir,
                hide_map_prob=0.0,
                dropout_images_prob=[0.0, 0.0, 0.0, 0.0, 0.0],
            )

            print(f"Total validation samples: {len(self.val_dataset)}.")

        if stage in (None, "test"):
            self.test_dataset = Tedd1104Dataset(
                dataset_dir=self.test_dir,
                hide_map_prob=0.0,
                dropout_images_prob=[0.0, 0.0, 0.0, 0.0, 0.0],
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
