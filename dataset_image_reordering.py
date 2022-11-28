from __future__ import print_function, division
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from typing import List, Optional, Dict
from utils import IOHandler, get_mask
import pytorch_lightning as pl
from dataset import (
    RemoveMinimap,
    RemoveImage,
    SplitImages,
    Normalize,
    SequenceColorJitter,
    collate_fn,
    set_worker_sharing_strategy,
)


class ReOrderImages(object):
    """Reorders the image given a tensor of positions"""

    def __call__(self, sample: Dict[str, torch.tensor]) -> (torch.tensor, torch.tensor):
        """
        Applies the transformation to the sequence of images.

        :param Dict[str, torch.tensor] sample: Sequence of images
        :return: Dict[str, torch.tensor]- Reordered sequence of images
        """
        images, y = sample

        return images[y], sample


class Tedd1104Dataset(Dataset):
    """TEDD1104 Reordering dataset."""

    def __init__(
        self,
        dataset_dir: str,
        hide_map_prob: float,
        token_mask_prob: float,
        transformer_nheads: int = None,
        dropout_images_prob: List[float] = 0.0,
        sequence_length: int = 5,
        train: bool = False,
    ):
        """
        INIT

        :param str dataset_dir: The directory of the dataset.
        :param bool hide_map_prob: Probability of hiding the minimap (0<=hide_map_prob<=1)
        :param bool token_mask_prob: Probability of masking a token in the transformer model (0<=token_mask_prob<=1)
        :param int transformer_nheads: Number of heads in the transformer model, None if LSTM is used
        :param List[float] dropout_images_prob: Probability of dropping an image (0<=dropout_images_prob<=1)
        :param int sequence_length: Length of the image sequence
        :param bool train: If True, the dataset is used for training.
        """

        self.dataset_dir = dataset_dir
        self.hide_map_prob = hide_map_prob
        self.dropout_images_prob = dropout_images_prob
        self.sequence_length = sequence_length
        self.token_mask_prob = token_mask_prob
        self.transformer_nheads = transformer_nheads
        self.train = train

        assert 0 <= hide_map_prob <= 1.0, (
            f"hide_map_prob not in 0 <= hide_map_prob <= 1.0 range. "
            f"hide_map_prob: {hide_map_prob}"
        )

        assert len(dropout_images_prob) == 5, (
            f"dropout_images_prob must have 5 probabilities, one for each image in the sequence. "
            f"dropout_images_prob len: {len(dropout_images_prob)}"
        )

        for dropout_image_prob in dropout_images_prob:
            assert 0 <= dropout_image_prob < 1.0, (
                f"All probabilities in dropout_image_prob must be in the range 0 <= dropout_image_prob < 1.0. "
                f"dropout_images_prob: {dropout_images_prob}"
            )

        assert 0 <= token_mask_prob < 1.0, (
            f"token_mask_prob not in 0 <= token_mask_prob < 1.0 range. "
            f"token_mask_prob: {token_mask_prob}"
        )

        if train:
            self.transform = transforms.Compose(
                [
                    RemoveMinimap(hide_map_prob=hide_map_prob),
                    RemoveImage(dropout_images_prob=dropout_images_prob),
                    SplitImages(),
                    SequenceColorJitter(),
                    Normalize(),
                    ReOrderImages(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    RemoveMinimap(hide_map_prob=hide_map_prob),
                    # RemoveImage(dropout_images_prob=dropout_images_prob),
                    SplitImages(),
                    # SequenceColorJitter(),
                    Normalize(),
                    ReOrderImages(),
                ]
            )
        self.dataset_files = glob.glob(os.path.join(dataset_dir, "*.jpeg"))

        self.IOHandler = IOHandler()

    def __len__(self):
        """
        Returns the length of the dataset.

        :return: int - Length of the dataset.
        """
        return len(self.dataset_files)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.

        :param int idx: Index of the sample.
        :return: Dict[str, torch.tensor]- Transformed sequence of images
        """
        if torch.is_tensor(idx):
            idx = int(idx)

        img_name = self.dataset_files[idx]
        image = None
        while image is None:
            try:
                image = torchvision.io.read_image(img_name)
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

        image, y = self.transform((image, y))

        mask = get_mask(
            train=self.train,
            nheads=self.transformer_nheads,
            mask_prob=self.token_mask_prob,
            sequence_length=self.sequence_length,
        )

        return image, mask, y


class Tedd1104ataModuleForImageReordering(pl.LightningDataModule):
    """
    Tedd1104DataModule is a PyTorch Lightning DataModule for the Tedd1104 dataset.
    """

    def __init__(
        self,
        batch_size: int,
        train_dir: str = None,
        val_dir: str = None,
        test_dir: str = None,
        token_mask_prob: float = 0.0,
        transformer_nheads: int = None,
        sequence_length: int = 5,
        hide_map_prob: float = 0.0,
        dropout_images_prob: List[float] = None,
        num_workers: int = os.cpu_count(),
    ):
        """
        Initializes the Tedd1104DataModule.

        :param int batch_size: Batch size for the dataset.
        :param str train_dir: Directory containing the training dataset.
        :param str val_dir: Directory containing the validation dataset.
        :param str test_dir: Directory containing the test dataset.
        :param bool token_mask_prob: Probability of masking a token in the transformer model (0<=token_mask_prob<=1)
        :param int transformer_nheads: Number of heads in the transformer model, None if LSTM is used
        :param int sequence_length: Length of the image sequence
        :param bool hide_map_prob: Probability of hiding the minimap (0<=hide_map_prob<=1)
        :param float dropout_images_prob: Probability of dropping an image (0<=dropout_images_prob<=1)
        :param int num_workers: Number of workers to use to load the dataset.
        """

        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.token_mask_prob = token_mask_prob
        self.transformer_nheads = transformer_nheads
        self.sequence_length = sequence_length
        self.hide_map_prob = hide_map_prob
        self.dropout_images_prob = (
            dropout_images_prob if dropout_images_prob else [0.0, 0.0, 0.0, 0.0, 0.0]
        )

        if num_workers > 32:
            print(
                "WARNING: num_workers is greater than 32, this may cause memory issues, consider using a smaller value."
                "Go ahead if you have a lot of RAM."
            )

        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the dataset.

        :param str stage: Stage of the setup.
        """
        if stage in (None, "fit"):
            self.train_dataset = Tedd1104Dataset(
                dataset_dir=self.train_dir,
                hide_map_prob=self.hide_map_prob,
                dropout_images_prob=self.dropout_images_prob,
                train=True,
                token_mask_prob=self.token_mask_prob,
                transformer_nheads=self.transformer_nheads,
                sequence_length=self.sequence_length,
            )

            print(f"Total training samples: {len(self.train_dataset)}.")

            self.val_dataset = Tedd1104Dataset(
                dataset_dir=self.val_dir,
                hide_map_prob=0.0,
                dropout_images_prob=[0.0, 0.0, 0.0, 0.0, 0.0],
                token_mask_prob=0.0,
                transformer_nheads=self.transformer_nheads,
                sequence_length=self.sequence_length,
            )

            print(f"Total validation samples: {len(self.val_dataset)}.")

        if stage in (None, "test"):
            self.test_dataset = Tedd1104Dataset(
                dataset_dir=self.test_dir,
                hide_map_prob=0.0,
                dropout_images_prob=[0.0, 0.0, 0.0, 0.0, 0.0],
                token_mask_prob=0.0,
                transformer_nheads=self.transformer_nheads,
                sequence_length=self.sequence_length,
            )

            print(f"Total test samples: {len(self.test_dataset)}.")

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader.

        :return: DataLoader - Training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
            collate_fn=collate_fn,
            worker_init_fn=set_worker_sharing_strategy,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader.

        :return: DataLoader - Validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            collate_fn=collate_fn,
            worker_init_fn=set_worker_sharing_strategy,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader.

        :return: DataLoader - Test dataloader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            collate_fn=collate_fn,
            worker_init_fn=set_worker_sharing_strategy,
        )
