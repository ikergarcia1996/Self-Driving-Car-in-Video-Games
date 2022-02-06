"""
Experimental segmentation module based on SegFormer.
Only intended for testing purposes.
Only supported in inference, may be part of the model in the future.
It uses too much GPU resources, not viable for training or real time inference yet.
Nvidia pls launch faster GPUs :)

Requires the transformers library from huggingface to be installed (huggingface.co/transformers)
"""

import torch
from torch.nn import functional
from torchvision import transforms
import numpy as np
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from typing import List, Dict


def cityscapes_palette():
    """
    Returns the cityscapes palette.

    :return: List[List[int]] - The cityscapes palette.
    """
    return [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]


class SequenceResize(object):
    """Prepares the images for the model"""

    def __init__(self, size=(1024, 1024)):
        """
        INIT

        :param Tuple[int, int] size:  - The size of the output images.
        """
        self.size = size

    def __call__(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applies the transformation to the images.

        :param List[np.ndarray] images: - The images to transform.
        :return: List[np.ndarray] - The transformed images.
        """
        return functional.interpolate(
            images,
            size=self.size,
            mode="bilinear",
            align_corners=False,
        )


class ToTensor(object):
    """Convert np.ndarray images to Tensors."""

    def __call__(self, images: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Applies the transformation to the sequence of images.

        :param List[np.ndarray] images: - The images to transform.
        :return: List[torch.Tensor] - The transformed images.
        """
        image1, image2, image3, image4, image5 = (
            images[0],
            images[1],
            images[2],
            images[3],
            images[4],
        )

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image1 = image1.transpose((2, 0, 1)).astype(float)
        image2 = image2.transpose((2, 0, 1)).astype(float)
        image3 = image3.transpose((2, 0, 1)).astype(float)
        image4 = image4.transpose((2, 0, 1)).astype(float)
        image5 = image5.transpose((2, 0, 1)).astype(float)

        return [
            torch.from_numpy(image1),
            torch.from_numpy(image2),
            torch.from_numpy(image3),
            torch.from_numpy(image4),
            torch.from_numpy(image5),
        ]


class MergeImages(object):
    """Merges the images into one torch.Tensor"""

    def __call__(self, images: List[torch.tensor]) -> torch.tensor:
        """
        Applies the transformation to the sequence of images.

        :param List[torch.tensor] images: - The images to transform.
        :return: torch.Tensor - The transformed image.
        """
        image1, image2, image3, image4, image5 = (
            images[0],
            images[1],
            images[2],
            images[3],
            images[4],
        )

        return torch.stack([image1, image2, image3, image4, image5])


class ImageSegmentation:
    """
    Class for performing image segmentation.
    """

    def __init__(
        self,
        device: torch.device,
        model_name: str = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    ):
        """
        INIT

        :param torch.device device: - The device to use.
        :param str model_name: - The name of the model to use (https://huggingface.co/models)
        """
        print(f"Loading feature extractor for {model_name}")
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
        print(f"Loading segmentation model for {model_name}")
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.device = device
        self.model = self.model.to(device=self.device)

        self.image_transforms = transforms.Compose(
            [ToTensor(), MergeImages(), SequenceResize()]
        )

    def add_segmentation(self, images: np.ndarray) -> np.ndarray:
        """
        Adds the segmentation to the images. The segmentation is added as a mask over the original images.

        :param np.ndarray images: - The images to add the segmentation to.
        :return: np.ndarray - The images with the segmentation added.
        """

        original_image_size = images[0].shape
        inputs = torch.vstack(
            [
                self.feature_extractor(images=image, return_tensors="pt")[
                    "pixel_values"
                ]
                for image in images
            ]
        ).to(device=self.device)

        outputs = self.model(inputs).logits.detach().cpu()

        logits = functional.interpolate(
            outputs,
            size=(original_image_size[0], original_image_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        segmented_images = logits.argmax(dim=1)

        for image_no, seg in enumerate(segmented_images):
            color_seg = np.zeros(
                (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
            )  # height, width, 3
            palette = np.array(cityscapes_palette())
            for label, color in enumerate(palette):
                color_seg[seg == label, :] = color

            images[image_no] = images[image_no] * 0.5 + color_seg * 0.5

        return images
