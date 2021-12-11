import torch
from torch.nn import functional
from torchvision import transforms
import numpy as np
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation


def cityscapes_palette():
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
    """Prepares the images for the model, unique dictionary instead of 5"""

    def __init__(self, size=(1024, 1024)):
        self.size = size

    def __call__(self, images):

        return functional.interpolate(
            images, size=self.size, mode="bilinear", align_corners=False,
        )


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, images):
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
    """Prepares the images for the model, unique dictionary instead of 5"""

    def __call__(self, images):
        image1, image2, image3, image4, image5 = (
            images[0],
            images[1],
            images[2],
            images[3],
            images[4],
        )

        return torch.stack([image1, image2, image3, image4, image5])


class ImageSegmentation:
    def __init__(
        self,
        device: torch.device,
        model_name: str = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    ):
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.device = device
        self.model = self.model.to(device=self.device)

        self.image_transforms = transforms.Compose(
            [ToTensor(), MergeImages(), SequenceResize()]
        )

    def add_segmentation(self, images: np.ndarray):
        """
        Given a list of images, we will perform image segmentation and the detected entities will be
        printed over the original image to highlight them.

        Input:
        -images: Array of images (num_images x height x width x num_channels)
        Output:
        -images modified with segmented entities printed over them: (num_images x height x width x num_channels)
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
