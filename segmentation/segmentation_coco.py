"""
Semantic segmentation using pytorch pretrained models.
Classes: ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
Documentation: https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
"""
import torch
import torch.hub
from PIL import Image
from torchvision import transforms
import numpy as np


class ImageSegmentation:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        fp16: bool,
        apex_opt_level: str = "O2",
    ):
        """
        Image segmentation

        Input:
        -model_name: Name of the pretrained model to use from pytorch/vision
        -device: Torch.device where the model will be loaded (GPU recommended)
        -fp16: Use FP16 (only available if device = cuda:x)
        -apex_opt_level: Apex opt level for FP16
        Output:
        """
        if model_name == "fcn_resnet101":
            self.model = torch.hub.load(
                "pytorch/vision", "fcn_resnet101", pretrained=True
            ).to(device)

        elif model_name == "deeplabv3_resnet101":
            self.model = torch.hub.load(
                "pytorch/vision", "deeplabv3_resnet101", pretrained=True
            ).to(device)
        else:
            raise ValueError(
                f"model: {model_name} not supported. Choose between [fcn_resnet101, deeplabv3_resnet101]"
            )

        if fp16:
            try:
                from apex import amp

                self.model = amp.initialize(
                    self.model, opt_level=apex_opt_level, keep_batchnorm_fp32=True,
                )

            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )

        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        self.colors = (colors % 255).numpy().astype("uint8")

        self.device = device

    def add_segmentation(self, images: np.ndarray):
        """
        Given a list of images, we will perform image segmentation and the detected entities will be
        printed over the original image to highlight them.

        Input:
        -images: Array of images (num_images x height x width x num_channels)
        Output:
        -images modified with segmented entities printed over them: (num_images x height x width x num_channels)
        """

        img_size = (images.shape[2], images.shape[1])
        input_batch = torch.stack(
            [self.preprocess(Image.fromarray(image).convert("RGB")) for image in images]
        )

        input_batch.to(self.device)

        with torch.no_grad():
            output = self.model(input_batch.to(self.device))["out"]

        output_predictions = output.argmax(1).byte().cpu().numpy()

        segmented_images = [
            Image.fromarray(output_prediction).resize(img_size)
            for output_prediction in output_predictions
        ]

        [image.putpalette(self.colors) for image in segmented_images]

        masks = [np.array(image) != 0 for image in segmented_images]

        segmented_images = [
            np.array(image.convert("RGB")) for image in segmented_images
        ]

        for image, segmented_image, mask in zip(images, segmented_images, masks):
            image[mask] = segmented_image[mask]

        return images
