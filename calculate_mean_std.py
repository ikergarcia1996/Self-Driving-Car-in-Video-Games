from dataset import SplitImages
import os
import glob
import torch
import torchvision
from tqdm.auto import tqdm
import argparse


def calculate_mean_str(dataset_dir: str):
    dataset_files = list(glob.glob(os.path.join(dataset_dir, "*.jpeg")))
    transform = torchvision.transforms.Compose(
        [
            SplitImages(),
        ]
    )

    mean_sum = torch.tensor(0)
    stds_sum = torch.tensor(0)
    total = 0
    for img_name in tqdm(
        dataset_files,
        desc=f"Reading images. Mean: {mean_sum/(total if total>0 else 1)}. STD: {stds_sum/(total if total>0 else 1)}",
    ):
        images = torchvision.io.read_image(img_name)
        y = 0
        images, _ = transform((images, y))
        for image in images:
            mean_sum += torch.mean(image / 255.0)
            stds_sum += torch.std(image / 255.0)
            total += 1

    mean = mean_sum / total
    std = stds_sum / total

    print(f"Mean: {mean}")
    print(f"std: {std}")

    return mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset directory containing jpeg files.",
    )

    args = parser.parse_args()

    calculate_mean_str(dataset_dir=args.dataset_dir)
