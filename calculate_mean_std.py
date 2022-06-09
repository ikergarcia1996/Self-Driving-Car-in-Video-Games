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

    means = []
    stds = []

    for img_name in tqdm(dataset_files, desc="Reading images"):
        images = torchvision.io.read_image(img_name)
        y = 0
        images, _ = transform((images, y))
        for image in images:
            means.append(torch.mean(image / 255.0))
            stds.append(torch.std(image / 255.0))

    mean = torch.mean(torch.tensor(means))
    std = torch.mean(torch.tensor(stds))

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
