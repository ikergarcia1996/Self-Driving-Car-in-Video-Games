from dataset import SplitImages
import os
import glob
import torch
import torchvision
from tqdm.auto import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader


class Tedd1104Dataset(Dataset):
    """TEDD1104 dataset."""

    def __init__(
        self,
        dataset_dir: str,
    ):
        self.dataset_dir = dataset_dir

        self.transform = torchvision.transforms.Compose(
            [
                SplitImages(),
            ]
        )

        self.dataset_files = glob.glob(os.path.join(dataset_dir, "*.jpeg"))

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
        :return: torch.tensor- Transformed sequence of images
        """
        if torch.is_tensor(idx):
            idx = int(idx)

        image = torchvision.io.read_image(self.dataset_files[idx])
        images, _ = self.transform((image, 0))
        return images


def collate_fn(batch):
    """
    Collate function for the dataloader.

    :param batch: List of samples
    :return: torch.tensor - Transformed sequence of images
    """

    return torch.cat(batch, dim=0)


def calculate_mean_str(dataset_dir: str):
    list(glob.glob(os.path.join(dataset_dir, "*.jpeg")))

    mean_sum = torch.tensor([0.0, 0.0, 0.0])
    stds_sum = torch.tensor([0.0, 0.0, 0.0])
    total = 0
    dataset = Tedd1104Dataset(dataset_dir=dataset_dir)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        collate_fn=collate_fn,
        num_workers=os.cpu_count() // 2,
    )
    with tqdm(
        total=len(dataloader),
        desc="Reading images",
    ) as pbar:
        for batch in dataloader:
            for image in batch:
                for dim in range(3):
                    channel = image[dim] / 255.0
                    mean_sum[dim] += torch.mean(channel)
                    stds_sum[dim] += torch.std(channel)
                total += 1
            pbar.update(1)
            pbar.set_description(
                desc=f"Reading images. "
                f"Mean: [{round(mean_sum[0].item()/total,6)},{round(mean_sum[1].item()/total,6)},{round(mean_sum[2].item()/total,6)}]. "
                f"STD: [{round(stds_sum[0].item()/total,6)},{round(stds_sum[1].item()/total,6)},{round(stds_sum[2].item()/total,6)}].",
            )

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

    mean, std = calculate_mean_str(dataset_dir=args.dataset_dir)
    with open("image_metrics.txt", "w", encoding="utf8") as output_file:
        print(f"Mean: {mean.numpy()}", file=output_file)
        print(f"STD: {std.numpy()}", file=output_file)
