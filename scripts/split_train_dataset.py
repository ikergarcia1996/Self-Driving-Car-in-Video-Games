import glob
import os
from shutil import copyfile
from tqdm.auto import tqdm
from shlex import quote
import argparse
import math


def split_and_compress_dataset(
    dataset_dir: str,
    output_dir: str,
    splits: int = 20,
):
    """
    Split the dataset into "splits" subfolders of images and compress them.

    :param str dataset_dir: Path to the dataset directory.
    :param str output_dir: Path to the output directory.
    :param int splits: Number of splits to create.

    """

    dataset_files = glob.glob(os.path.join(dataset_dir, "*.jpeg"))
    img_per_file = math.ceil(len(dataset_files) / splits)

    print(
        f"Splitting dataset into {splits} subfolders of {img_per_file} images each. Total images: {len(dataset_files)}"
    )
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print("This may take a while, go grab a coffee!")
    print()

    # Split the dataset into multiple subfolders of img_per_file images
    for i in tqdm(
        range(0, len(dataset_files), img_per_file), desc="Splitting dataset", position=0
    ):
        os.makedirs(os.path.join(output_dir, str(i // img_per_file)), exist_ok=True)
        for dataset_file in tqdm(
            dataset_files[i : i + img_per_file], desc="Copying images", position=1
        ):
            copyfile(
                dataset_file,
                os.path.join(
                    output_dir,
                    str(i // img_per_file),
                    os.path.basename(dataset_file),
                ),
            )

        # Create zip file

        filename = f"TEDD1140_dataset_{i // img_per_file}.zip"
        os.system(
            f"zip -r {quote(os.path.join(output_dir, filename))}.zip "
            f"{quote(os.path.join(output_dir, str(i // img_per_file)))}"
        )
        # Remove folder
        os.system(f"rm -rf {quote(os.path.join(output_dir, str(i // img_per_file)))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into multiple subfolders and compress them."
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset dir",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output dir",
    )

    parser.add_argument(
        "--splits",
        type=int,
        default=20,
        help="Number of splits to create",
    )

    args = parser.parse_args()

    split_and_compress_dataset(
        dataset_dir=args.dataset_dir, output_dir=args.output_dir, splits=args.splits
    )
