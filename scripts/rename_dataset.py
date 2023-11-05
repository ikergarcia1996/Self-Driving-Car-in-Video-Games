import os
import glob
import argparse
from tqdm import tqdm


def rename_dataset(dataset_dir: str):
    """
    Rename legacy dataset to be consistent with the V5 naming convention.

    :param str dataset_dir: Path to the dataset directory.
    """
    dataset_files = glob.glob(os.path.join(dataset_dir, "*.jpeg"))
    for dataset_file in tqdm(dataset_files):
        metadata = os.path.basename(dataset_file)[:-5]
        imageno, key = metadata.split("_")

        y = [[-1], [-1], [-1], [-1], [key]]

        new_name = (
            "K"
            + str(imageno)
            + "%"
            + "_".join([",".join([str(e) for e in elem]) for elem in y])
            + ".jpeg"
        )

        new_name = os.path.join(dataset_dir, new_name)
        os.rename(dataset_file, new_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rename dataset to be consistent with V5 naming convention."
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset dir",
    )

    args = parser.parse_args()

    rename_dataset(args.dataset_dir)
