import os
import argparse
from model import Tedd1104ModelPL
from dataset import Tedd1104Dataset
import pytorch_lightning as pl
from typing import List, Union
from torch.utils.data import DataLoader
from tabulate import tabulate


def eval_model(
    checkpoint_path: str,
    test_dirs: List[str],
    batch_size: int,
    dataloader_num_workers: int = 16,
    output_path: str = None,
):

    """
    Evaluates a trained model on a set of test data.

    :param str checkpoint_path: Path to the checkpoint file.
    :param List[str] test_dirs: List of directories containing test data.
    :param int batch_size: Batch size for the dataloader.
    :param int dataloader_num_workers: Number of workers for the dataloader.
    :param str output_path: Path to where the results should be saved.
    """
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    print(f"Restoring model from {checkpoint_path}")
    model = Tedd1104ModelPL.load_from_checkpoint(checkpoint_path=checkpoint_path)

    trainer = pl.Trainer(
        precision=16,
        gpus=1,
        # accelerator="ddp",
        default_root_dir=os.path.join(
            os.path.dirname(os.path.abspath(checkpoint_path)), "trainer_checkpoint"
        ),
    )

    results: List[List[Union[str, float]]] = []
    for test_dir in test_dirs:

        dataloader = DataLoader(
            Tedd1104Dataset(
                dataset_dir=test_dir,
                hide_map_prob=0.0,
                dropout_images_prob=[0.0, 0.0, 0.0, 0.0, 0.0],
                control_mode="keyboard",
            ),
            batch_size=batch_size,
            num_workers=dataloader_num_workers,
            pin_memory=True,
            shuffle=False,
        )
        print(f"Testing dataset: {os.path.basename(test_dir)}: ")
        print()
        out = trainer.test(
            ckpt_path=checkpoint_path, model=model, dataloaders=[dataloader]
        )[0]

        results.append(
            [
                os.path.basename(test_dir),
                round(out["Test/acc_k@1_micro"] * 100, 1),
                round(out["Test/acc_k@3_micro"] * 100, 1),
                round(out["Test/acc_k@1_macro"] * 100, 1),
                round(out["Test/acc_k@3_macro"] * 100, 1),
            ]
        )

    print(
        tabulate(
            results,
            headers=[
                "Micro-Accuracy K@1",
                "Micro-Accuracy K@3",
                "Macro-Accuracy K@1",
                "Macro-Accuracy K@3",
            ],
        )
    )

    if output_path:
        with open(output_path, "w+", encoding="utf8") as output_file:
            print(
                tabulate(
                    results,
                    headers=[
                        "Micro-Accuracy K@1",
                        "Micro-Accuracy K@3",
                        "Macro-Accuracy K@1",
                        "Macro-Accuracy K@3",
                    ],
                    tablefmt="tsv",
                ),
                file=output_file,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate a trained model.")

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the checkpoint file.",
    )

    parser.add_argument(
        "--test_dirs",
        type=str,
        nargs="+",
        help="List of directories containing test data.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size for the dataloader.",
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=min(os.cpu_count(), 16),
        help="Number of workers for the dataloader.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to where the results should be saved.",
    )

    args = parser.parse_args()

    eval_model(
        checkpoint_path=args.checkpoint_path,
        test_dirs=args.test_dirs,
        batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        output_path=args.output_path,
    )
