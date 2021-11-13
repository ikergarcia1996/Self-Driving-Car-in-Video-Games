import os
import argparse
from model import Tedd1104ModelPL
from dataset import Tedd1104Dataset
import pytorch_lightning as pl
from typing import List
from torch.utils.data import DataLoader


def eval_model(
    checkpoint_path: str,
    test_dirs: List[str],
    batch_size: int,
    hparams_path: str = None,
    dataloader_num_workers: int = 16,
):

    if hparams_path is None:
        # Try to find hparams file
        model_dir = os.path.dirname(checkpoint_path)
        hparamsp = os.path.join(model_dir, "hparams.yaml")

        if os.path.exists(hparamsp):
            hparams_path = hparamsp

        else:
            model_dir = os.path.dirname(model_dir)
            hparamsp = os.path.join(model_dir, "hparams.yaml")
            if os.path.exists(hparamsp):
                hparams_path = hparamsp
            else:
                raise FileNotFoundError(
                    f"Unable to find an hparams.yaml file, "
                    f"please set the path for your hyperparameter file using the flag --hparams_path."
                )

    model = Tedd1104ModelPL.load_from_checkpoint(
        checkpoint_path=checkpoint_path, hparams_file=hparams_path
    )

    test_dataloaders = [
        DataLoader(
            Tedd1104Dataset(
                dataset_dir=dataset_dir,
                hide_map_prob=0.0,
                dropout_images_prob=[0.0, 0.0, 0.0, 0.0, 0.0],
                control_mode="keyboard",
            ),
            batch_size=batch_size,
            num_workers=dataloader_num_workers,
            pin_memory=True,
            shuffle=False,
        )
        for dataset_dir in test_dirs
    ]

    print(f"Restoring checkpoint: {checkpoint_path}. hparams: {hparams_path}")

    trainer = pl.Trainer(
        resume_from_checkpoint=checkpoint_path,
        precision=16,
        gpus=1,
        # accelerator="ddp",
        default_root_dir=os.path.join(
            os.path.dirname(os.path.abspath(checkpoint_path)), "trainer_checkpoint"
        ),
    )

    out = trainer.test(model, test_dataloaders=test_dataloaders)
    print(out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path of the checkpoint to evaluate",
    )

    parser.add_argument(
        "--hparams_path",
        type=str,
        default=None,
        help="[continue_training] Path of the hparams file for the current checkpoint,"
        "if not provided we will try to automatically find it",
    )

    parser.add_argument(
        "--test_dirs",
        type=str,
        nargs="+",
        help="Path of the test datasets directories",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Eval Batch size",
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=min(os.cpu_count(), 16),
        help="Eval Batch size",
    )

    args = parser.parse_args()

    eval_model(
        checkpoint_path=args.checkpoint_path,
        test_dirs=args.test_dirs,
        batch_size=args.batch_size,
        hparams_path=args.hparams_path,
        dataloader_num_workers=args.dataloader_num_workers,
    )
