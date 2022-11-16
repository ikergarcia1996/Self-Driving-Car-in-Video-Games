import os
import argparse
from model import Tedd1104ModelPL
from dataset import Tedd1104Dataset
import pytorch_lightning as pl
from typing import List, Union
from torch.utils.data import DataLoader
from tabulate import tabulate
from dataset import collate_fn, set_worker_sharing_strategy
from pytorch_lightning import loggers as pl_loggers


def eval_model(
    checkpoint_path: str,
    test_dirs: List[str],
    batch_size: int,
    dataloader_num_workers: int = 16,
    output_path: str = None,
    devices: str = 1,
    accelerator: str = "auto",
    precision: str = "16",
    strategy=None,
    report_to: str = "none",
    experiment_name: str = "test",
):

    """
    Evaluates a trained model on a set of test data.

    :param str checkpoint_path: Path to the checkpoint file.
    :param List[str] test_dirs: List of directories containing test data.
    :param int batch_size: Batch size for the dataloader.
    :param int dataloader_num_workers: Number of workers for the dataloader.
    :param str output_path: Path to where the results should be saved.
    :param str devices: Number of devices to use.
    :param str accelerator: Accelerator to use. If 'auto', tries to automatically detect TPU, GPU, CPU or IPU system.
    :param str precision: Precision to use. Double precision (64), full precision (32), half precision (16) or bfloat16
                          precision (bf16). Can be used on CPU, GPU or TPUs.
    :param str strategy: Strategy to use for data parallelism. "None" for no data parallelism,
                         ddp_find_unused_parameters_false for DDP.
    :param str report_to: Where to report the results. "none" for no reporting, "tensorboard" for TensorBoard,
                          "wandb" for W&B.
    :param str experiment_name: Name of the experiment for W&B.
    """
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    print(f"Restoring model from {checkpoint_path}")
    model = Tedd1104ModelPL.load_from_checkpoint(checkpoint_path=checkpoint_path)

    if report_to == "tensorboard":
        logger = pl_loggers.TensorBoardLogger(
            save_dir=os.path.dirname(checkpoint_path),
            name=experiment_name,
        )
    elif report_to == "wandb":
        logger = pl_loggers.WandbLogger(
            name=experiment_name,
            # id=experiment_name,
            # resume=None,
            project="TEDD1104",
            save_dir=os.path.dirname(checkpoint_path),
        )
    elif report_to == "none":
        logger = None
    else:
        raise ValueError(
            f"Unknown logger: {report_to}. Please use 'tensorboard' or 'wandb'."
        )

    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        precision=precision if precision == "bf16" else int(precision),
        strategy=strategy,
        # default_root_dir=os.path.join(
        #    os.path.dirname(os.path.abspath(checkpoint_path)), "trainer_checkpoint"
        # ),
    )

    results: List[List[Union[str, float]]] = []
    for test_dir in test_dirs:

        dataloader = DataLoader(
            Tedd1104Dataset(
                dataset_dir=test_dir,
                hide_map_prob=0.0,
                dropout_images_prob=[0.0, 0.0, 0.0, 0.0, 0.0],
                control_mode="keyboard",
                token_mask_prob=0.0,
                train=False,
                transformer_nheads=None
                if model.encoder_type == "lstm"
                else model.nhead,
            ),
            batch_size=batch_size,
            num_workers=dataloader_num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            collate_fn=collate_fn,
            worker_init_fn=set_worker_sharing_strategy,
        )
        print(f"Testing dataset: {os.path.basename(test_dir)}: ")
        print()
        out = trainer.test(
            ckpt_path=checkpoint_path,
            model=model,
            dataloaders=[dataloader],
            verbose=False,
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

        if logger is not None:
            log_metric_dict = {}
            for metric_name, metric_value in out.items():
                log_metric_dict[
                    f"{os.path.basename(test_dir)}/{metric_name.split('/')[-1]}"
                ] = metric_value
            logger.log_metrics(log_metric_dict, step=0)

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

    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs/TPUs to use. ",
    )

    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "tpu", "gpu", "cpu", "ipu"],
        help="Accelerator to use. If 'auto', tries to automatically detect TPU, GPU, CPU or IPU system",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["bf16", "16", "32", "64"],
        help=" Double precision (64), full precision (32), "
        "half precision (16) or bfloat16 precision (bf16). "
        "Can be used on CPU, GPU or TPUs.",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Supports passing different training strategies with aliases (ddp, ddp_spawn, etc)",
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard", "none"],
        help="Report to wandb or tensorboard",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="test",
        help="Experiment name for wandb",
    )

    args = parser.parse_args()

    eval_model(
        checkpoint_path=args.checkpoint_path,
        test_dirs=args.test_dirs,
        batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        output_path=args.output_path,
        devices=args.devices,
        accelerator=args.accelerator,
        precision=args.precision,
        strategy=args.strategy,
        report_to=args.report_to,
        experiment_name=args.experiment_name,
    )
