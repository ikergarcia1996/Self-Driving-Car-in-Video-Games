from model import Tedd1104ModelPLForImageReordering
from typing import List
import argparse
from dataset_image_reordering import Tedd1104ataModuleForImageReordering
import os
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl


def train(
    model: Tedd1104ModelPLForImageReordering,
    train_dir: str,
    val_dir: str,
    output_dir: str,
    batch_size: int,
    accumulation_steps: int,
    max_epochs: int,
    hide_map_prob: float,
    dropout_images_prob: List[float],
    test_dir: str = None,
    mask_prob: float = 0.2,
    val_check_interval: float = 0.25,
    dataloader_num_workers=os.cpu_count(),
    devices: str = 1,
    accelerator: str = "auto",
    precision: str = "bf16",
    strategy=None,
    report_to: str = "wandb",
):
    """
    Train the model.

    :param Tedd1104ModelPL model: The model to train.
    :param str train_dir: The directory containing the training data.
    :param str val_dir: The directory containing the validation data.
    :param str output_dir: The directory to save the model to.
    :param int batch_size: The batch size.
    :param int accumulation_steps: The number of steps to accumulate gradients.
    :param int max_epochs: The maximum number of epochs to train for.
    :param bool hide_map_prob: Probability of hiding the minimap (0<=hide_map_prob<=1)
    :param float dropout_images_prob: Probability of dropping an image (0<=dropout_images_prob<=1)
    :param str test_dir: The directory containing the test data.
    :param float val_check_interval: The interval to check the validation accuracy.
    :param str devices: Number of devices to use.
    :param str accelerator: Accelerator to use. If 'auto', tries to automatically detect TPU, GPU, CPU or IPU system.
    :param str precision: Precision to use. Double precision (64), full precision (32), half precision (16) or bfloat16
                          precision (bf16). Can be used on CPU, GPU or TPUs.
    :param str strategy: Strategy to use for data parallelism. "None" for no data parallelism,
                         ddp_find_unused_parameters_false for DDP.
    :param int dataloader_num_workers: The number of workers to use for the dataloader.
    :param str report_to: Where to report the results. "tensorboard" for TensorBoard, "wandb" for W&B.
    """

    if not os.path.exists(output_dir):
        print(f"{output_dir} does not exits. We will create it.")
        os.makedirs(output_dir)

    data = Tedd1104ataModuleForImageReordering(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=batch_size,
        hide_map_prob=hide_map_prob,
        dropout_images_prob=dropout_images_prob,
        num_workers=dataloader_num_workers,
        token_mask_prob=mask_prob,
        transformer_nheads=None if model.encoder_type == "lstm" else model.nhead,
        sequence_length=model.sequence_size,
    )

    experiment_name = os.path.basename(
        output_dir if output_dir[-1] != "/" else output_dir[:-1]
    )
    if report_to == "tensorboard":
        logger = pl_loggers.TensorBoardLogger(
            save_dir=output_dir,
            name=experiment_name,
        )
    elif report_to == "wandb":
        logger = pl_loggers.WandbLogger(
            name=experiment_name,
            # id=experiment_name,
            # resume=None,
            project="TEDD1104_reorder",
            save_dir=output_dir,
        )
    else:
        raise ValueError(
            f"Unknown logger: {report_to}. Please use 'tensorboard' or 'wandb'."
        )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir, monitor="Validation/acc", mode="max", save_last=True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"

    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        precision=precision if precision == "bf16" else int(precision),
        strategy=strategy,
        val_check_interval=val_check_interval,
        accumulate_grad_batches=accumulation_steps,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        # default_root_dir=os.path.join(output_dir, "trainer_checkpoint"),
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=data)

    print(f"Best model path: {checkpoint_callback.best_model_path}")
    if test_dir:
        trainer.test(datamodule=data, ckpt_path="best")


def train_new_model(
    train_dir: str,
    val_dir: str,
    output_dir: str,
    batch_size: int,
    max_epochs: int,
    cnn_model_name: str,
    devices: str = 1,
    accelerator: str = "auto",
    precision: str = "bf16",
    strategy=None,
    accumulation_steps: int = 1,
    hide_map_prob: float = 0.0,
    test_dir: str = None,
    dropout_images_prob=None,
    val_check_interval: float = 0.25,
    dataloader_num_workers=os.cpu_count(),
    pretrained_cnn: bool = True,
    embedded_size: int = 512,
    nhead: int = 8,
    num_layers_encoder: int = 1,
    dropout_cnn_out: float = 0.1,
    positional_embeddings_dropout: float = 0.1,
    dropout_encoder: float = 0.1,
    dropout_encoder_features: float = 0.8,
    mask_prob: float = 0.0,
    sequence_size: int = 5,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-3,
    report_to: str = "wandb",
):

    """
    Train a new model.

    :param str train_dir: The directory containing the training data.
    :param str val_dir: The directory containing the validation data.
    :param str output_dir: The directory to save the model to.
    :param int batch_size: The batch size.
    :param int accumulation_steps: The number of steps to accumulate gradients.
    :param int max_epochs: The maximum number of epochs to train for.
    :param bool hide_map_prob: Probability of hiding the minimap (0<=hide_map_prob<=1)
    :param float dropout_images_prob: Probability of dropping an image (0<=dropout_images_prob<=1)
    :param str test_dir: The directory containing the test data.
    :param float val_check_interval: The interval to check the validation accuracy.
    :param str devices: Number of devices to use.
    :param str accelerator: Accelerator to use. If 'auto', tries to automatically detect TPU, GPU, CPU or IPU system.
    :param str precision: Precision to use. Double precision (64), full precision (32), half precision (16) or bfloat16
                          precision (bf16). Can be used on CPU, GPU or TPUs.
    :param str strategy: Strategy to use for data parallelism. "None" for no data parallelism,
                         ddp_find_unused_parameters_false for DDP.
    :param str report_to: Where to report the results. "tensorboard" for TensorBoard, "wandb" for W&B.
    :param int dataloader_num_workers: The number of workers to use for the dataloader.
    :param int embedded_size: Size of the output embedding
    :param float dropout_cnn_out: Dropout rate for the output of the CNN
    :param str cnn_model_name: Name of the CNN model from torchvision.models
    :param bool pretrained_cnn: If True, the model will be loaded with pretrained weights
    :param int embedded_size: Size of the input feature vectors
    :param int nhead: Number of heads in the multi-head attention
    :param int num_layers_encoder: number of transformer layers in the encoder
    :param float mask_prob: probability of masking each input vector in the transformer
    :param float positional_embeddings_dropout: Dropout rate for the positional embeddings
    :param int sequence_size: Length of the input sequence
    :param float dropout_encoder: Dropout rate for the encoder
    :param float dropout_encoder_features: Dropout probability of the encoder output
    :param float learning_rate: Learning rate
    :param float weight_decay: Weight decay
    """

    if dropout_images_prob is None:
        dropout_images_prob = [0.0, 0.0, 0.0, 0.0, 0.0]

    model: Tedd1104ModelPLForImageReordering = Tedd1104ModelPLForImageReordering(
        cnn_model_name=cnn_model_name,
        pretrained_cnn=pretrained_cnn,
        embedded_size=embedded_size,
        nhead=nhead,
        num_layers_encoder=num_layers_encoder,
        dropout_cnn_out=dropout_cnn_out,
        positional_embeddings_dropout=positional_embeddings_dropout,
        dropout_encoder=dropout_encoder,
        dropout_encoder_features=dropout_encoder_features,
        sequence_size=sequence_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        accelerator=accelerator,
    )

    train(
        model=model,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        max_epochs=max_epochs,
        mask_prob=mask_prob,
        hide_map_prob=hide_map_prob,
        dropout_images_prob=dropout_images_prob,
        val_check_interval=val_check_interval,
        dataloader_num_workers=dataloader_num_workers,
        devices=devices,
        accelerator=accelerator,
        precision=precision,
        strategy=strategy,
        report_to=report_to,
    )


def continue_training(
    checkpoint_path: str,
    train_dir: str,
    val_dir: str,
    batch_size: int,
    max_epochs: int,
    output_dir,
    accumulation_steps,
    devices: str = 1,
    accelerator: str = "auto",
    precision: str = "bf16",
    strategy=None,
    test_dir: str = None,
    hparams_path: str = None,
    mask_prob: float = 0.2,
    hide_map_prob: float = 0.0,
    dropout_images_prob=None,
    dataloader_num_workers=os.cpu_count(),
    val_check_interval: float = 0.25,
    report_to: str = "wandb",
):

    """
    Continues training a model from a checkpoint.

    :param str checkpoint_path: Path to the checkpoint to continue training from
    :param str train_dir: The directory containing the training data.
    :param str val_dir: The directory containing the validation data.
    :param str output_dir: The directory to save the model to.
    :param int batch_size: The batch size.
    :param int accumulation_steps: The number of steps to accumulate gradients.
    :param str devices: Number of devices to use.
    :param str accelerator: Accelerator to use. If 'auto', tries to automatically detect TPU, GPU, CPU or IPU system.
    :param str precision: Precision to use. Double precision (64), full precision (32), half precision (16) or bfloat16
                          precision (bf16). Can be used on CPU, GPU or TPUs.
    :param str strategy: Strategy to use for data parallelism. "None" for no data parallelism,
                         ddp_find_unused_parameters_false for DDP.
    :param str report_to: Where to report the results. "tensorboard" for TensorBoard, "wandb" for W&B.
    :param int max_epochs: The maximum number of epochs to train for.
    :param bool hide_map_prob: Probability of hiding the minimap (0<=hide_map_prob<=1)
    :param float dropout_images_prob: Probability of dropping an image (0<=dropout_images_prob<=1)
    :param str test_dir: The directory containing the test data.
    :param int dataloader_num_workers: The number of workers to use for the dataloaders.
    :param float val_check_interval: The interval in epochs to check the validation accuracy.
    """

    if dropout_images_prob is None:
        dropout_images_prob = [0.0, 0.0, 0.0, 0.0, 0.0]

    print(f"Restoring checkpoint: {checkpoint_path}")

    model = Tedd1104ModelPLForImageReordering.load_from_checkpoint(
        checkpoint_path=checkpoint_path
    )

    print("Done! Preparing to continue training...")

    data = Tedd1104ataModuleForImageReordering(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=batch_size,
        hide_map_prob=hide_map_prob,
        dropout_images_prob=dropout_images_prob,
        num_workers=dataloader_num_workers,
        token_mask_prob=mask_prob,
        transformer_nheads=None if model.encoder_type == "lstm" else model.nhead,
        sequence_length=model.sequence_size,
    )

    print(f"Restoring checkpoint: {checkpoint_path}. hparams: {hparams_path}")

    experiment_name = os.path.basename(
        output_dir if output_dir[-1] != "/" else output_dir[:-1]
    )
    if report_to == "tensorboard":
        logger = pl_loggers.TensorBoardLogger(
            save_dir=output_dir,
            name=experiment_name,
        )
    elif report_to == "wandb":
        logger = pl_loggers.WandbLogger(
            # name=experiment_name,
            # id=experiment_name,
            resume="allow",
            project="TEDD1104_reorder",
            save_dir=output_dir,
        )
    else:
        raise ValueError(
            f"Unknown logger: {report_to}. Please use 'tensorboard' or 'wandb'."
        )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="Validation/acc", mode="max", save_last=True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"

    model.accelerator = accelerator

    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        precision=precision if precision == "bf16" else int(precision),
        strategy=strategy,
        val_check_interval=val_check_interval,
        accumulate_grad_batches=accumulation_steps,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=os.path.join(output_dir, "trainer_checkpoint"),
        log_every_n_steps=10,
    )

    trainer.fit(
        ckpt_path=checkpoint_path,
        model=model,
        datamodule=data,
    )

    # print(f"Best model path: {checkpoint_callback.best_model_path}")

    if test_dir:
        trainer.test(datamodule=data, ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train_new",
        action="store_true",
        help="Train a new model",
    )

    group.add_argument(
        "--continue_training",
        action="store_true",
        help="Continues training a model from a checkpoint.",
    )

    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="The directory containing the training data.",
    )

    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="The directory containing the validation data.",
    )

    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="The directory containing the test data.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory to save the model to.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="The batch size for training and eval.",
    )

    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="The number of steps to accumulate gradients.",
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        required=True,
        help="The maximum number of epochs to train for.",
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of CPU workers for the Data Loaders",
    )

    parser.add_argument(
        "--hide_map_prob",
        type=float,
        default=1.0,
        help="Probability of hiding the minimap in the sequence (0<=hide_map_prob<=1)",
    )

    parser.add_argument(
        "--dropout_images_prob",
        type=float,
        nargs=5,
        default=[0.0, 0.0, 0.0, 0.0, 0.0],
        help="Probability of dropping each image in the sequence (0<=dropout_images_prob<=1)",
    )

    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help="The interval in epochs between validation checks.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="[NEW MODEL] The learning rate for the optimizer.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="[NEW MODEL]] AdamW Weight Decay",
    )

    parser.add_argument(
        "--cnn_model_name",
        type=str,
        default="efficientnet_b4",
        help="[NEW MODEL] CNN model name from torchvision models, see https://pytorch.org/vision/stable/models.html "
        "for a list of available models.",
    )

    parser.add_argument(
        "--do_not_load_pretrained_cnn",
        action="store_true",
        help="[NEW MODEL] Do not load the pretrained weights for the cnn model",
    )

    parser.add_argument(
        "--embedded_size",
        type=int,
        default=512,
        help="[NEW MODEL] The size of the embedding for the encoder.",
    )

    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="[NEW MODEL Transformers] Number of heads in the multi-head attention",
    )

    parser.add_argument(
        "--num_layers_encoder",
        type=int,
        default=4,
        help="[NEW MODEL] Number of transformer layers in the encoder",
    )

    parser.add_argument(
        "--dropout_cnn_out",
        type=float,
        default=0.3,
        help="[NEW MODEL] Dropout rate for the output of the CNN",
    )

    parser.add_argument(
        "--positional_embeddings_dropout",
        type=float,
        default=0.1,
        help="[NEW MODEL Transformer] Dropout rate for the positional embeddings",
    )

    parser.add_argument(
        "--dropout_encoder",
        type=float,
        default=0.1,
        help="[NEW MODEL] Dropout rate for the encoder",
    )

    parser.add_argument(
        "--dropout_encoder_features",
        type=float,
        default=0.3,
        help="[NEW MODEL] Dropout probability of the encoder output",
    )

    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.2,
        help="[NEW MODEL Transformers] Probability of masking each input vector in the transformer encoder",
    )

    parser.add_argument(
        "--sequence_size",
        type=int,
        default=5,
        help="[NEW MODEL] Length of the input sequence. Placeholder for the future, only 5 supported",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="If new_model is True, the path to the checkpoint to a pretrained model in the image reordering task. "
        "If continue_training is True, the path to the checkpoint to continue training from.",
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
        default="bf16",
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
        choices=["wandb", "tensorboard"],
        help="Report to wandb or tensorboard",
    )

    args = parser.parse_args()

    if args.train_new:
        train_new_model(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            cnn_model_name=args.cnn_model_name,
            accumulation_steps=args.accumulation_steps,
            hide_map_prob=args.hide_map_prob,
            dropout_images_prob=args.dropout_images_prob,
            val_check_interval=args.val_check_interval,
            dataloader_num_workers=args.dataloader_num_workers,
            pretrained_cnn=not args.do_not_load_pretrained_cnn,
            embedded_size=args.embedded_size,
            nhead=args.nhead,
            num_layers_encoder=args.num_layers_encoder,
            dropout_cnn_out=args.dropout_cnn_out,
            dropout_encoder_features=args.dropout_encoder_features,
            positional_embeddings_dropout=args.positional_embeddings_dropout,
            dropout_encoder=args.dropout_encoder,
            mask_prob=args.mask_prob,
            sequence_size=args.sequence_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            devices=args.devices,
            accelerator=args.accelerator,
            precision=args.precision,
            strategy=args.strategy,
            report_to=args.report_to,
        )

    else:
        continue_training(
            checkpoint_path=args.checkpoint_path,
            hparams_path=args.hparams_path,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            max_epochs=args.max_epochs,
            mask_prob=args.mask_prob,
            hide_map_prob=args.hide_map_prob,
            dropout_images_prob=args.dropout_images_prob,
            dataloader_num_workers=args.dataloader_num_workers,
            devices=args.devices,
            accelerator=args.accelerator,
            precision=args.precision,
            strategy=args.strategy,
            report_to=args.report_to,
        )
