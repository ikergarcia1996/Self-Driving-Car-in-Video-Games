from model import Tedd1104ModelPL
from typing import List
import argparse
from dataset import Tedd1104ataModule
import os
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl


def train(
    model: Tedd1104ModelPL,
    train_dir: str,
    val_dir: str,
    output_dir: str,
    batch_size: int,
    accumulation_steps: int,
    max_epochs: int,
    hide_map_prob: float,
    dropout_images_prob: List[float],
    test_dir: str = None,
    control_mode: str = "keyboard",
    val_check_interval: float = 0.25,
    dataloader_num_workers=os.cpu_count(),
):

    """
    Train a model

    Input:
    - model: TEDD1104 model to train
    - train_dir: Directory where the train files are stored
    - dev_dir: Directory where the development files are stored
    - test_dir: Directory where the test files are stored
    - output_dir: Directory where the model and the checkpoints are going to be saved
    - batch_size: Batch size (Around 32-64 for 24GB GPU)
    - accumulation_steps: Number of accumulation steps (training batch size = accumulation_steps * batch_size)
    - max_epochs: Number of epochs to do
    - hide_map_prob: Probability for removing the minimap (put a black square)
       from a training example (0<=hide_map_prob<=1)
    - dropout_images_prob List of 5 floats or None, probability for removing each input image during training
     (black image) from a training example (0<=dropout_images_prob<=1)
    - variable_weights: List of 3 floats, weights for each output variable [LX, LT, RT]
      for the weighted mean squared error
    - fp16: Use FP16 for training
    - control_mode: Set if the dataset true values will be keyboard inputs (9 classes)
      or Controller Inputs (2 continuous values)
    -val_check_interval: Validate model every val_check_interval of epoch 0<val_check_interval<=1

    """

    if not os.path.exists(output_dir):
        print(f"{output_dir} does not exits. We will create it.")
        os.makedirs(output_dir)

    data = Tedd1104ataModule(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=batch_size,
        hide_map_prob=hide_map_prob,
        dropout_images_prob=dropout_images_prob,
        control_mode=control_mode,
        num_workers=dataloader_num_workers,
    )

    tb_logger = pl_loggers.TensorBoardLogger(output_dir)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="Val/acc_k@1", mode="max", save_last=True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"

    trainer = pl.Trainer(
        precision=16,
        gpus=1,
        val_check_interval=val_check_interval,
        accumulate_grad_batches=accumulation_steps,
        max_epochs=max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        # accelerator="ddp",
        default_root_dir=os.path.join(output_dir, "trainer_checkpoint"),
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
    accumulation_steps: int = 1,
    hide_map_prob: float = 0.0,
    test_dir: str = None,
    dropout_images_prob=None,
    variable_weights: List[float] = None,
    control_mode: str = "keyboard",
    val_check_interval: float = 0.25,
    dataloader_num_workers=os.cpu_count(),
    pretrained_cnn: bool = True,
    embedded_size: int = 512,
    nhead: int = 8,
    num_layers_encoder: int = 1,
    lstm_hidden_size: int = 512,
    dropout_cnn_out: float = 0.1,
    positional_embeddings_dropout: float = 0.1,
    dropout_encoder: float = 0.1,
    dropout_encoder_features: float = 0.8,
    mask_prob: float = 0.0,
    sequence_size: int = 5,
    encoder_type: str = "transformer",
    bidirectional_lstm=True,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-3,
    checkpoint_path: str = None,
):

    """
    Train a new model

    Input LSTM:
    - train_dir: Directory where the train files are stored
    - val_dir: Directory where the development files are stored
    - test_dir: Directory where the test files are stored
    - output_dir: Directory where the model and the checkpoints are going to be saved
    - encoder_type: Use a transformer or a lstm encoder
    - batch_size: Batch size (Around 10 for RTX 2080 with 8GB , 32 for a RTX 3090 with 24GB)
    - num_epochs: Number of epochs to do
    - optimizer_name: Name of the optimizer to use [SGD, Adam]
    - optimizer: Optimizer (torch.optim)
    - resnet: resnet module to use [18,34,50,101,152]
    - pretrained_resnet: Load pretrained resnet weights
    - sequence_size: Length of each series of features
    - embedded_size: Size of the feature vectors
    - hidden_size: LSTM hidden size
    - nhead: number of heads for the transformer layer
    - num_layers: number of transformer layers in the LSTM o r Transformer encoder
    - bidirectional_lstm: forward or bidirectional LSTM
    - layers_out: list of integer, for each integer i a linear layer with i neurons will be added.
    - dropout_cnn: dropout probability for the CNN layers
    - dropout_cnn_out: dropout probability for the cnn features (output layer)
    - positional_embeddings_dropout: dropout probability for the transformer input embeddings
    - dropout_lstm: dropout probability for the LSTM
    - dropout_encoder_out: dropout probability for the LSTM or transformer features (output layer)
    - hide_map_prob: Probability for removing the minimap (put a black square)
      from a training example (0<=hide_map_prob<=1)
    - dropout_images_prob List of 5 floats or None, probability for removing each input image during training
      (black image) from a training example (0<=dropout_images_prob<=1)
    - fp16: Use FP16 for training
    - save_checkpoints: save a checkpoint each epoch (Each checkpoint will rewrite the previous one)
    - save_best: save the model that achieves the lowest loss in the development set
    - keyboard_input: Set this flag if dataset uses keyboard input (V2 dataset), the keys will be converted to
     controller input
    - log_interval: Save running loss in tensorboard every log_interval iterations
    - log_dir: Tensorboard logging directory
    Output:

    """

    assert control_mode.lower() in [
        "keyboard",
        "controller",
    ], f"{control_mode.lower()} control mode not supported. Supported dataset types: [keyboard, controller].  "

    if dropout_images_prob is None:
        dropout_images_prob = [0.0, 0.0, 0.0, 0.0, 0.0]

    if not checkpoint_path:
        model: Tedd1104ModelPL = Tedd1104ModelPL(
            cnn_model_name=cnn_model_name,
            pretrained_cnn=pretrained_cnn,
            embedded_size=embedded_size,
            nhead=nhead,
            num_layers_encoder=num_layers_encoder,
            lstm_hidden_size=lstm_hidden_size,
            dropout_cnn_out=dropout_cnn_out,
            positional_embeddings_dropout=positional_embeddings_dropout,
            dropout_encoder=dropout_encoder,
            dropout_encoder_features=dropout_encoder_features,
            mask_prob=mask_prob,
            control_mode=control_mode,
            sequence_size=sequence_size,
            encoder_type=encoder_type,
            bidirectional_lstm=bidirectional_lstm,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weights=variable_weights,
        )

    else:

        print(f"Restoring model from {checkpoint_path}.")
        model = Tedd1104ModelPL.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            dropout_cnn_out=dropout_cnn_out,
            positional_embeddings_dropout=positional_embeddings_dropout,
            dropout_encoder=dropout_encoder,
            dropout_encoder_features=dropout_encoder_features,
            mask_prob=mask_prob,
            control_mode=control_mode,
            strict=False,
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
        hide_map_prob=hide_map_prob,
        dropout_images_prob=dropout_images_prob,
        control_mode=control_mode,
        val_check_interval=val_check_interval,
        dataloader_num_workers=dataloader_num_workers,
    )


def continue_training(
    checkpoint_path: str,
    train_dir: str,
    val_dir: str,
    batch_size: int,
    max_epochs: int,
    output_dir,
    accumulation_steps,
    test_dir: str = None,
    hparams_path: str = None,
    hide_map_prob: float = 0.0,
    dropout_images_prob=None,
    dataloader_num_workers=os.cpu_count(),
    val_check_interval: float = 0.25,
):

    """
    Load a checkpoint and continue training, we will restore the model, the optimizer and the nvidia apex data if
    the model was trained using fp16. Note: If the model was trained using fp16 it cannot be restored as an fp32
    model and vice versa. The floating point precision used for training the model will be restored automatically
    from the checkpoint.

    Input:
    - checkpoint_path: Path of the checkpoint to restore
    - train_dir: Directory where the train files are stored
    - dev_dir: Directory where the development files are stored
    - test_dir: Directory where the test files are stored
    - output_dir: Directory where the model and the checkpoints are going to be saved
    - batch_size: Batch size (Around 10 for 8GB GPU)
    - num_epochs: Number of epochs to do
    - optimizer_name: Name of the optimizer to use [SGD, Adam]
    - hide_map_prob: Probability for removing the minimap (put a black square)
      from a training example (0<=hide_map_prob<=1)
    - Probability for removing each input image during training (black image)
      from a training example (0<=dropout_images_prob<=1)
    - save_checkpoints: save a checkpoint each epoch (Each checkpoint will rewrite the previous one)
    - save_best: save the model that achieves the lowest loss in the development set
    - keyboard_input: Set this flag if dataset uses keyboard input (V2 dataset), the keys will be converted to
      controller input
    - log_interval: Save running loss in tensorboard every log_interval iterations
    - log_dir: Tensorboard logging directory
    Output:

    """

    if dropout_images_prob is None:
        dropout_images_prob = [0.0, 0.0, 0.0, 0.0, 0.0]

    """
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
    """
    model = Tedd1104ModelPL.load_from_checkpoint(
        checkpoint_path=checkpoint_path
    )  # "hparams_file=hparams_path", strict=True

    data = Tedd1104ataModule(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=batch_size,
        hide_map_prob=hide_map_prob,
        dropout_images_prob=dropout_images_prob,
        control_mode=model.control_mode,
        num_workers=dataloader_num_workers,
    )

    print(f"Restoring checkpoint: {checkpoint_path}. hparams: {hparams_path}")

    tb_logger = pl_loggers.TensorBoardLogger(output_dir)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="Val/acc_k@1", mode="max", save_last=True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"

    trainer = pl.Trainer(
        resume_from_checkpoint=checkpoint_path,
        precision=16,
        gpus=1,
        val_check_interval=val_check_interval,
        accumulate_grad_batches=accumulation_steps,
        max_epochs=max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        # accelerator="ddp",
        default_root_dir=os.path.join(output_dir, "trainer_checkpoint"),
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=data)

    print(f"Best model path: {checkpoint_callback.best_model_path}")

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
        help="Restore a checkpoint and continue training",
    )

    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Directory containing the train files",
    )

    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="Directory containing the development files",
    )

    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Directory containing the test files",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the model and checkpoints are going to be saved",
    )

    parser.add_argument(
        "--encoder_type",
        type=str,
        choices=["lstm", "transformer"],
        default="transformer",
        help="Type of encoder to use, lstm or transformer",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="batch size for training (10 for a 8GB GPU seems fine)",
    )

    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient steps to accumulate. True batch size =  --batch_size * --accumulation_steps",
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        required=True,
        help="Number of epochs to perform",
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
        default=0.0,
        help="Probability for removing the minimap (put a black square) from a training example (0<=hide_map_prob<=1)",
    )

    parser.add_argument(
        "--dropout_images_prob",
        type=float,
        nargs=5,
        default=[0.0, 0.0, 0.0, 0.0, 0.0],
        help="List of 5 floats. Probability for removing each input image during training (black image) "
        "from a training example (0<=dropout_images_prob<=1) ",
    )

    parser.add_argument(
        "--variable_weights",
        type=float,
        nargs="+",
        default=None,
        help="List of 3 floats, weights for each output variable [LX, LT, RT]",
    )

    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=0.25,
        help="Evaluate model every val_check_interval of epoch",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="[new_model] Optimizer learning rate",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="[new_model] AdamW Weight Decay",
    )

    parser.add_argument(
        "--cnn_model_name",
        type=str,
        default="resnet152",
        help="[new_model] CNN model name from torchvision models, see https://pytorch.org/vision/stable/models.html "
        "for a list of available models.",
    )

    parser.add_argument(
        "--do_not_load_pretrained_cnn",
        action="store_true",
        help="[new_model] Do not load the pretrained weights for the cnn model",
    )

    parser.add_argument(
        "--embedded_size",
        type=int,
        default=512,
        help="[new_model] Size of the feature vectors (CNN encoder output size)",
    )

    parser.add_argument(
        "--lstm_hidden_size",
        type=int,
        default=512,
        help="[new_model LSTM] LSTM hidden size",
    )

    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="[new_model Transformer] number of heads for the transformer layer",
    )

    parser.add_argument(
        "--num_layers_encoder",
        type=int,
        default=1,
        help="[new_model] number of layers in the LSTM or the Transformer",
    )

    parser.add_argument(
        "--bidirectional_lstm",
        action="store_true",
        help="[new_model LSTM] Use a bidirectional LSTM instead of a forward LSTM",
    )

    parser.add_argument(
        "--dropout_cnn_out",
        type=float,
        default=0.1,
        help="[new_model] Dropout of the CNN representations (output layer) between 0.0 and 1.0",
    )

    parser.add_argument(
        "--positional_embeddings_dropout",
        type=float,
        default=0.1,
        help="[new_model transformer] dropout probability for the transformer input embeddings between 0.0 and 1.0",
    )

    parser.add_argument(
        "--dropout_encoder",
        type=float,
        default=0.1,
        help="[new_model] Dropout of the encoder layer between 0.0 and 1.0",
    )

    parser.add_argument(
        "--dropout_encoder_features",
        type=float,
        default=0.2,
        help="[new_model] Dropout of the encoder output between 0.0 and 1.0",
    )

    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.0,
        help="[new_model] Probability of marking each feature of the transformer encoder",
    )

    parser.add_argument(
        "--sequence_size",
        type=int,
        default=5,
        help="[new_model] Place holder for future releases, sequence size is always 5 in the current model",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path of the checkpoint to load for continue training it (Could be a TEDD model or a pretrained "
        "TEEDforImageReordering model",
    )

    parser.add_argument(
        "--control_mode",
        type=str,
        choices=["keyboard", "controller"],
        help="Set if the dataset true values will be keyboard inputs (9 classes) "
        "or Controller Inputs (2 continuous values)",
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
            variable_weights=args.variable_weights,
            control_mode=args.control_mode,
            val_check_interval=args.val_check_interval,
            dataloader_num_workers=args.dataloader_num_workers,
            pretrained_cnn=not args.do_not_load_pretrained_cnn,
            embedded_size=args.embedded_size,
            nhead=args.nhead,
            num_layers_encoder=args.num_layers_encoder,
            lstm_hidden_size=args.lstm_hidden_size,
            dropout_cnn_out=args.dropout_cnn_out,
            dropout_encoder_features=args.dropout_encoder_features,
            positional_embeddings_dropout=args.positional_embeddings_dropout,
            dropout_encoder=args.dropout_encoder,
            mask_prob=args.mask_prob,
            sequence_size=args.sequence_size,
            encoder_type=args.encoder_type,
            bidirectional_lstm=args.bidirectional_lstm,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            checkpoint_path=args.checkpoint_path,
        )

    else:
        continue_training(
            checkpoint_path=args.checkpoint_path,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            max_epochs=args.max_epochs,
            hide_map_prob=args.hide_map_prob,
            dropout_images_prob=args.dropout_images_prob,
            dataloader_num_workers=args.dataloader_num_workers,
        )
