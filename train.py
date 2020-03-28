import os

from torch.nn import CrossEntropyLoss
from utils import *
from model import TEDD1104, save_model, load_checkpoint, save_checkpoint
import torch
import torch.optim as optim
from typing import List
import time
import argparse

if torch.cuda.is_available():
    device: torch.device = torch.device("cuda:0")
else:
    device: torch.device = torch.device("cpu")
    logging.warning(
        "GPU not found, using CPU, training will be very slow. CPU NOT COMPATIBLE WITH FP16"
    )


def train(
    model: TEDD1104,
    optimizer_name: str,
    optimizer: torch.optim,
    train_dir: str,
    dev_dir: str,
    test_dir: str,
    output_dir: str,
    batch_size: int,
    initial_epoch: int,
    num_epoch: int,
    max_acc: float,
    hide_map_prob: float,
    fp16: bool = True,
    amp_opt_level=None,
    save_checkpoints: bool = True,
    save_best: bool = True,
):

    """
    Train a model

    Input:
    - model: TEDD1104 model to train
    - optimizer_name: Name of the optimizer to use [SGD, Adam]
    - optimizer: Optimizer (torch.optim)
    - train_dir: Directory where the train files are stored
    - dev_dir: Directory where the development files are stored
    - test_dir: Directory where the test files are stored
    - output_dir: Directory where the model and the checkpoints are going to be saved
    - batch_size: Batch size (Around 10 for 8GB GPU)
    - initial_epoch: Number of previous epochs used to train the model (0 unless the model has been
      restored from checkpoint)
    - num_epochs: Number of epochs to do
    - max_acc: Accuracy in the development set (0 unless the model has been
      restored from checkpoint)
    - hide_map_prob: Probability for removing the minimap (black square) from the image (0<=hide_map_prob<=1)
    - fp16: Use FP16 for training
    - amp_opt_level: If FP16 training Nvidia apex opt level
    - save_checkpoints: save a checkpoint each epoch (Each checkpoint will rewrite the previous one)
    - save_best: save the model that achieves the higher accuracy in the development set

    Output:
     - float: Accuracy in the development test of the best model
    """

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )

    criterion: CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    X_dev, y_dev = load_dataset(dev_dir, fp=16 if fp16 else 31)
    X_dev = torch.from_numpy(X_dev)

    X_test, y_test = load_dataset(test_dir, fp=16 if fp16 else 21)
    X_test = torch.from_numpy(X_test)

    acc_dev: float = 0.0

    printTrace("Training...")
    for epoch in range(num_epoch):

        for file_t in glob.glob(os.path.join(train_dir, "*.npz")):
            model.train()
            start_time: float = time.time()

            X, y = load_file(
                path=file_t, fp=16 if fp16 else 32, hide_map_prob=hide_map_prob
            )
            running_loss = 0.0
            num_batchs = 0

            for X_bacth, y_batch in nn_batchs(X, y, batch_size):
                X_bacth, y_batch = (
                    torch.from_numpy(X_bacth).to(device),
                    torch.from_numpy(y_batch).long().to(device),
                )
                optimizer.zero_grad()
                outputs = model.forward(X_bacth)
                loss = criterion(outputs, y_batch)
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                running_loss += loss.item()
                num_batchs += 1

            # Print Statistics
            acc_train = evaluate(
                model=model,
                X=torch.from_numpy(X),
                golds=y,
                device=device,
                batch_size=batch_size,
            )

            acc_dev = evaluate(
                model=model, X=X_dev, golds=y_dev, device=device, batch_size=batch_size,
            )

            acc_test = evaluate(
                model=model,
                X=X_test,
                golds=y_test,
                device=device,
                batch_size=batch_size,
            )

            printTrace(
                f"EPOCH: {initial_epoch+epoch}. Current File {file_t}. Training time: {time.time() - start_time} secs"
            )

            printTrace(
                f"Loss: {running_loss / num_batchs}. Acc training set: {acc_train}. "
                f"Acc dev set: {acc_dev}. Acc test set: {acc_test}"
            )

            if acc_dev > max_acc and save_best:
                max_acc = acc_dev
                printTrace(f"New max acc in dev set {max_acc}. Saving model...")
                save_model(
                    model=model,
                    save_dir=output_dir,
                    fp16=fp16,
                    amp_opt_level=amp_opt_level,
                )

        if save_checkpoints:
            printTrace("Saving checkpoint...")
            save_checkpoint(
                path=os.path.join(output_dir, "checkpoint.pt"),
                model=model,
                optimizer_name=optimizer_name,
                optimizer=optimizer,
                acc_dev=acc_dev,
                epoch=initial_epoch + epoch,
                fp16=fp16,
                opt_level=amp_opt_level,
            )

    return max_acc


def train_new_model(
    train_dir="Data\\GTAV-AI\\data-v2\\train\\",
    dev_dir="Data\\GTAV-AI\\data-v2\\dev\\",
    test_dir="Data\\GTAV-AI\\data-v2\\test\\",
    output_dir="Data\\models\\",
    batch_size=10,
    num_epoch=20,
    optimizer_name="SGD",
    resnet: int = 18,
    pretrained_resnet: bool = True,
    sequence_size: int = 5,
    embedded_size: int = 256,
    hidden_size: int = 128,
    num_layers_lstm: int = 1,
    bidirectional_lstm: bool = False,
    layers_out: List[int] = None,
    dropout_cnn: float = 0.1,
    dropout_cnn_out: float = 0.1,
    dropout_lstm: float = 0.1,
    dropout_lstm_out: float = 0.1,
    hide_map_prob: float = 0.0,
    fp16=True,
    apex_opt_level="O2",
    save_checkpoints=True,
    save_best=True,
):

    """
    Train a new model

    Input:
    - train_dir: Directory where the train files are stored
    - dev_dir: Directory where the development files are stored
    - test_dir: Directory where the test files are stored
    - output_dir: Directory where the model and the checkpoints are going to be saved
    - batch_size: Batch size (Around 10 for 8GB GPU)
    - num_epochs: Number of epochs to do
    - optimizer_name: Name of the optimizer to use [SGD, Adam]
    - optimizer: Optimizer (torch.optim)
    - resnet: resnet module to use [18,34,50,101,152]
    - pretrained_resnet: Load pretrained resnet weights
    - sequence_size: Length of each series of features
    - embedded_size: Size of the feature vectors
    - hidden_size: LSTM hidden size
    - num_layers_lstm: number of layers in the LSTM
    - bidirectional_lstm: forward or bidirectional LSTM
    - layers_out: list of integer, for each integer i a linear layer with i neurons will be added.
    - dropout_cnn: dropout probability for the CNN layers
    - dropout_cnn_out: dropout probability for the cnn features (output layer)
    - dropout_lstm: dropout probability for the LSTM
    - dropout_lstm_out: dropout probability for the LSTM features (output layer)
    - hide_map_prob: Probability for removing the minimap (black square) from the image (0<=hide_map_prob<=1)
    - fp16: Use FP16 for training
    - amp_opt_level: If FP16 training Nvidia apex opt level
    - save_checkpoints: save a checkpoint each epoch (Each checkpoint will rewrite the previous one)
    - save_best: save the model that achieves the higher accuracy in the development set

    Output:

    """

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )

    print("Loading new model")
    model: TEDD1104 = TEDD1104(
        resnet=resnet,
        pretrained_resnet=pretrained_resnet,
        sequence_size=sequence_size,
        embedded_size=embedded_size,
        hidden_size=hidden_size,
        num_layers_lstm=num_layers_lstm,
        bidirectional_lstm=bidirectional_lstm,
        layers_out=layers_out,
        dropout_cnn=dropout_cnn,
        dropout_cnn_out=dropout_cnn_out,
        dropout_lstm=dropout_lstm,
        dropout_lstm_out=dropout_lstm_out,
    ).to(device)

    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        raise ValueError(
            f"Optimizer {optimizer_name} not implemented. Available optimizers: SGD, Adam"
        )

    if fp16:
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level=apex_opt_level,
            keep_batchnorm_fp32=True,
            loss_scale="dynamic",
        )

    max_acc = train(
        model=model,
        optimizer_name=optimizer_name,
        optimizer=optimizer,
        train_dir=train_dir,
        dev_dir=dev_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        initial_epoch=0,
        num_epoch=num_epoch,
        max_acc=0.0,
        hide_map_prob=hide_map_prob,
        fp16=fp16,
        amp_opt_level=apex_opt_level if fp16 else None,
        save_checkpoints=save_checkpoints,
        save_best=save_best,
    )

    print(f"Training finished, max accuracy in the development set {max_acc}")


def continue_training(
    checkpoint_path: str,
    train_dir: str = "Data\\GTAV-AI\\data-v2\\train\\",
    dev_dir: str = "Data\\GTAV-AI\\data-v2\\dev\\",
    test_dir: str = "Data\\GTAV-AI\\data-v2\\test\\",
    output_dir: str = "Data\\models\\",
    batch_size: int = 10,
    num_epoch: int = 20,
    hide_map_prob: float = 0.0,
    save_checkpoints=True,
    save_best=True,
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
    - hide_map_prob: Probability for removing the minimap (black square) from the image (0<=hide_map_prob<=1)
    - save_checkpoints: save a checkpoint each epoch (Each checkpoint will rewrite the previous one)
    - save_best: save the model that achieves the higher accuracy in the development set

    Output:

    """

    model, optimizer_name, optimizer, acc_dev, epoch, fp16, opt_level = load_checkpoint(
        checkpoint_path, device
    )
    model = model.to(device)

    max_acc = train(
        model=model,
        optimizer_name=optimizer_name,
        optimizer=optimizer,
        train_dir=train_dir,
        dev_dir=dev_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        initial_epoch=epoch,
        num_epoch=num_epoch,
        max_acc=acc_dev,
        hide_map_prob=hide_map_prob,
        fp16=fp16,
        amp_opt_level=opt_level if fp16 else None,
        save_checkpoints=save_checkpoints,
        save_best=save_best,
    )

    print(f"Training finished, max accuracy in the development set {max_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train_new", action="store_true", help="Train a new model",
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
        "--dev_dir",
        type=str,
        required=True,
        help="Directory containing the development files",
    )

    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Directory containing the test files",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the model and checkpoints are going to be saved",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="batch size for training (10 for a 8GB GPU seems fine)",
    )

    parser.add_argument(
        "--num_epochs", type=int, required=True, help="Number of epochs to perform",
    )

    parser.add_argument(
        "--hide_map_prob",
        type=float,
        default=0.0,
        help="Probability for removing the minimap (black square) from the image (0<=hide_map_prob<=1)",
    )

    parser.add_argument(
        "--not_save_checkpoints",
        action="store_false",
        help="Do NOT save a checkpoint each epoch (Each checkpoint will rewrite the previous one)",
    )

    parser.add_argument(
        "--not_save_best",
        action="store_false",
        help="Dot NOT save the best model in the development set",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="[new_model] Use FP16 floating point precision: "
        "Requires Nvidia Apex: https://www.github.com/nvidia/apex "
        "and a modern Nvidia GPU FP16 capable (Volta, Turing and future architectures)."
        "If you restore a checkpoint the original FP configuration of the model will be restored.",
    )

    parser.add_argument(
        "--amp_opt_level",
        type=str,
        default="O2",
        help="[new_model] If FP16 training, the Apex OPT level",
    )

    parser.add_argument(
        "--optimizer_name",
        type=str,
        default="SGD",
        choices=["SGD", "Adam"],
        help="[new_model] Optimizer to use for training a new model: SGD or Adam",
    )

    parser.add_argument(
        "--resnet",
        type=int,
        default=18,
        choices=[18, 34, 50, 101, 152],
        help="[new_model] Which of the resnet model availabel in torchvision.models use. Availabel model:"
        "18, 34, 50, 101 and 152.",
    )

    parser.add_argument(
        "--do_not_load_pretrained_resnet",
        action="store_false",
        help="[new_model] Do not load the pretrained weights for the resnet model",
    )

    parser.add_argument(
        "--sequence_size",
        type=int,
        default=5,
        help="[new_model] Number of images to use to decide witch key press. Note: Only 5 supported for right now",
    )

    parser.add_argument(
        "--embedded_size",
        type=int,
        default=256,
        help="[new_model] Size of the feature vectors (CNN encoder output size)",
    )

    parser.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden size")

    parser.add_argument(
        "--num_layers_lstm",
        type=int,
        default=1,
        help="[new_model] number of layers in the LSTM",
    )

    parser.add_argument(
        "--bidirectional_lstm",
        action="store_true",
        help="[new_model] Use a bidirectional LSTM instead of a forward LSTM",
    )

    parser.add_argument(
        "--layers_out",
        nargs="+",
        type=int,
        required=False,
        help="[new_model] list of integer, for each integer i a linear layer with i neurons will be added to the "
        " output, if none layers are provided the ouput layer will be just a linear layer with input size hidden_size "
        "and output size 9. Note: The input size of the first layer and last layer will automatically be added "
        "regardless of the user input, so you don't need to care about the size of these layers. ",
    )

    parser.add_argument(
        "--dropout_cnn",
        type=float,
        default=0.1,
        help="[new_model] Dropout of the CNN layers between 0.0 and 1.0",
    )

    parser.add_argument(
        "--dropout_cnn_out",
        type=float,
        default=0.1,
        help="[new_model] Dropout of the CNN representations (output layer) between 0.0 and 1.0",
    )

    parser.add_argument(
        "--dropout_lstm",
        type=float,
        default=0.1,
        help="[new_model] Dropout of the LSTM layer between 0.0 and 1.0",
    )

    parser.add_argument(
        "--dropout_lstm_out",
        type=float,
        default=0.1,
        help="[new_model] Dropout of the LSTM representations (output layer) between 0.0 and 1.0",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="[continue_training] Path of the checkpoint to load for continue training it",
    )

    args = parser.parse_args()

    if args.train_new:
        train_new_model(
            train_dir=args.train_dir,
            dev_dir=args.dev_dir,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_epoch=args.num_epochs,
            hide_map_prob=args.hide_map_prob,
            optimizer_name=args.optimizer_name,
            resnet=args.resnet,
            pretrained_resnet=args.do_not_load_pretrained_resnet,
            sequence_size=args.sequence_size,
            embedded_size=args.embedded_size,
            hidden_size=args.hidden_size,
            num_layers_lstm=args.num_layers_lstm,
            bidirectional_lstm=args.bidirectional_lstm,
            layers_out=args.layers_out,
            dropout_cnn=args.dropout_cnn,
            dropout_cnn_out=args.dropout_cnn_out,
            dropout_lstm=args.dropout_lstm,
            dropout_lstm_out=args.dropout_lstm_out,
            fp16=args.fp16,
            apex_opt_level=args.amp_opt_level,
            save_checkpoints=args.not_save_checkpoints,
            save_best=args.not_save_best,
        )

    else:
        continue_training(
            checkpoint_path=args.checkpoint_path,
            train_dir=args.train_dir,
            dev_dir=args.dev_dir,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            hide_map_prob=args.hide_map_prob,
            save_checkpoints=args.not_save_checkpoints,
            save_best=args.not_save_best,
        )
