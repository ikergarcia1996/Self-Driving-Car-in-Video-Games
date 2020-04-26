from torch.nn import CrossEntropyLoss
from utils import *
from model import TEDD1104, save_model, load_checkpoint, save_checkpoint
import torch
import torch.optim as optim
from typing import List
import time
import argparse
import random
import math
from torch.utils.tensorboard import SummaryWriter
from DataLoader import DataLoaderTEDD

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
    scheduler: torch.optim.lr_scheduler,
    train_dir: str,
    dev_dir: str,
    test_dir: str,
    output_dir: str,
    batch_size: int,
    accumulation_steps: int,
    initial_epoch: int,
    num_epoch: int,
    max_acc: float,
    hide_map_prob: float,
    dropout_images_prob: List[float],
    num_load_files_training: int,
    fp16: bool = True,
    amp_opt_level=None,
    save_checkpoints: bool = True,
    eval_every: int = 5,
    save_every: int = 20,
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
    - hide_map_prob: Probability for removing the minimap (put a black square)
       from a training example (0<=hide_map_prob<=1)
    - dropout_images_prob List of 5 floats or None, probability for removing each input image during training
     (black image) from a training example (0<=dropout_images_prob<=1)
    - fp16: Use FP16 for training
    - amp_opt_level: If FP16 training Nvidia apex opt level
    - save_checkpoints: save a checkpoint each epoch (Each checkpoint will rewrite the previous one)
    - save_best: save the model that achieves the higher accuracy in the development set

    Output:
     - float: Accuracy in the development test of the best model
    """
    writer: SummaryWriter = SummaryWriter()

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )

    criterion: CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    print("Loading dev set")
    X_dev, y_dev = load_dataset(dev_dir, fp=16 if fp16 else 32)
    X_dev = torch.from_numpy(X_dev)
    print("Loading test set")
    X_test, y_test = load_dataset(test_dir, fp=16 if fp16 else 32)
    X_test = torch.from_numpy(X_test)
    total_training_exampels: int = 0
    model.zero_grad()

    printTrace("Training...")
    for epoch in range(num_epoch):
        step_no: int = 0
        iteration_no: int = 0
        num_used_files: int = 0
        data_loader = DataLoaderTEDD(
            dataset_dir=train_dir,
            nfiles2load=num_load_files_training,
            hide_map_prob=hide_map_prob,
            dropout_images_prob=dropout_images_prob,
            fp=16 if fp16 else 32,
        )

        data = data_loader.get_next()
        # Get files in batches, all files will be loaded and data will be shuffled
        while data:
            X, y = data
            model.train()
            start_time: float = time.time()
            total_training_exampels += len(y)
            running_loss: float = 0.0
            num_batchs: int = 0
            acc_dev: float = 0.0

            for X_bacth, y_batch in nn_batchs(X, y, batch_size):
                X_bacth, y_batch = (
                    torch.from_numpy(X_bacth).to(device),
                    torch.from_numpy(y_batch).long().to(device),
                )

                outputs = model.forward(X_bacth)
                loss = criterion(outputs, y_batch) / accumulation_steps
                running_loss += loss.item()

                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if (step_no + 1) % accumulation_steps or (
                    num_used_files + 1 > len(data_loader) - num_load_files_training
                    and num_batchs == math.ceil(len(y) / batch_size) - 1
                ):  # If we are in the last bach of the epoch we also want to perform gradient descent
                    optimizer.step()
                    model.zero_grad()

                num_batchs += 1
                step_no += 1

            num_used_files += num_load_files_training

            # Print Statistics
            printTrace(
                f"EPOCH: {initial_epoch+epoch}. Iteration {iteration_no}. "
                f"{num_used_files} of {len(data_loader)} files. "
                f"Total examples used for training {total_training_exampels}. "
                f"Iteration time: {round(time.time() - start_time,2)} secs."
            )
            printTrace(
                f"Loss: {-1 if num_batchs == 0 else running_loss / num_batchs}. "
                f"Learning rate {optimizer.state_dict()['param_groups'][0]['lr']}"
            )
            writer.add_scalar("Loss/train", running_loss / num_batchs, iteration_no)

            scheduler.step(running_loss / num_batchs)

            if (iteration_no + 1) % eval_every == 0:
                start_time_eval: float = time.time()
                if len(X) > 0 and len(y) > 0:
                    acc_train: float = evaluate(
                        model=model,
                        X=torch.from_numpy(X),
                        golds=y,
                        device=device,
                        batch_size=batch_size,
                    )
                else:
                    acc_train = -1.0

                acc_dev: float = evaluate(
                    model=model,
                    X=X_dev,
                    golds=y_dev,
                    device=device,
                    batch_size=batch_size,
                )

                acc_test: float = evaluate(
                    model=model,
                    X=X_test,
                    golds=y_test,
                    device=device,
                    batch_size=batch_size,
                )

                printTrace(
                    f"Acc training set: {round(acc_train,2)}. "
                    f"Acc dev set: {round(acc_dev,2)}. "
                    f"Acc test set: {round(acc_test,2)}.  "
                    f"Eval time: {round(time.time() - start_time_eval,2)} secs."
                )

                if 0.0 < acc_dev > max_acc and save_best:
                    max_acc = acc_dev
                    printTrace(
                        f"New max acc in dev set {round(max_acc,2)}. Saving model..."
                    )
                    save_model(
                        model=model,
                        save_dir=output_dir,
                        fp16=fp16,
                        amp_opt_level=amp_opt_level,
                    )
                if acc_train > -1:
                    writer.add_scalar("Accuracy/train", acc_train, iteration_no)
                writer.add_scalar("Accuracy/dev", acc_dev, iteration_no)
                writer.add_scalar("Accuracy/test", acc_test, iteration_no)

            if save_checkpoints and (iteration_no + 1) % save_every == 0:
                printTrace("Saving checkpoint...")
                save_checkpoint(
                    path=os.path.join(output_dir, "checkpoint.pt"),
                    model=model,
                    optimizer_name=optimizer_name,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    acc_dev=acc_dev,
                    epoch=initial_epoch + epoch,
                    fp16=fp16,
                    opt_level=amp_opt_level,
                )

            iteration_no += 1
            data = data_loader.get_next()

        data_loader.close()

    return max_acc


def train_new_model(
    train_dir="Data\\GTAV-AI\\data-v2\\train\\",
    dev_dir="Data\\GTAV-AI\\data-v2\\dev\\",
    test_dir="Data\\GTAV-AI\\data-v2\\test\\",
    output_dir="Data\\models\\",
    batch_size=10,
    accumulation_steps: int = 1,
    num_epoch=20,
    optimizer_name="SGD",
    learning_rate: float = 0.01,
    scheduler_patience: int = 100,
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
    dropout_images_prob=None,
    num_load_files_training: int = 5,
    fp16=True,
    apex_opt_level="O2",
    save_checkpoints=True,
    eval_every: int = 5,
    save_every: int = 20,
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
    - hide_map_prob: Probability for removing the minimap (put a black square)
      from a training example (0<=hide_map_prob<=1)
    - dropout_images_prob List of 5 floats or None, probability for removing each input image during training
     (black image) from a training example (0<=dropout_images_prob<=1)
    - fp16: Use FP16 for training
    - amp_opt_level: If FP16 training Nvidia apex opt level
    - save_checkpoints: save a checkpoint each epoch (Each checkpoint will rewrite the previous one)
    - save_best: save the model that achieves the higher accuracy in the development set

    Output:

    """

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
        optimizer = optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True
        )
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        raise ValueError(
            f"Optimizer {optimizer_name} not implemented. Available optimizers: SGD, Adam"
        )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, patience=scheduler_patience, factor=0.5
    )

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )

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
        scheduler=scheduler,
        train_dir=train_dir,
        dev_dir=dev_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        initial_epoch=0,
        num_epoch=num_epoch,
        max_acc=0.0,
        hide_map_prob=hide_map_prob,
        dropout_images_prob=dropout_images_prob,
        num_load_files_training=num_load_files_training,
        fp16=fp16,
        amp_opt_level=apex_opt_level if fp16 else None,
        save_checkpoints=save_checkpoints,
        eval_every=eval_every,
        save_every=save_every,
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
    accumulation_steps: int = 1,
    num_epoch: int = 20,
    hide_map_prob: float = 0.0,
    dropout_images_prob: List[float] = None,
    num_load_files_training: int = 5,
    save_checkpoints=True,
    eval_every: int = 5,
    save_every: int = 100,
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
    - hide_map_prob: Probability for removing the minimap (put a black square)
      from a training example (0<=hide_map_prob<=1)
    -Probability for removing each input image during training (black image)
     from a training example (0<=dropout_images_prob<=1)
    - save_checkpoints: save a checkpoint each epoch (Each checkpoint will rewrite the previous one)
    - save_best: save the model that achieves the higher accuracy in the development set

    Output:

    """

    (
        model,
        optimizer_name,
        optimizer,
        scheduler,
        acc_dev,
        epoch,
        fp16,
        opt_level,
    ) = load_checkpoint(checkpoint_path, device)
    model = model.to(device)

    max_acc = train(
        model=model,
        optimizer_name=optimizer_name,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dir=train_dir,
        dev_dir=dev_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        initial_epoch=epoch,
        num_epoch=num_epoch,
        max_acc=acc_dev,
        hide_map_prob=hide_map_prob,
        dropout_images_prob=dropout_images_prob,
        num_load_files_training=num_load_files_training,
        fp16=fp16,
        amp_opt_level=opt_level if fp16 else None,
        save_checkpoints=save_checkpoints,
        eval_every=eval_every,
        save_every=save_every,
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient steps to accumulate. True batch size =  --batch_size * --accumulation_steps",
    )

    parser.add_argument(
        "--num_epochs", type=int, required=True, help="Number of epochs to perform",
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
        "--num_load_files_training",
        type=int,
        default=5,
        help="Number of dataset files to load each iteration for training. Files will be merged and shuffled. Loading"
        "more may be helpful for training, but it will use more RAM",
    )

    parser.add_argument(
        "--not_save_checkpoints",
        action="store_false",
        help="Do NOT save a checkpoint each epoch (Each checkpoint will rewrite the previous one)",
    )

    parser.add_argument(
        "--eval_every",
        type=int,
        default=5,
        help="Evaluate the model every --eval_every iterations (1 iteration = --num_load_files_training files used) ",
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=20,
        help="Save the model every --save_every iterations (1 iteration = --num_load_files_training files used) ",
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
        "--learning_rate",
        type=float,
        default=0.01,
        help="[new_model] Optimizer learning rate",
    )

    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=100,
        help="[new_model] Number of steps where the loss does not decrease until decrease the learning rate",
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
            accumulation_steps=args.gradient_accumulation_steps,
            num_epoch=args.num_epochs,
            hide_map_prob=args.hide_map_prob,
            dropout_images_prob=args.dropout_images_prob,
            num_load_files_training=args.num_load_files_training,
            optimizer_name=args.optimizer_name,
            learning_rate=args.learning_rate,
            scheduler_patience=args.scheduler_patience,
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
            eval_every=args.eval_every,
            save_every=args.save_every,
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
            accumulation_steps=args.gradient_accumulation_steps,
            hide_map_prob=args.hide_map_prob,
            dropout_images_prob=args.dropout_images_prob,
            num_load_files_training=args.num_load_files_training,
            save_checkpoints=args.not_save_checkpoints,
            eval_every=args.eval_every,
            save_every=args.save_every,
            save_best=args.not_save_best,
        )
