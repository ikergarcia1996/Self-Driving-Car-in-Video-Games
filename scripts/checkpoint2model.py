import sys

sys.path.append("..")
import model
import argparse
import os
from utils import print_message


def checkpoint2model(checkpoint_path: str, model_dir: str):
    """
    Given a checkpoint file, generates a model file that can be loaded by run_TEDD1104.py script.
    Input:
     - checkpoint_path path of checkpoint file (checkpoint.pt)
     - model_path directory where the model is going to be saved (model.bin and model_hyperparameters.json)
    Output:
     """

    if not os.path.exists(model_dir):
        print(f"{model_dir} does not exits. We will create it.")
        os.makedirs(model_dir)

    print_message(f"Loading checkpoint: {checkpoint_path}")

    (
        tedd1104_model,
        _,
        _,
        _,
        running_loss,
        total_batches,
        total_training_examples,
        acc_dev,
        epoch,
        fp16,
        _,
    ) = model.load_checkpoint(path=checkpoint_path, device=model.torch.device("cpu"))

    print(
        f">>>>>> Checkpoint info <<<<<<\n"
        f"Running loss: {running_loss/total_batches}\n"
        f"Num epochs: {epoch+1}\n"
        f"Total training examples: {total_training_examples}\n"
        f"Acc dev set: {round(acc_dev*100,2)}\n"
        f"FP16: {fp16}\n"
    )

    print_message(f"Saving model in {model_dir}")

    model.save_model(model=tedd1104_model, save_dir=model_dir, fp16=fp16)

    print_message(f"Done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path of checkpoint file (checkpoint.pt)",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory where the model is going to be saved (model.bin and model_hyperparameters.json)",
    )

    args = parser.parse_args()

    checkpoint2model(checkpoint_path=args.checkpoint_path, model_dir=args.model_dir)
