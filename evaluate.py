import torchmetrics
from transformers import EvalPrediction
import torch


def compute_metrics(eval_preds: EvalPrediction):
    logits = torch.from_numpy(eval_preds.predictions)
    labels = torch.from_numpy(eval_preds.label_ids)

    acc_k1_macro = torchmetrics.functional.accuracy(
        logits, labels, num_classes=9, average="macro", top_k=1, task="multiclass"
    ).item()
    acc_k1_micro = torchmetrics.functional.accuracy(
        logits, labels, num_classes=9, average="micro", top_k=1, task="multiclass"
    ).item()
    acc_k3_macro = torchmetrics.functional.accuracy(
        logits, labels, num_classes=9, average="macro", top_k=3, task="multiclass"
    ).item()
    acc_k3_micro = torchmetrics.functional.accuracy(
        logits, labels, num_classes=9, average="micro", top_k=3, task="multiclass"
    ).item()

    return {
        "acc_k1_macro": acc_k1_macro,
        "acc_k1_micro": acc_k1_micro,
        "acc_k3_macro": acc_k3_macro,
        "acc_k3_micro": acc_k3_micro,
    }
