from torch.optim import AdamW
from fairseq.optim.adafactor import Adafactor


def get_adamw(parameters, learning_rate: float, weight_decay: float):
    """
    Get AdamW optimizer.
    :param Iterable parameters: The parameters to optimize.
    :param float learning_rate: The learning rate.
    :param float weight_decay: The weight decay.
    :return: torch.optim.AdamW. The AdamW optimizer.
    """
    optimizer = AdamW(
        params=parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-4,
    )
    return optimizer


def get_adafactor(parameters, learning_rate, weight_decay):
    """
    Get Adafactor optimizer (Requires fairseq, pip install fairseq).
    :param Iterable parameters: The parameters to optimize.
    :param float learning_rate: The learning rate.
    :param float weight_decay: The weight decay.
    :return:  fairseq.optim.adafactor.Adafactor. The Adafactor optimizer.
    """
    optimizer = Adafactor(
        params=parameters,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=learning_rate,
        clip_threshold=1.0,
        weight_decay=weight_decay,
    )

    return optimizer
