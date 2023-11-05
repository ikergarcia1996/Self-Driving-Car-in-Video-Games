from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The local path or huggingface hub name of the model and tokenizer to use."
        },
    )

    quantization_inference: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Whether to use '4' or '8' bit quantization for inference. Requires bitsandbytes library:"
                " https://github.com/TimDettmers/bitsandbytes. This parameter is only used for training."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task: str = field(
        default="video-classification",
        metadata={
            "help": "The name of the task to train on: video-classification, video-masking"
        },
    )

    train_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory where the training data is stored."},
    )

    validation_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory where the validation data is stored."},
    )

    test_dirs: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The directories where the test data is stored."},
    )

    hide_map_probability_train: Optional[float] = field(
        default=0.0,
        metadata={"help": "The probability of hiding the map in the training data."},
    )

    hide_map_probability_validation: Optional[float] = field(
        default=0.0,
        metadata={"help": "The probability of hiding the map in the validation data."},
    )

    hide_map_probability_test: Optional[float] = field(
        default=0.0,
        metadata={"help": "The probability of hiding the map in the test data."},
    )

    mask_ratio: Optional[float] = field(
        default=0.9,
        metadata={"help": "The ratio of the image to be masked."},
    )
