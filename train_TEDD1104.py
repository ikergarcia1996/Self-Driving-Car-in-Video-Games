import gc
import os.path

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)

from modeling_videomae import VideoMAEForPreTraining, VideoMAEForVideoClassification

import torch
import logging
from config import ModelArguments, DataTrainingArguments
from utils import get_trainable_parameters, init_logger
from dataset import Tedd1104Dataset

import sys
import json
from evaluate import compute_metrics

from functools import partial


def clean_cache():
    """Clean cache to avoid memory leak.
    This fixes this issue: https://github.com/huggingface/transformers/issues/22801"""

    logging.info(
        f"Cleaning GPU memory. Current memory usage: {torch.cuda.memory_allocated()}"
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info(
            f"GPU memory usage after cleaning: {torch.cuda.memory_allocated()}"
        )


def train_tedd1104(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
):
    logging.info(f"Loading {model_args.model_name_or_path} model...")

    if data_args.task == "video-classification":
        load_fn = partial(VideoMAEForVideoClassification.from_pretrained, num_labels=9)
    elif data_args.task == "video-masking":
        load_fn = VideoMAEForPreTraining.from_pretrained
    else:
        raise ValueError(
            f"task not in ['video-classification', 'video-masking']. "
            f"task: {data_args.task}"
        )

    model = load_fn(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
    )

    logging.info(f"Model dtype: {model.dtype}")

    logging.info(f"=== Training {model_args.model_name_or_path} model ===")
    logging.info(f"Task: {data_args.task}")
    trainable_params, total_params, trainable_percentage = get_trainable_parameters(
        model
    )
    logging.info(
        f"---> Trainable params: {trainable_params} || all params: {total_params} ||"
        f" trainable%: {round(trainable_percentage, 6)}\n"
    )

    logging.info(f"Loading {data_args.task} dataset...")
    train_dataset = Tedd1104Dataset(
        dataset_dir=data_args.train_dir,
        hide_map_prob=data_args.hide_map_probability_train,
        tubelet_mask_ratio=data_args.tubelet_mask_ratio,
        image_mask_ratio=data_args.image_mask_ratio,
        patch_size=model.config.patch_size,
        tubelet_size=model.config.tubelet_size,
        task=data_args.task,
        inference=False,
    )

    validation_dataset = Tedd1104Dataset(
        dataset_dir=data_args.validation_dir,
        hide_map_prob=data_args.hide_map_probability_validation,
        tubelet_mask_ratio=0.0,
        image_mask_ratio=0.0,
        patch_size=model.config.patch_size,
        tubelet_size=model.config.tubelet_size,
        task=data_args.task,
        inference=True,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        args=training_args,
        data_collator=None,
        compute_metrics=compute_metrics
        if data_args.task == "video-classification"
        else None,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    logging.info(f"Saving model to {training_args.output_dir}")

    trainer.save_model(training_args.output_dir)


def inference_tedd1104(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
):
    if training_args.do_train:
        model_path = training_args.output_dir
    else:
        model_path = model_args.model_name_or_path

    logging.info(f"Loading {model_args.model_name_or_path} model...")

    if training_args.bf16_full_eval:
        model_dtype = torch.bfloat16
    elif training_args.fp16_full_eval:
        model_dtype = torch.float16
    else:
        model_dtype = None

    if model_args.quantization_inference:
        from bitsandbytes import BitsAndBytesConfig

        quant_args = (
            {"load_in_4bit": True}
            if model_args.quantization_inference == 4
            else {"load_in_8bit": True}
        )
        if model_args.quantization_inference == 4:
            if torch.cuda.is_bf16_supported() and not training_args.fp16_full_eval:
                bnb_4bit_compute_dtype = torch.bfloat16
            elif training_args.fp16_full_eval:
                bnb_4bit_compute_dtype = torch.float16
            else:
                bnb_4bit_compute_dtype = torch.float32

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            )

        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        logging.info(
            f"Bits and Bytes config: {json.dumps(bnb_config.to_dict(),indent=4,ensure_ascii=False)}"
        )
    else:
        logging.info(f"Loading model with dtype: {model_dtype}")
        bnb_config = None
        quant_args = {}

    if data_args.task == "video-classification":
        load_fn = partial(VideoMAEForVideoClassification.from_pretrained, num_labels=9)
    elif data_args.task == "video-masking":
        load_fn = VideoMAEForPreTraining.from_pretrained
    else:
        raise ValueError(
            f"task not in ['video-classification', 'video-masking']. "
            f"task: {data_args.task}"
        )

    model = load_fn(
        model_path,
        torch_dtype=model_dtype,
        quantization_config=bnb_config,
        **quant_args,
    )

    logging.info(f"Model dtype: {model.dtype}")

    logging.info(f"=== Inference {model_args.model_name_or_path} model ===")
    logging.info(f"Task: {data_args.task}")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=None,
        compute_metrics=compute_metrics
        if data_args.task == "video-classification"
        else None,
    )

    metrics = {}

    for test_dir in data_args.test_dirs:
        logging.info(f"Loading {test_dir} dataset...")
        test_dataset = Tedd1104Dataset(
            dataset_dir=test_dir,
            hide_map_prob=data_args.hide_map_probability_test,
            tubelet_mask_ratio=0.0,
            image_mask_ratio=0.0,
            patch_size=model.config.patch_size,
            tubelet_size=model.config.tubelet_size,
            task=data_args.task,
            inference=True,
        )

        dataset_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        for metric, value in dataset_metrics.items():
            metrics[f"test_{os.path.basename(test_dir)}_{metric}"] = value

    print(json.dumps(metrics, ensure_ascii=False, indent=4))

    with open(
        os.path.join(training_args.output_dir, "metrics.json"), "w", encoding="utf8"
    ) as output_file:
        print(json.dumps(metrics, ensure_ascii=False, indent=4), file=output_file)

    trainer.log(metrics)


if __name__ == "__main__":
    init_logger()

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    logging.info(f"Sys args {sys.argv}")

    if len(sys.argv) > 0 and sys.argv[-1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        logging.info(f"Loading json config {sys.argv[-1]}")
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[-1])
        )

    elif len(sys.argv) > 0 and sys.argv[-1].endswith(".yaml"):
        # If we pass only one argument to the script, and it's the path to a yaml file,
        # let's parse it to get our arguments.
        logging.info(f"Loading yaml config {sys.argv[-1]}")
        model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[-1])
        )
    else:
        logging.info("No config file passed, using command line arguments.")
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.do_train:
        train_tedd1104(model_args, data_args, training_args)
        clean_cache()

    if training_args.do_predict:
        inference_tedd1104(model_args, data_args, training_args)
        clean_cache()
