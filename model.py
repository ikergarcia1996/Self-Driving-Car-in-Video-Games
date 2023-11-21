from transformers import VideoMAEModel, VideoMAEConfig
from utils import get_trainable_parameters
import os
import argparse


class VideoMAEsmall:
    def __init__(self):
        self.name = "VideoMAE-small"
        self.config = VideoMAEConfig(
            image_size=(270, 480),
            patch_size=15,
            num_channels=3,
            num_frames=5,
            tubelet_size=1,
            hidden_size=384,
            num_hidden_layers=4,
            num_attention_heads=6,
            intermediate_size=1536,
            decoder_num_attention_heads=3,
            decoder_hidden_size=192,
            decoder_num_hidden_layers=3,
            decoder_intermediate_size=1536,
        )

    def get_model(self):
        print(f"Loading {self.name} model")
        return VideoMAEModel(self.config)


class VideoMAEbase:
    def __init__(self):
        self.name = "VideoMAE-base"
        self.config = VideoMAEConfig(
            image_size=(270, 480),
            patch_size=15,
            num_channels=3,
            num_frames=5,
            tubelet_size=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            decoder_num_attention_heads=6,
            decoder_hidden_size=384,
            decoder_num_hidden_layers=6,
            decoder_intermediate_size=1536,
        )

    def get_model(self):
        print(f"Loading {self.name} model")
        return VideoMAEModel(self.config)


class VideoMAEbase_PS30:
    def __init__(self):
        self.name = "VideoMAE-base_ps30"
        self.config = VideoMAEConfig(
            image_size=(270, 480),
            patch_size=30,
            num_channels=3,
            num_frames=5,
            tubelet_size=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            decoder_num_attention_heads=6,
            decoder_hidden_size=384,
            decoder_num_hidden_layers=6,
            decoder_intermediate_size=1536,
        )

    def get_model(self):
        print(f"Loading {self.name} model")
        return VideoMAEModel(self.config)


class VideoMAEbase_TS5:
    def __init__(self):
        self.name = "VideoMAE-base_ts5"
        self.config = VideoMAEConfig(
            image_size=(270, 480),
            patch_size=15,
            num_channels=3,
            num_frames=5,
            tubelet_size=5,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            decoder_num_attention_heads=6,
            decoder_hidden_size=384,
            decoder_num_hidden_layers=6,
            decoder_intermediate_size=1536,
        )

    def get_model(self):
        print(f"Loading {self.name} model")
        return VideoMAEModel(self.config)


class VideoMAEbase_PS30_TS5:
    def __init__(self):
        self.name = "VideoMAE-base_ps30_ts5"
        self.config = VideoMAEConfig(
            image_size=(270, 480),
            patch_size=30,
            num_channels=3,
            num_frames=5,
            tubelet_size=5,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            decoder_num_attention_heads=6,
            decoder_hidden_size=384,
            decoder_num_hidden_layers=6,
            decoder_intermediate_size=1536,
        )

    def get_model(self):
        print(f"Loading {self.name} model")
        return VideoMAEModel(self.config)


class VideoMAElarge:
    def __init__(self):
        self.name = "VideoMAE-large"
        self.config = VideoMAEConfig(
            image_size=(270, 480),
            patch_size=15,
            num_channels=3,
            num_frames=5,
            tubelet_size=5,
            hidden_size=1024,
            num_hidden_layers=16,
            num_attention_heads=16,
            intermediate_size=4096,
            decoder_num_attention_heads=8,
            decoder_hidden_size=512,
            decoder_num_hidden_layers=8,
            decoder_intermediate_size=2048,
        )

    def get_model(self):
        print(f"Loading {self.name} model")
        return VideoMAEModel(self.config)


class VideoMAElarge_TS5:
    def __init__(self):
        self.name = "VideoMAE-large_ts5"
        self.config = VideoMAEConfig(
            image_size=(270, 480),
            patch_size=15,
            num_channels=3,
            num_frames=5,
            tubelet_size=5,
            hidden_size=1024,
            num_hidden_layers=16,
            num_attention_heads=16,
            intermediate_size=4096,
            decoder_num_attention_heads=8,
            decoder_hidden_size=512,
            decoder_num_hidden_layers=8,
            decoder_intermediate_size=2048,
        )

    def get_model(self):
        print(f"Loading {self.name} model")
        return VideoMAEModel(self.config)


class VideoMAElarge_PS30_TS5:
    def __init__(self):
        self.name = "VideoMAE-large_ps30_ts5"
        self.config = VideoMAEConfig(
            image_size=(270, 480),
            patch_size=15,
            num_channels=3,
            num_frames=5,
            tubelet_size=5,
            hidden_size=1024,
            num_hidden_layers=16,
            num_attention_heads=16,
            intermediate_size=4096,
            decoder_num_attention_heads=8,
            decoder_hidden_size=512,
            decoder_num_hidden_layers=8,
            decoder_intermediate_size=2048,
        )

    def get_model(self):
        print(f"Loading {self.name} model")
        return VideoMAEModel(self.config)


class VideoMAExxl:
    def __init__(self):
        self.name = "VideoMAE-xxl"
        self.config = VideoMAEConfig(
            image_size=(270, 480),
            patch_size=15,
            num_channels=3,
            num_frames=5,
            tubelet_size=5,
            hidden_size=1280,
            num_hidden_layers=24,
            num_attention_heads=24,
            intermediate_size=5120,
            decoder_num_attention_heads=10,
            decoder_hidden_size=640,
            decoder_num_hidden_layers=10,
            decoder_intermediate_size=2560,
        )

    def get_model(self):
        print(f"Loading {self.name} model")
        return VideoMAEModel(self.config)


def initialize_models(output_path: str):
    os.makedirs(output_path, exist_ok=True)
    for model in [
        VideoMAEsmall,
        VideoMAEbase,
        VideoMAEbase_PS30,
        VideoMAEbase_TS5,
        VideoMAEbase_PS30_TS5,
        VideoMAElarge,
        VideoMAExxl,
    ]:
        model_cls = model()
        model = model_cls.get_model()
        print(f"Model: {model_cls.name}")
        _, params, _ = get_trainable_parameters(model)
        print(f"Number of trainable parameters: {params}")
        model_output_path = os.path.join(output_path, model_cls.name)
        print(f"Saving model to {model_output_path}")
        model.save_pretrained(model_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the models",
        default="models",
    )

    args = parser.parse_args()

    initialize_models(args.output_path)
