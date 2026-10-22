import tomllib
from dataclasses import dataclass, field
from typing import Optional

from diffusers import DDPMScheduler, UNet2DModel


@dataclass
class TrainingConfig:
    # data
    data_dir: str
    image_size: int = 64
    in_channels: int = 1  # 1 = grayscale, 3 = RGB
    num_workers: int = 4

    # model
    block_out_channels: tuple = (128, 128, 256, 256, 512, 512)
    num_train_timesteps: int = 1000

    # optimisation
    train_batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    seed: int = 42

    # output and sampling
    output_dir: str = "checkpoints/ellipses_ddpm"
    save_model_epochs: int = 10
    save_full_state: bool = False
    eval_batch_size: int = 16
    eval_num_inference_steps: int = 1000

    # optional upload to the Hugging Face hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


def load_training_config(path) -> TrainingConfig:
    """Read a TOML config. Unknown keys raise, so typos do not pass silently."""
    with open(path, "rb") as f:
        return TrainingConfig(**tomllib.load(f))


class DDPM:
    """Holds the UNet and the noise scheduler that define the diffusion model."""

    def __init__(self, config: TrainingConfig):
        n = len(config.block_out_channels)
        # attention in the second-to-last block, as in the shipped ellipsesLQ model
        down = ["AttnDownBlock2D" if i == n - 2 else "DownBlock2D" for i in range(n)]
        up = ["AttnUpBlock2D" if i == 1 else "UpBlock2D" for i in range(n)]

        self.model = UNet2DModel(
            sample_size=config.image_size,
            in_channels=config.in_channels,
            out_channels=config.in_channels,
            layers_per_block=2,
            block_out_channels=tuple(config.block_out_channels),
            down_block_types=tuple(down),
            up_block_types=tuple(up),
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)
