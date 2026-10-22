# Standard library
import argparse
import math
import os
from pathlib import Path

# DL imports
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from huggingface_hub import create_repo, upload_folder
from torch.nn import functional as F
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

# Local imports
from inverse_problem_with_diffusion.data import make_train_dataloader
from inverse_problem_with_diffusion.model import DDPM, TrainingConfig, load_training_config


class Trainer:
    def __init__(self, config: TrainingConfig, model: DDPM):
        self.config = config
        self.model = model.model
        self.scheduler = model.scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate
        )

        self.train_dataloader = make_train_dataloader(self.config)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=(len(self.train_dataloader) * self.config.num_epochs),
        )

    def train(self):
        set_seed(self.config.seed)
        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(self.config.output_dir, "logs"),
        )

        if accelerator.is_main_process:
            if self.config.output_dir is not None:
                os.makedirs(self.config.output_dir, exist_ok=True)
                if self.config.push_to_hub:
                    repo_id = create_repo(
                        repo_id=self.config.hub_model_id
                        or Path(self.config.output_dir).name,
                        exist_ok=True,
                    ).repo_id
                accelerator.init_trackers("ddpm-training")
            else:
                raise ValueError("Output directory not specified")

        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = (
            accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        )

        global_step = 0

        for epoch in range(self.config.num_epochs):
            progress_bar = tqdm(
                total=len(self.train_dataloader),
                disable=not accelerator.is_local_main_process,
            )
            progress_bar.set_description(f"Epoch {epoch}")

            # NOTE set should_save condition
            should_save = (
                (epoch + 1) % self.config.save_model_epochs == 0
                or epoch == self.config.num_epochs - 1
            )

            # INFO: Training loop for MLL training starts here
            for step, batch in enumerate(self.train_dataloader):
                clean_images = batch  # (B, C, H, W) in [-1, 1], see data.py
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                batch_size = clean_images.shape[0]

                timesteps = torch.randint(
                    0,
                    self.scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=clean_images.device,
                    dtype=torch.int64,
                )

                # INFO: Add noise to images (forward diffusion)
                # WARN: Not sure if this needs to be changed @ sometime
                noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)

                with accelerator.accumulate(self.model):
                    # TODO: Check in notebook what the 0th, 1st, ... outputs are.
                    noise_pred = self.model(
                        noisy_images, timesteps, return_dict=False
                    )[  # NOTE: return_dict=false returns a tuple (sample, ...)
                        0
                    ]

                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    progress_bar.update(1)
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": self.lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                    global_step += 1



            # NOTE: Weight saving, post training
            if accelerator.is_main_process:
                if should_save:
                    pipeline = DDPMPipeline(
                        unet=accelerator.unwrap_model(self.model),
                        scheduler=self.scheduler,
                    )
                    # NOTE: save inference weights
                    pipeline.save_pretrained(self.config.output_dir)

                    # NOTE: save all states variables
                    # WARN: This will bloat the upload to 1.82 GB
                    if self.config.save_full_state:
                        accelerator.save_state(
                            output_dir=os.path.join(
                                self.config.output_dir, "checkpoint_state"
                            )
                        )

                    self.evaluate(epoch, pipeline, accelerator)

                    # NOTE Model saving on hub
                    if self.config.push_to_hub and (
                        epoch % 10 == 0 or epoch == self.config.num_epochs - 1
                    ):
                        upload_folder(
                            repo_id=repo_id,
                            folder_path=self.config.output_dir,
                            commit_message=f"Epoch {epoch}",
                            ignore_patterns=["step_*", "epoch_*"],
                        )

        accelerator.end_training()

    def evaluate(self, epoch: int, pipeline: DDPMPipeline, accelerator: Accelerator) -> None:
        """Evaluate the model by generating sample images."""
        pipeline.to(accelerator.device)
        images = pipeline(
            batch_size=self.config.eval_batch_size,
            num_inference_steps=self.config.eval_num_inference_steps,
            generator=torch.Generator().manual_seed(self.config.seed),
        ).images

        # Create image grid
        cols = math.ceil(math.sqrt(len(images)))
        rows = math.ceil(len(images) / cols)
        image_grid = make_image_grid(images, rows=rows, cols=cols)

        writer = accelerator.get_tracker("tensorboard").writer
        writer.add_image(
            tag="Generated Images/Epoch Samples",
            img_tensor=ToTensor()(image_grid),
            global_step=epoch,
        )
        writer.flush()


def main(args):
    config = load_training_config(args.config)

    # command line overrides the toml
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.seed is not None:
        config.seed = args.seed

    # set model
    model = DDPM(config=config)

    # init trainer + run
    trainer = Trainer(config=config, model=model)
    trainer.train()


# WARN Here we set command line arguments, maybe we use hydra later.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ddpm_ellipses.toml",
        help="path to config file",
    )
    parser.add_argument("--num_epochs", type=int, help="overrides num_epochs in the config")
    parser.add_argument("--output_dir", type=str, help="overrides output_dir in the config")
    parser.add_argument("--seed", type=int, help="overrides seed in the config")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
