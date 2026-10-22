"""
Reconstruct images from degraded measurements with the diffusion prior (DPS).

For every selected ground-truth image x: simulate y = A(x) + noise, then draw several
reconstructions from p(x | y). The spread across the draws is the uncertainty signal.

    uv run python -m inverse_problem_with_diffusion.reconstruct --config configs/reconstruct.toml
"""

import argparse
import json
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMPipeline
from diffusers.utils import make_image_grid
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from inverse_problem_with_diffusion.dps import dps_sample
from inverse_problem_with_diffusion.operators import make_operator, measure


@dataclass
class ReconstructConfig:
    # prior
    checkpoint: str = "checkpoints/ellipsesLQ"

    # which ground-truth images to use: explicit file names, or the first num_images
    image_dir: str = "data/ellipses/train"
    images: list = field(default_factory=list)
    num_images: int = 1

    # measurement y = operator(x) + N(0, noise_sigma^2); see operators.py
    operator: str = "gaussian_blur"
    operator_params: dict = field(default_factory=lambda: {"sigma": 2.0})
    noise_sigma: float = 0.01

    # DPS sampling
    num_samples: int = 8  # reconstructions per measurement
    batch_size: int = 8  # how many of them to run at once (lower if memory runs out)
    scheduler: str = "ddim"  # "ddim" (fast) or "ddpm" (stochastic, the training sampler)
    num_inference_steps: int = 100
    guidance_scale: float = 1.0
    eta: float = 0.0  # ddim only

    seed: int = 0
    output_dir: str = "outputs/reconstruction"
    device: str = "auto"  # auto, cuda, mps or cpu


def load_reconstruct_config(path) -> ReconstructConfig:
    """Read a TOML config. Unknown keys raise, so typos do not pass silently."""
    with open(path, "rb") as f:
        return ReconstructConfig(**tomllib.load(f))


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(path) -> torch.Tensor:
    """PNG -> (1, 1, H, W) grayscale tensor in [0, 1], the space the operators work in."""
    img = Image.open(path).convert("L")
    return torch.from_numpy(np.array(img)).float().div(255)[None, None]


def pick_images(cfg: ReconstructConfig) -> list:
    folder = Path(cfg.image_dir)
    if cfg.images:
        return [folder / name for name in cfg.images]
    paths = sorted(folder.glob("*.png"))[: cfg.num_images]
    if not paths:
        raise FileNotFoundError(f"No .png files found in {folder.resolve()}")
    return paths


def to_display(t: torch.Tensor, size: int) -> Image.Image:
    """(1, H, W) in [0, 1] -> PIL image, nearest-neighbour resized so all panels match."""
    img = to_pil_image(t.clamp(0, 1).cpu())
    return img.resize((size, size), Image.NEAREST) if img.size != (size, size) else img


def reconstruct_image(cfg, model, scheduler, operator, image_path, device, generator) -> dict:
    """One ground truth -> one measurement -> cfg.num_samples reconstructions."""
    x_true = load_image(image_path).to(device)
    y = measure(operator, x_true, cfg.noise_sigma, generator)

    recons = []
    for start in range(0, cfg.num_samples, cfg.batch_size):
        n = min(cfg.batch_size, cfg.num_samples - start)
        x_init = torch.randn((n, *x_true.shape[1:]), generator=generator).to(device)
        recons.append(
            dps_sample(
                model=model,
                scheduler=scheduler,
                measurement=y.expand(n, *y.shape[1:]),
                operator=operator,
                x_init=x_init,
                device=device,
                guidance_scale=cfg.guidance_scale,
                eta=cfg.eta,
            ).detach()
        )
    return {"x_true": x_true.cpu(), "y": y.cpu(), "recons": torch.cat(recons).cpu()}


def save_result(result: dict, out: Path) -> None:
    """Write PNGs, a summary grid and the raw tensors for one image."""
    out.mkdir(parents=True, exist_ok=True)
    x_true, y, recons = result["x_true"][0], result["y"][0], result["recons"]
    size = x_true.shape[-1]

    mean, std = recons.mean(0), recons.std(0) if len(recons) > 1 else torch.zeros_like(recons[0])
    panels = [x_true, y, mean, std / (std.max() + 1e-8)] + list(recons)
    imgs = [to_display(p, size) for p in panels]

    imgs[0].save(out / "ground_truth.png")
    imgs[1].save(out / "measurement.png")
    for k, img in enumerate(imgs[4:]):
        img.save(out / f"recon_{k:03d}.png")

    # summary: ground truth | measurement | mean | std (normalised) | each reconstruction
    cols = 4
    rows = -(-len(imgs) // cols)
    make_image_grid(imgs + [Image.new("L", (size, size))] * (rows * cols - len(imgs)), rows, cols).save(
        out / "summary.png"
    )

    # raw, unclamped values in [0, 1]: this is what an evaluation should read
    torch.save(result, out / "tensors.pt")


def run(cfg: ReconstructConfig) -> dict:
    """Run the whole experiment described by cfg. Returns {image name: result}."""
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)  # also seeds the stochastic DDPM step
    generator = torch.Generator().manual_seed(cfg.seed)

    pipe = DDPMPipeline.from_pretrained(cfg.checkpoint)
    model = pipe.unet.to(device).eval().requires_grad_(False)

    if cfg.scheduler == "ddim":
        scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif cfg.scheduler == "ddpm":
        scheduler = pipe.scheduler
    else:
        raise ValueError("scheduler must be 'ddim' or 'ddpm'")
    scheduler.set_timesteps(cfg.num_inference_steps)

    operator = make_operator(cfg.operator, **cfg.operator_params)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    results = {}
    for path in pick_images(cfg):
        print(f"{path.name}: {cfg.operator} {cfg.operator_params}, noise {cfg.noise_sigma}, device {device}")
        results[path.name] = reconstruct_image(cfg, model, scheduler, operator, path, device, generator)
        save_result(results[path.name], out_dir / path.stem)
    print(f"Saved to {out_dir.resolve()}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Reconstruct images from degraded measurements.")
    parser.add_argument("--config", default="configs/reconstruct.toml")
    parser.add_argument("--output_dir", help="overrides output_dir in the config")
    parser.add_argument("--seed", type=int, help="overrides seed in the config")
    args = parser.parse_args()

    cfg = load_reconstruct_config(args.config)
    if args.output_dir is not None:
        cfg = replace(cfg, output_dir=args.output_dir)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    run(cfg)


if __name__ == "__main__":
    main()
