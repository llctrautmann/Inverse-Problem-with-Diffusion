import torch
from diffusers import DDIMPipeline, DDIMScheduler, DDPMPipeline

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

ddpm = DDPMPipeline.from_pretrained("checkpoints/ellipsesLQ").to(device)
# same trained UNet, deterministic DDIM sampler with far fewer steps
ddim = DDIMPipeline(unet=ddpm.unet, scheduler=DDIMScheduler.from_config(ddpm.scheduler.config))

samplers = {
    "ddpm": lambda: ddpm(batch_size=4, generator=torch.manual_seed(0)),
    "ddim": lambda: ddim(batch_size=4, num_inference_steps=50, eta=0.0, generator=torch.manual_seed(0)),
}

for name, sample in samplers.items():
    for i, image in enumerate(sample().images):
        image.save(f"sample_{name}_{i}.png")
