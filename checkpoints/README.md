# Checkpoints

Trained diffusion models, saved as Hugging Face `diffusers` pipelines. Each subdirectory is one model and loads with `from_pretrained`.

| Name | Pipeline | Image size | Channels | Size on disk |
|---|---|---|---|---|
| `ellipsesLQ` | `DDPMPipeline` | 64 x 64 | 1 (grayscale) | ~434 MB |

## Layout

```
<name>/
├── model_index.json    # pipeline class and component classes
├── scheduler/          # DDPMScheduler config
└── unet/               # UNet2DModel config + weights (safetensors)
```

## Loading

```python
from diffusers import DDPMPipeline

pipe = DDPMPipeline.from_pretrained("checkpoints/ellipsesLQ")
images = pipe(batch_size=4).images
```

`scripts/generate.py` does this and also samples with DDIM from the same UNet.

## ellipsesLQ

Read from the saved configs:

- **UNet** (`UNet2DModel`): 1 input and 1 output channel, `sample_size` 64, six blocks with channels 128, 128, 256, 256, 512, 512, 2 layers per block, self-attention in one down block and one up block, SiLU activation, positional time embedding.
- **Scheduler** (`DDPMScheduler`): 1000 training timesteps, linear betas from 0.0001 to 0.02, epsilon prediction, `clip_sample` on, fixed-small variance.
- **Unconditional**: no class or other conditioning (`num_class_embeds` is null).
- **Saved with** `diffusers` 0.35.2.

Not recorded in the checkpoint, fill in:

- Training data (dataset name, generator config, seed, number of images): TODO
- Training length, batch size, learning rate, optimiser: TODO
- What "LQ" stands for: TODO
- Date and the script or commit that produced it: TODO

## Notes

- This directory is git-ignored because the weights are large. Keep this README and any model notes up to date by hand.
- `.DS_Store` files can be ignored or deleted.
