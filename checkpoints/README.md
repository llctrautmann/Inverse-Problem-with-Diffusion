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
