from typing import Callable, Optional, Tuple, Union

from math import sqrt
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from diffusers import DDIMScheduler


def dps_sample(
    model: nn.Module,
    scheduler,
    measurement: torch.Tensor,
    operator: Callable[[torch.Tensor], torch.Tensor],
    shape: Optional[Tuple[int, ...]] = None,
    x_init: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = "cuda",
    guidance_scale: float = 1.0,
    eta: float = 0.0,
    normalize_gradient: bool = False,
    show_progress: bool = True,
) -> torch.Tensor:
    """Solves an inverse problem by Diffusion Posterior Sampling (DPS).

    Runs the reverse diffusion process, guiding each step toward consistency
    with the measurement ``y = A(x) + noise``.

    Args:
        model: Noise predictor; its output has a ``.sample`` attribute.
        scheduler: Diffusion scheduler with ``.timesteps`` and
            ``.step(model_output, t, x_t)`` returning ``.prev_sample`` and
            ``.pred_original_sample``.
        measurement: Observed measurement ``y`` of shape (B, ...).
        operator: Differentiable forward operator ``A`` on [0, 1] images.
        shape: Shape (B, C, H, W) of the reconstruction. Required if ``x_init``
            is not given.
        x_init: Initial noise; drawn from N(0, I) if None.
        device: Device to sample on.
        guidance_scale: Scale on the guidance gradient, multiplied by
            ``sqrt(1000 / num_steps)`` in each step.
        eta: DDIM stochasticity, forwarded to a ``DDIMScheduler``.
        normalize_gradient: Unused.
        show_progress: Whether to show a progress bar.

    Returns:
        Reconstruction in [0, 1] with the shape of ``x_init`` or ``shape``.

    Raises:
        ValueError: If neither ``shape`` nor ``x_init`` is given.
    """
    if shape is None and x_init is None:
        raise ValueError("Either 'shape' or 'x_init' must be provided.")

    measurement = measurement.to(device)

    if x_init is not None:
        x_t = x_init.to(device)
    else:
        x_t = torch.randn(shape, device=device)

    timesteps = scheduler.timesteps
    iterator = tqdm(timesteps, desc="DPS Sampling", leave=False) if show_progress else timesteps

    for _, t in enumerate(iterator):
        x_t = _dps_step(
            x_t=x_t,
            t=t,
            model=model,
            scheduler=scheduler,
            measurement=measurement,
            operator=operator,
            eta=eta,
            guidance_scale=guidance_scale
        )

    # Convert from [-1, 1] to [0, 1] range
    reconstruction = (x_t + 1) / 2

    return reconstruction


def _dps_step(
    x_t: torch.Tensor,
    t: torch.Tensor,
    model: nn.Module,
    scheduler,
    measurement: torch.Tensor,
    operator: Callable[[torch.Tensor], torch.Tensor],
    eta: float = 0.0,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Performs one DPS step.

    Estimates ``x_0`` from ``x_t`` (Tweedie), takes the gradient of
    ``||y - A(x_0_hat)||`` with respect to ``x_t``, and subtracts it, scaled,
    from the unguided reverse step.

    Args:
        x_t: Current noisy sample.
        t: Current timestep.
        model: Noise predictor.
        scheduler: Diffusion scheduler.
        measurement: Observed measurement ``y``.
        operator: Differentiable forward operator ``A`` on [0, 1] images.
        eta: DDIM stochasticity.
        guidance_scale: Scale on the guidance gradient.

    Returns:
        The sample ``x_{t-1}``.
    """
    x_t = x_t.detach().requires_grad_(True)

    model_output = model(x_t, t).sample[:, :3]

    # Tweedie estimate of x_0, from the scheduler
    if isinstance(scheduler, DDIMScheduler):
        step_output = scheduler.step(model_output, int(t), x_t, eta=eta)
    else:
        step_output = scheduler.step(model_output, int(t), x_t)

    x_0_hat = step_output.pred_original_sample

    # the operator works on [0, 1] images
    x_0_hat_scaled = (x_0_hat + 1) / 2

    predicted_measurement = operator(x_0_hat_scaled)
    residual = (measurement - predicted_measurement).flatten(start_dim=1)  # (B, -1)
    per_sample_norm = torch.linalg.vector_norm(residual, dim=1)            # (B,)
    loss = per_sample_norm.sum()

    grad = torch.autograd.grad(outputs=loss, inputs=x_t)[0]

    prev_sample = step_output.prev_sample

    x_t_minus_1 = prev_sample - sqrt(1000. / float(len(scheduler.timesteps))) * guidance_scale * grad



    return x_t_minus_1.detach()
