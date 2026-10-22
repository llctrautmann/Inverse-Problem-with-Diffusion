import numpy as np

from inverse_problem_with_diffusion.generator.dependencies.core import EllipseDatasetGenerator
from inverse_problem_with_diffusion.generator.dependencies.covariations import CovariationPatterns


class LinearPositionRotationGenerator(EllipseDatasetGenerator):
    """
    Aspect ratio and rotation both depend on the x-position of the ellipse:
    Ar oscillates sinusoidally with cx, theta grows linearly with cx.
    """

    def sample_geometry(self, rng, max_tries: int = 1000):
        H, W, inset = self.canvas.H, self.canvas.W, self.canvas.inset
        af = self._sample_Af(rng)

        for _ in range(max_tries):
            cx = rng.uniform(inset, W - inset)
            ar = CovariationPatterns.sinusoidal_aspect(
                cx, W, Ar_mean=2.0, amplitude=1.0, n_cycles=4
            ) + rng.uniform(-0.1, 0.1)
            ar = np.clip(ar, self.sampler.Ar_lo, self.sampler.Ar_hi)
            theta = CovariationPatterns.linear_position_rotation(cx, W, rng=rng)

            g = self._place(rng, af, ar, theta, cx=cx)
            if g is not None:
                return g
        raise RuntimeError("No feasible placement.")
