import numpy as np

from inverse_problem_with_diffusion.generator.dependencies.core import EllipseDatasetGenerator
from inverse_problem_with_diffusion.generator.dependencies.covariations import CovariationPatterns


class OscillatoryAspectAreaGenerator(EllipseDatasetGenerator):
    """
    Aspect ratio is a sinusoidal function of the area fraction.
    Pair with intensity = "aspect_oscillation" to also make intensity oscillate with Ar.
    """

    def _sample_Ar(self, rng, Af=None, **latents):
        ar = CovariationPatterns.sinusoidal_constraint(
            Af, Ar_mean=2.5, amplitude=1.5, k=10, noise_std=0.1, rng=rng
        )
        return np.clip(ar, self.sampler.Ar_lo, self.sampler.Ar_hi)
