import numpy as np

from inverse_problem_with_diffusion.generator.dependencies.core import EllipseDatasetGenerator
from inverse_problem_with_diffusion.generator.dependencies.covariations import CovariationPatterns


class PositionModulatedAspectGenerator(EllipseDatasetGenerator):
    """
    The area -> aspect relationship is modulated by the y-position of the ellipse.
    Pair with intensity = "eccentricity" so intensity follows the resulting shape.
    """

    def sample_geometry(self, rng, max_tries: int = 1000):
        H, inset = self.canvas.H, self.canvas.inset
        af = self._sample_Af(rng)

        for _ in range(max_tries):
            cy = rng.uniform(inset, H - inset)
            ar = CovariationPatterns.position_modulated_area_aspect(af, cy, H, k=2.0, c=1.2)
            ar = np.clip(ar, self.sampler.Ar_lo, self.sampler.Ar_hi)
            theta = self._sample_theta(rng)

            g = self._place(rng, af, ar, theta, cy=cy)
            if g is not None:
                return g
        raise RuntimeError("No feasible placement.")
