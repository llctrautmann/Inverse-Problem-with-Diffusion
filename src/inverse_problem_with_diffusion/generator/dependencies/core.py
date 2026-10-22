import json
import multiprocessing as mp
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from tqdm import tqdm


class EllipseDatasetGenerator:
    """
    Default ellipse generator: every latent is sampled independently.

    Templates subclass this and override only the hooks they need:
      _sample_Af / _sample_Ar / _sample_theta   draw one latent
      _sample_intensity                         intensity of one ellipse
      sample_geometry                           change the *order* latents are drawn in
                                                (see _place for the placement helper)
    """

    def __init__(
        self,
        canvas,
        sampler,
        render,
        misc,
        n_ellipses: int = 8,
        intensity_func=None,
    ):
        self.canvas = canvas
        self.sampler = sampler
        self.render = render
        self.misc = misc
        self.base_seed = misc.seed
        self.n_ellipses = n_ellipses
        self.intensity_func = intensity_func

    # deterministic RNG for image i (order-independent, parallel-safe)
    def _rng_for(self, index: int):
        ss = np.random.SeedSequence([self.base_seed, int(index)])
        return np.random.default_rng(ss)

    # ---- latent sampling hooks ----
    def _sample_Af(self, rng, **latents):
        """Area fraction: ellipse area / canvas area."""
        s = self.sampler
        if s.log_Af:
            return np.exp(rng.uniform(np.log(s.Af_lo), np.log(s.Af_hi)))
        return rng.uniform(s.Af_lo, s.Af_hi)

    def _sample_Ar(self, rng, **latents):
        """Aspect ratio: major axis / minor axis."""
        s = self.sampler
        if s.log_Ar:
            return np.exp(rng.uniform(np.log(s.Ar_lo), np.log(s.Ar_hi)))
        return rng.uniform(s.Ar_lo, s.Ar_hi)

    def _sample_theta(self, rng, **latents):
        s = self.sampler
        return rng.uniform(s.theta_lo, s.theta_hi)

    def _sample_intensity(self, rng, g):
        """Intensity of one ellipse. `g` is the geometry dict from sample_geometry."""
        if self.intensity_func is None:
            return int(rng.integers(self.render.intensity_lo, self.render.intensity_hi))
        return self.intensity_func(
            **g, W=self.canvas.W, H=self.canvas.H, rng=rng
        )

    # ---- geometry ----
    def sample_geometry(self, rng, max_tries: int = 1000):
        """Draw Af, Ar, then retry (theta, position) until the ellipse fits."""
        af = self._sample_Af(rng)
        ar = self._sample_Ar(rng, Af=af)
        for _ in range(max_tries):
            theta = self._sample_theta(rng, Af=af, Ar=ar)
            g = self._place(rng, af, ar, theta)
            if g is not None:
                return g
        raise RuntimeError("No feasible placement for the sampled Af/Ar.")

    def _place(self, rng, af, ar, theta, cx=None, cy=None):
        """
        Build the geometry dict, drawing the centre uniformly from where the
        rotated ellipse fits inside the inset canvas. Pass cx and/or cy to fix a
        coordinate that was already drawn. Returns None if it does not fit.
        """
        H, W, inset = self.canvas.H, self.canvas.W, self.canvas.inset
        a, b = self.a_b_from_Af_Ar(af, ar, H, W)
        hx, hy = self.rot_half_extents(a, b, theta)
        xmin, xmax = inset + hx, (W - inset) - hx
        ymin, ymax = inset + hy, (H - inset) - hy
        if not (xmin < xmax and ymin < ymax):
            return None
        if cx is not None and not xmin <= cx <= xmax:
            return None
        if cy is not None and not ymin <= cy <= ymax:
            return None
        if cx is None:
            cx = rng.uniform(xmin, xmax)
        if cy is None:
            cy = rng.uniform(ymin, ymax)
        return {"cx": cx, "cy": cy, "theta": theta, "Af": af, "Ar": ar, "a": a, "b": b}

    def _coverage_mask(self, cx, cy, a, b, theta):
        """Fraction of each pixel covered by the ellipse, via supersampling."""
        H, W, ss = self.canvas.H, self.canvas.W, self.render.ss_factor

        # sub-pixel centres on the upsampled grid, in pixel units
        yy, xx = np.meshgrid(
            np.arange(H * ss) + 0.5, np.arange(W * ss) + 0.5, indexing="ij"
        )
        xx, yy = xx / ss, yy / ss

        # shift to the ellipse centre and rotate into its frame
        c, s = np.cos(theta), np.sin(theta)
        x0, y0 = xx - cx, yy - cy
        xr = c * x0 + s * y0
        yr = -s * x0 + c * y0
        inside = (xr / a) ** 2 + (yr / b) ** 2 <= 1

        # average each ss x ss block back down to one pixel
        return (
            inside.reshape(H, ss, W, ss).mean(axis=(1, 3)).astype(np.float32)
        )

    # ---- image ----
    def sample_image(self, index: int, max_tries: int = 1000):
        rng = self._rng_for(index)
        H, W = self.canvas.H, self.canvas.W
        bg, overlap = self.render.bg, self.render.overlap
        moat = self.sampler.moat_pixels

        img = np.full((H, W), bg, dtype=np.float32)
        acc_msk = np.zeros((H, W), dtype=bool)
        ellipse_meta = []

        for _ in range(self.n_ellipses):
            # the accumulated mask is fixed while we retry this ellipse
            forbidden = None
            if moat > 0:
                se = np.ones((1 + 2 * moat, 1 + 2 * moat), dtype=bool)
                forbidden = binary_dilation(acc_msk, structure=se)

            for _ in range(max_tries):
                g = self.sample_geometry(rng)
                msk = self._coverage_mask(g["cx"], g["cy"], g["a"], g["b"], g["theta"])
                new_msk = msk > 0.0
                if forbidden is None or not np.any(forbidden & new_msk):
                    break
            else:
                raise RuntimeError(
                    f"Image {index}: could not place an ellipse; "
                    "lower n_ellipses or moat_pixels."
                )

            I = self._sample_intensity(rng, g)
            acc_msk |= new_msk
            layer = bg * (1 - msk) + I * msk

            if overlap == "last":
                w = msk > 0
                img[w] = layer[w]
            elif overlap == "max":
                t = layer >= img
                img[t] = layer[t]
            elif overlap == "add":
                img += (I - bg) * msk
            else:
                raise ValueError("overlap must be 'last' | 'max' | 'add'.")

            ellipse_meta.append({**g, "intensity": float(I)})

        # unit-variance noise on the empty background only
        empty = ~acc_msk
        img[empty] += rng.normal(0, 1, size=empty.sum())
        img = np.clip(img, 0, 255).astype(np.uint8)

        meta = {
            "index": int(index),
            "seed": int(self.base_seed),
            "globals": {
                "H": H,
                "W": W,
                "inset": self.canvas.inset,
                "ss": self.render.ss_factor,
                "bg": bg,
                "overlap": overlap,
                "n_ellipses": len(ellipse_meta),
            },
            "ellipses": ellipse_meta,
        }
        return img, meta

    # ---- saving ----
    def _save_one(self, index: int):
        img, meta = self.sample_image(index)
        Image.fromarray(img).save(self.misc.save_path / f"{index:06d}.png")
        return meta

    def save_images(self, workers: Optional[int] = None):
        """
        Write every image as a PNG plus metadata.json into misc.save_path.
        workers=None uses all cores, workers=1 runs in-process.
        """
        self.misc.save_path.mkdir(parents=True, exist_ok=True)
        n = self.misc.n_datapoints

        if workers == 1:
            metadata = [self._save_one(i) for i in tqdm(range(n), desc="Generating")]
        else:
            with mp.Pool(workers) as pool:
                metadata = list(
                    tqdm(pool.imap(self._save_one, range(n), chunksize=32), total=n, desc="Generating")
                )

        with open(self.misc.save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved {len(metadata)} images and metadata to {self.misc.save_path}")

    # ---- geometry helpers ----
    @staticmethod
    def a_b_from_Af_Ar(Af: float, Ar: float, H: int, W: int):
        """Semi-axes from area fraction and aspect ratio: pi*a*b = Af*H*W and a/b = Ar."""
        K = (Af * H * W) / np.pi  # a * b
        return np.sqrt(K * Ar), np.sqrt(K / Ar)

    @staticmethod
    def rot_half_extents(a: float, b: float, theta: float):
        """Half-width and half-height of the axis-aligned bounding box of the rotated ellipse."""
        c, s = np.cos(theta), np.sin(theta)
        return np.hypot(a * c, b * s), np.hypot(a * s, b * c)
