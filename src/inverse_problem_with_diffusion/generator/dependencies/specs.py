import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class CanvasSpec:
    H: int
    W: int
    inset: int


@dataclass
class SamplerSpec:
    Af_lo: float  # min area fraction of an ellipse
    Af_hi: float  # max area fraction of an ellipse
    Ar_lo: float  # min aspect ratio
    Ar_hi: float  # max aspect ratio
    log_Af: bool
    log_Ar: bool
    theta_lo: float  # min rotation (radians)
    theta_hi: float  # max rotation (radians)
    moat_pixels: int  # 0 allows overlaps, >0 keeps ellipses this far apart


@dataclass
class RenderSpec:
    intensity_lo: int
    intensity_hi: int  # exclusive
    ss_factor: int = 1  # 1 gives hard edges, >1 anti-aliases
    bg: int = 0
    overlap: Literal["last", "max", "add"] = "add"


@dataclass
class MiscSpec:
    seed: int
    save_path: Path
    n_datapoints: int


@dataclass
class GeneratorSpec:
    template: str = "default"  # key in templates.TEMPLATES
    n_ellipses: int = 8
    intensity: Optional[str] = None  # name of a CovariationPatterns.<name>_intensity


@dataclass
class Config:
    canvas: CanvasSpec
    sampler: SamplerSpec
    render: RenderSpec
    misc: MiscSpec
    generator: GeneratorSpec = field(default_factory=GeneratorSpec)


def load_config(path) -> Config:
    """Read a TOML config. Unknown keys raise, so typos do not pass silently."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    return Config(
        canvas=CanvasSpec(**raw["canvas"]),
        sampler=SamplerSpec(**raw["sampler"]),
        render=RenderSpec(**raw["render"]),
        misc=MiscSpec(
            seed=raw["dataset"]["seed"],
            n_datapoints=raw["dataset"]["n_samples"],
            save_path=Path(raw["paths"]["name"]).resolve(),
        ),
        generator=GeneratorSpec(**raw.get("generator", {})),
    )
