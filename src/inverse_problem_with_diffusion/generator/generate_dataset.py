import argparse
from dataclasses import replace

from inverse_problem_with_diffusion.generator.dependencies.covariations import CovariationPatterns
from inverse_problem_with_diffusion.generator.dependencies.specs import load_config
from inverse_problem_with_diffusion.generator.templates import TEMPLATES


def build_generator(config_path, out=None, n_samples=None):
    """Build the generator described by a TOML config (optionally overriding output path / size)."""
    cfg = load_config(config_path)
    misc = cfg.misc
    if out is not None:
        misc = replace(misc, save_path=out.resolve())
    if n_samples is not None:
        misc = replace(misc, n_datapoints=n_samples)

    intensity_func = None
    if cfg.generator.intensity:
        intensity_func = getattr(CovariationPatterns, f"{cfg.generator.intensity}_intensity")

    return TEMPLATES[cfg.generator.template](
        canvas=cfg.canvas,
        sampler=cfg.sampler,
        render=cfg.render,
        misc=misc,
        n_ellipses=cfg.generator.n_ellipses,
        intensity_func=intensity_func,
    )


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate an ellipse dataset from a TOML config.")
    parser.add_argument("--config", default="configs/ellipse_baseline_train.toml")
    parser.add_argument("--out", type=Path, help="override [paths] name")
    parser.add_argument("--n-samples", type=int, help="override [dataset] n_samples")
    parser.add_argument("--workers", type=int, default=None, help="processes (default: all cores, 1: in-process)")
    args = parser.parse_args()

    gen = build_generator(args.config, args.out, args.n_samples)
    print(f"{type(gen).__name__} -> {gen.misc.save_path} ({gen.misc.n_datapoints} images, seed {gen.misc.seed})")
    gen.save_images(workers=args.workers)


if __name__ == "__main__":
    main()
