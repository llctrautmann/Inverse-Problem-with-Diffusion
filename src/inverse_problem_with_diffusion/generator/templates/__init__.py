from inverse_problem_with_diffusion.generator.dependencies.core import EllipseDatasetGenerator
from inverse_problem_with_diffusion.generator.templates.linear_position_rotation import LinearPositionRotationGenerator
from inverse_problem_with_diffusion.generator.templates.oscillatory_aspect_area import OscillatoryAspectAreaGenerator
from inverse_problem_with_diffusion.generator.templates.position_modulated_aspect import PositionModulatedAspectGenerator

# names usable as `template = "..."` in the [generator] section of a config
TEMPLATES = {
    "default": EllipseDatasetGenerator,
    "linear_position_rotation": LinearPositionRotationGenerator,
    "oscillatory_aspect_area": OscillatoryAspectAreaGenerator,
    "position_modulated_aspect": PositionModulatedAspectGenerator,
}
