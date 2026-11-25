from src.contracts import RegionPack

from .adapter import build_regions
from .baselines import (
    global_single_region,
    scan_alphas_uniform,
    uniform_grid_partition,
)
from .graph import (
    build_region_graph,
    coordination_stats,
    equal_weight_graph,
)
from .masks import (
    compute_effective_areas,
    generate_soft_masks,
    validate_soft_masks,
)

__all__ = [
    "RegionPack",
    "build_regions",
    "uniform_grid_partition",
    "global_single_region",
    "scan_alphas_uniform",
    "build_region_graph",
    "equal_weight_graph",
    "coordination_stats",
    "generate_soft_masks",
    "validate_soft_masks",
    "compute_effective_areas",
]
