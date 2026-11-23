"""M6: scoring, constrained local search, baselines and experiments.

Layering rule (see M6/README.md): this package must stay importable without
torch for the pure search core (M6.search.solver). torch is only required by
M6.search.objective, M6.search.adapter and M6.regions_grid, which are only
imported in the real integration path.
"""

__version__ = "0.1.0"
