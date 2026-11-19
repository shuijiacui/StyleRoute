"""Baselines for the M6 experiment matrix (README section 6).

Fairness rules enforced here:

* plain local search and random-state search reuse ``solver.run_search`` with
  the SAME comparator, budget, cache and restart machinery — only the variable
  selection strategy differs;
* the global strength scan is a separate sweep over region-uniform states and
  reports its REAL evaluation count; it never pretends to be a 40-state search;
* every method keeps its own feasibility archive and counts the default state
  exactly once, exactly like the risk-guided method.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from .solver import (
    Comparator,
    EvaluationEngine,
    Score,
    SearchConfig,
    SearchOutcome,
    default_alphas,
    run_search,
)


def run_plain_local(
    *,
    evaluate: Callable[[tuple[float, ...]], tuple[Score, Any]],
    n_regions: int,
    config: SearchConfig | None = None,
    initial_decoder_calls: int = 0,
    decoder_counter: Callable[[], None] | None = None,
) -> SearchOutcome:
    """Plain local search: identical to risk-guided except random permutation
    ordering of variables (fixed seed)."""
    config = config or SearchConfig()
    config = SearchConfig(**{**config.__dict__, "mode": "random_permutation"})
    return run_search(
        evaluate=evaluate,
        n_regions=n_regions,
        config=config,
        initial_decoder_calls=initial_decoder_calls,
        decoder_counter=decoder_counter,
    )


def run_random_states(
    *,
    evaluate: Callable[[tuple[float, ...]], tuple[Score, Any]],
    n_regions: int,
    config: SearchConfig | None = None,
    initial_decoder_calls: int = 0,
    decoder_counter: Callable[[], None] | None = None,
) -> SearchOutcome:
    """Uniform random sampling of unseen states until the budget ends."""
    config = config or SearchConfig()
    config = SearchConfig(**{**config.__dict__, "mode": "random_states"})
    return run_search(
        evaluate=evaluate,
        n_regions=n_regions,
        config=config,
        initial_decoder_calls=initial_decoder_calls,
        decoder_counter=decoder_counter,
    )


def run_risk_guided(
    *,
    evaluate: Callable[[tuple[float, ...]], tuple[Score, Any]],
    n_regions: int,
    config: SearchConfig | None = None,
    initial_decoder_calls: int = 0,
    decoder_counter: Callable[[], None] | None = None,
) -> SearchOutcome:
    config = config or SearchConfig()
    config = SearchConfig(**{**config.__dict__, "mode": "risk"})
    return run_search(
        evaluate=evaluate,
        n_regions=n_regions,
        config=config,
        initial_decoder_calls=initial_decoder_calls,
    )


def run_global_scan(
    *,
    evaluate: Callable[[tuple[float, ...]], tuple[Score, Any]],
    n_regions: int,
    grid: Sequence[float] = (0.3, 0.5, 0.6, 0.7, 0.9),
    config: SearchConfig | None = None,
    initial_decoder_calls: int = 0,
    decoder_counter: Callable[[], None] | None = None,
) -> SearchOutcome:
    """Sweep the global strength over ``grid``; each state uses one alpha for
    all regions. Reports the true number of evaluated states."""
    config = config or SearchConfig()
    engine = EvaluationEngine(evaluate, config, decoder_counter=decoder_counter)
    comparator = Comparator(config.tolerance)
    initial = default_alphas(n_regions)
    default_score, default_image = engine.evaluate(
        initial, parent=None, region=None, reason="initial_default"
    )
    best_key = engine.key(initial)
    best_score, best_image = default_score, default_image
    for alpha in grid:
        if any(abs(alpha - lv) < 1e-9 for lv in config.levels) and engine.key(
            (alpha,) * n_regions
        ) == best_key:
            continue  # the default 0.6 state is evaluation #1, never repeated
        if engine.budget_left() <= 0:
            break
        alphas = (float(alpha),) * n_regions
        score, image = engine.evaluate(
            alphas, parent=None, region=None, reason=f"global_scan:{alpha}"
        )
        key = engine.key(alphas) if all(
            any(abs(a - lv) < 1e-9 for lv in config.levels) for a in alphas
        ) else None
        if key is not None and score.feasible and comparator.is_feasible_better(score, best_score):
            best_key, best_score, best_image = key, score, image
    # States outside the formal discrete domain cannot enter the archive; we
    # still report them as evaluated candidates with their real count.
    in_domain = all(
        any(abs(g - lv) < 1e-9 for lv in config.levels) for g in grid
    )
    improved = in_domain and comparator.is_feasible_better(best_score, default_score)
    used_default = not improved
    return SearchOutcome(
        best_alphas=(0.6,) * n_regions if used_default else engine.to_alphas(best_key),
        best_score=best_score,
        best_image=default_image if used_default else best_image,
        used_default=used_default,
        improved=improved,
        stop_reason="grid_complete" if in_domain else "grid_complete_mixed_domain",
        trajectory=engine.trajectory,
        candidate_count=engine.candidate_count,
        decoder_calls=initial_decoder_calls + engine.decoder_calls,
        state_cache_hits=engine.state_cache_hits,
        restarts_used=0,
    )
