"""Search baselines share the same scorer, domain, cache and candidate budget."""

from __future__ import annotations

from .solver import (
    Comparator,
    EvaluationEngine,
    SearchConfig,
    SearchOutcome,
    default_alphas,
    run_search,
)


def _run(mode, *, evaluate, n_regions, config=None, initial_decoder_calls=0, decoder_counter=None):
    config = config or SearchConfig()
    return run_search(
        evaluate=evaluate,
        n_regions=n_regions,
        config=SearchConfig(**{**config.__dict__, "mode": mode}),
        initial_decoder_calls=initial_decoder_calls,
        decoder_counter=decoder_counter,
    )


def run_plain_local(**kwargs):
    return _run("random_permutation", **kwargs)


def run_random_states(**kwargs):
    return _run("random_states", **kwargs)


def run_risk_guided(**kwargs):
    return _run("risk", **kwargs)


def run_global_scan(
    *,
    evaluate,
    n_regions,
    grid=(0.3, 0.6, 0.9),
    config=None,
    initial_decoder_calls=0,
    decoder_counter=None,
):
    config = config or SearchConfig()
    grid = tuple(float(a) for a in grid)
    if n_regions < 1 or not grid or any(a not in config.levels for a in grid):
        raise ValueError("Global scan grid must belong to the configured formal levels")
    engine = EvaluationEngine(evaluate, config, decoder_counter)
    comparator = Comparator(config.tolerance)
    initial = default_alphas(n_regions)
    default_score, default_image = engine.evaluate(
        initial, parent=None, region=None, reason="initial_default"
    )
    best_alphas, best_score, best_image = initial, default_score, default_image
    seen = {initial}
    stop = "grid_complete"
    for alpha in grid:
        alphas = (alpha,) * n_regions
        if alphas in seen:
            continue
        if engine.budget_left() <= 0:
            stop = "budget_exhausted"
            break
        seen.add(alphas)
        score, image = engine.evaluate(
            alphas, parent=None, region=None, reason=f"global_scan:{alpha}"
        )
        if comparator.is_feasible_better(score, best_score):
            best_alphas, best_score, best_image = alphas, score, image
            engine.mark_best(len(engine._records) - 1)
    improved = comparator.is_feasible_better(best_score, default_score)
    return SearchOutcome(
        best_alphas=best_alphas if improved else initial,
        best_score=best_score if improved else default_score,
        best_image=best_image if improved else default_image,
        used_default=not improved,
        improved=improved,
        stop_reason=stop,
        trajectory=engine.trajectory,
        candidate_count=engine.candidate_count,
        decoder_calls=initial_decoder_calls + engine.decoder_calls,
        state_cache_hits=engine.state_cache_hits,
        restarts_used=0,
    )
