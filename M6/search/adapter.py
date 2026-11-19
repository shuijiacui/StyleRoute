"""Adapter entry point registered in resources/configs/team.json as
``M6.search.adapter:solve``. This is the only symbol M4 imports.

The adapter wires the real evaluator + solver and converts the solver's
SearchOutcome into ``src.contracts.SearchResult`` with full provenance. It
imports torch and ``src.contracts`` — those are only available in the M4
integration environment (PYTHONPATH = repo root + M4).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

# Integration environment: repo root (for M6/M5) and M4/ (for src.*) on path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "M4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from src.contracts import SearchResult  # noqa: E402

from . import baselines, objective, solver
from .solver import Comparator, Score, SearchConfig


def _resolve_method(name: str):
    return {
        "risk": baselines.run_risk_guided,
        "plain": baselines.run_plain_local,
        "random_states": baselines.run_random_states,
        "global_scan": baselines.run_global_scan,
    }[name]


def solve(
    *,
    generate: Callable[[tuple[float, ...]], Any],
    pair: Any,
    regions: Any,
    initial_result: Any,
    search_config: dict | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> SearchResult:
    """Real M6 entry point. ``generate`` decodes one candidate without
    re-encoding; the default 0.6 image is reused, never regenerated."""
    cfg_dict = dict(search_config or {})
    method = str(cfg_dict.get("method", "risk"))
    config = SearchConfig(
        levels=tuple(float(v) for v in cfg_dict.get("levels", (0.3, 0.6, 0.9))),
        budget=int(cfg_dict.get("budget", 40)),
        max_restarts=int(cfg_dict.get("max_restarts", 2)),
        tolerance=float(cfg_dict.get("tolerance", 1e-9)),
        mode="risk",  # replaced per method below
        seed=int(cfg_dict.get("seed", 0)),
    )
    scorer_cfg = objective.ScorerConfig(
        delta=float(cfg_dict.get("delta", 1e-3)),
        delta_t=float(cfg_dict.get("delta_t", 1e-6)),
        lam=float(cfg_dict.get("lam", 1.0)),
        epsilon=float(cfg_dict.get("epsilon", 0.05)),
        tolerance=config.tolerance,
        weights=str(cfg_dict.get("weights", "IMAGENET1K_V1")),
    )

    n_regions = len(regions.region_ids)
    default_alphas = (0.6,) * n_regions

    # Independent evaluator: its own VGG19 + ImageNet normalisation. The
    # default image is scored through the same path as every candidate; q0/t0
    # therefore describe the real default result.
    evaluator = objective.VGG19Evaluator(
        content_tensor=pair.content_tensor,
        style_tensor=_style_tensor(pair),
        soft_masks=regions.soft_masks,
        edges=regions.edges,
        config=scorer_cfg,
        device=pair.content_tensor.device,
    )
    scorer = objective.PairScorer(evaluator)
    scorer.calibrate(initial_result.image_rgb)

    decode_count = {"n": 0}  # real generate() calls; default result is reused, not decoded

    def evaluate(alphas: tuple[float, ...]) -> tuple[Score, torch.Tensor]:
        t0 = time.perf_counter()
        if all(abs(a - d) < 1e-9 for a, d in zip(alphas, default_alphas)):
            image = initial_result.image_rgb  # cache/budget still handled by engine
        else:
            image = generate(tuple(float(a) for a in alphas)).image_rgb
            decode_count["n"] += 1
        score = scorer(image, alphas)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "candidate_scored",
                    "alphas": list(alphas),
                    "score_ms": elapsed_ms,
                    "decoder_calls": 1 + decode_count["n"],
                }
            )
        return score, image

    if progress_callback is not None:
        progress_callback({"event": "search_start", "method": method, "budget": config.budget})

    runner = _resolve_method(method)
    if method == "global_scan":
        outcome = runner(
            evaluate=evaluate,
            n_regions=n_regions,
            grid=tuple(float(v) for v in cfg_dict.get("global_grid", (0.3, 0.5, 0.6, 0.7, 0.9))),
            config=SearchConfig(**{**config.__dict__, "budget": max(config.budget, 16)}),
            initial_decoder_calls=1,
        )
    else:
        outcome = runner(
            evaluate=evaluate,
            n_regions=n_regions,
            config=SearchConfig(**{**config.__dict__, "mode": {
                "risk": "risk", "plain": "random_permutation",
                "random_states": "random_states",
            }[method]}),
            initial_decoder_calls=1,
        )

    decoder_calls_total = 1 + decode_count["n"]  # provided default + real generate calls

    metrics: dict[str, float | None] = {
        "Q": float(outcome.best_score.Q),
        "T": float(outcome.best_score.T),
        "S": float(outcome.best_score.S),
        "J": float(outcome.best_score.J),
    }
    metadata: dict[str, Any] = {
        "used_default": bool(outcome.used_default),
        "feasible": bool(outcome.best_score.feasible),
        "violation": float(outcome.best_score.violation),
        "stop_reason": outcome.stop_reason,
        "method": method,
        "seed": config.seed,
        "budget": config.budget,
        "max_restarts": config.max_restarts,
        "decoder_count_includes_initial": True,
        "restarts_used": outcome.restarts_used,
        "q0": scorer.q0,
        "t0": scorer.t0,
        "scorer": scorer.provenance(),
        "m6_version": _m6_version(),
    }
    if progress_callback is not None:
        progress_callback({"event": "search_end", "status": outcome.status})

    return SearchResult(
        best_image_rgb=outcome.best_image,
        region_alphas=tuple(float(a) for a in outcome.best_alphas),
        status=outcome.status,
        metrics=metrics,
        trajectory=tuple(record.__dict__ for record in outcome.trajectory),
        candidate_count=int(outcome.candidate_count),
        decoder_calls=int(decoder_calls_total),
        state_cache_hits=int(outcome.state_cache_hits),
        metadata=metadata,
    )


def _style_tensor(pair: Any) -> torch.Tensor:
    """Build the evaluator style input from the preprocessed PIL reference.

    Uses the same canonical conversion as M4 (``image_tensor``) then lets the
    evaluator apply its own resize/normalisation internally — never mutating
    shared tensors in place.
    """
    from src.models.preprocessing import image_tensor

    return image_tensor(pair.style_rgb).to(pair.content_tensor.device)


def _m6_version() -> str:
    from M6 import __version__

    return __version__
