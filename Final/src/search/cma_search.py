"""Joint continuous CMA-ES with a feasible archive and a shared hard budget."""

import time
from dataclasses import asdict

import numpy as np

from src.contracts import SearchResult
from src.models.compensation import ContentGeometry, LocalCompensator

from .aesthetics import AestheticMetric
from .landscape import LandscapeConfig, UniformityMetric
from .objective import PairScorer, ScorerConfig, VGG19Evaluator

CMA_METHODS = ("cma_legacy", "cma_landscape", "cma_compensated")


def optimize(
    evaluate,
    n_regions,
    *,
    budget=80,
    seed=0,
    compensation=True,
    sigma=0.14,
    minimum_improvement=0.002,
    progress_callback=None,
):
    """evaluate(theta) -> (metrics, image). All evaluated states count, including warmup.

    Rank whole populations lexicographically by feasibility/violation/objective.
    CMA-ES uses ranks, so no arbitrary scalar penalty mixes incompatible scales.
    The last incomplete population is evaluated/archived but not used in tell().
    """
    import cma

    if budget < 1 or n_regions < 1 or not 0 < sigma < 1:
        raise ValueError("Invalid CMA-ES budget, dimension or sigma")
    dimension = n_regions + int(compensation)
    initial = (0.6,) * n_regions + ((0.0,) if compensation else ())
    records, cache = [], {}
    best = None
    cache_hits = 0
    started = time.perf_counter()

    def visit(theta, reason, generation):
        nonlocal best, cache_hits
        key = tuple(float(x) for x in theta)
        if len(key) != dimension or any(not np.isfinite(x) or x < 0 or x > 1 for x in key):
            raise ValueError("CMA candidate outside [0,1]")
        if key in cache:
            cache_hits += 1
            return cache[key]
        metrics, image = evaluate(key)
        if any(not np.isfinite(v) for v in metrics.values() if v is not None):
            raise ValueError("Nonfinite candidate score")
        is_best = metrics["feasible"] and (best is None or metrics["J"] < best[1]["J"] - 1e-9)
        if is_best:
            best = (key, metrics, image)
        record = {
            "evaluation_index": len(records) + 1,
            "log_index": len(records),
            "alphas": list(key[:n_regions]),
            "beta": key[-1] if compensation else 0.0,
            "parent_alphas": None,
            "selected_region": None,
            "selection_reason": reason,
            "generation": generation,
            **metrics,
            "accepted": bool(is_best),
            "is_new_best": bool(is_best),
            "cache_hit": False,
            "restart_index": 0,
        }
        records.append(record)
        cache[key] = metrics
        if progress_callback:
            progress_callback(
                {
                    "event": "candidate_scored",
                    "candidate_count": len(records),
                    "budget": budget,
                    "generation": generation,
                    "elapsed_s": time.perf_counter() - started,
                }
            )
        return metrics

    baseline_metrics = visit(initial, "initial", 0)
    if not baseline_metrics["feasible"]:
        raise ValueError("The supplied AdaIN baseline must be feasible")
    baseline = best
    warmup = [(a,) * n_regions + ((0.0,) if compensation else ()) for a in (0.45, 0.75)]
    if compensation:
        warmup += [(0.6,) * n_regions + (b,) for b in (0.5, 1.0)]
    for state in warmup:
        if len(records) >= budget:
            break
        visit(state, "warmup", 0)
    stop_reason = "budget_exhausted"
    generations = 0
    if len(records) < budget:
        rng = np.random.RandomState(seed)
        es = cma.CMAEvolutionStrategy(
            np.clip(best[0], 0.05, 0.95).tolist(),
            sigma,
            {
                "bounds": [0.0, 1.0],
                "seed": seed,
                "randn": rng.randn,
                "verbose": -9,
                "verb_log": 0,
                # Rank values repeat every generation by construction. Objective-value
                # stopping tests would falsely declare convergence after ten generations.
                "tolfun": 0,
                "tolfunhist": 0,
                "tolflatfitness": 1000000000,
                "tolstagnation": 1000000000,
                "popsize": 4 + int(3 * np.log(dimension)),
            },
        )
        while len(records) < budget:
            if es.stop():
                stop_reason = "cma_converged"
                break
            generations += 1
            population = es.ask()
            measured, vectors = [], []
            for theta in population:
                if len(records) >= budget:
                    break
                measured.append(visit(theta, "cma_population", generations))
                vectors.append(theta)
            if len(vectors) != len(population):
                break
            order = sorted(
                range(len(measured)),
                key=lambda i: (
                    0 if measured[i]["feasible"] else 1,
                    0 if measured[i]["feasible"] else measured[i]["violation"],
                    measured[i]["J"],
                ),
            )
            ranks = np.empty(len(order))
            ranks[order] = np.arange(len(order))
            es.tell(vectors, ranks.tolist())
    used_default = baseline[1]["J"] - best[1]["J"] < minimum_improvement or best[0] == initial
    if used_default:
        best = baseline
    return {
        "best": best,
        "trajectory": tuple(records),
        "used_default": used_default,
        "status": "no_improvement" if used_default else "improved",
        "stop_reason": stop_reason,
        "cache_hits": cache_hits,
        "generations": generations,
    }


def solve_cma(
    *,
    generate,
    pair,
    regions,
    initial_result,
    search_config,
    reconstruct=None,
    progress_callback=None,
):
    from .adapter import _style_tensor

    cfg = dict(search_config)
    method = cfg["method"]
    compensation = method == "cma_compensated"
    landscape = method != "cma_legacy"
    scoring = ScorerConfig(**cfg.get("scoring", {}))
    new_config = LandscapeConfig(**cfg.get("landscape", {}))
    scorer = PairScorer(
        VGG19Evaluator(
            content_tensor=pair.content_tensor,
            style_tensor=_style_tensor(pair),
            soft_masks=regions.soft_masks,
            edges=regions.edges,
            config=scoring,
            device=pair.content_tensor.device,
        )
    )
    scorer.calibrate(initial_result.image_rgb)
    geometry = ContentGeometry(pair.content_tensor, regions.labels) if landscape else None
    uniformity = UniformityMetric(geometry) if landscape else None
    aesthetic = AestheticMetric(pair.content_tensor.device) if landscape else None
    correction = None
    decode_count = 1  # initial already rendered by caller
    if compensation:
        if reconstruct is None:
            raise ValueError("Compensated search requires the true alpha=0 reconstruction")
        correction = LocalCompensator(geometry, reconstruct().image_rgb)
        decode_count += 1
    n_regions = len(regions.region_ids)

    def evaluate(theta):
        nonlocal decode_count
        alphas = tuple(theta[:n_regions])
        beta = float(theta[-1]) if compensation else 0.0
        if alphas == (0.6,) * n_regions:
            image = initial_result.image_rgb
        else:
            image = generate(alphas).image_rgb
            decode_count += 1
        if correction is not None:
            image = correction(image, beta)
        old = scorer(image, alphas)
        metrics = {k: getattr(old, k) for k in ("Q", "T", "S", "J", "violation", "feasible")}
        metrics.update({"A": None, "U": None, "U_L": None, "U_ab": None})
        if landscape:
            metrics.update(uniformity(image))
            metrics["A"] = aesthetic(image)
            metrics["J"] = new_config.objective(
                metrics["A"], metrics["U"], old.Q, scorer.q0, scoring.delta
            )
        return metrics, image

    budget = int(cfg.get("budget", 80))
    started = time.perf_counter()
    if progress_callback:
        progress_callback({"event": "search_start", "method": method, "budget": budget})
    outcome = optimize(
        evaluate,
        n_regions,
        budget=budget,
        seed=int(cfg.get("seed", 0)),
        compensation=compensation,
        sigma=float(cfg.get("sigma", 0.14)),
        minimum_improvement=new_config.minimum_improvement if landscape else 1e-9,
        progress_callback=progress_callback,
    )
    theta, measured, image = outcome["best"]
    metrics = {k: v for k, v in measured.items() if k not in ("feasible", "violation")}
    metadata = {
        "used_default": outcome["used_default"],
        "feasible": measured["feasible"],
        "violation": measured["violation"],
        "stop_reason": outcome["stop_reason"],
        "method": method,
        "seed": int(cfg.get("seed", 0)),
        "budget": budget,
        "generations": outcome["generations"],
        "restarts_used": 0,
        "beta": float(theta[-1]) if compensation else 0.0,
        "continuous_alphas": True,
        "alpha_bounds": [0.0, 1.0],
        "q0": scorer.q0,
        "t0": scorer.t0,
        "scorer": scorer.provenance(),
        "landscape": asdict(new_config) if landscape else None,
        "aesthetic": aesthetic.provenance if landscape else None,
        "uniformity_version": "content-fixed-lab-multiscale-v1" if landscape else None,
        "m6_version": "landscape-v3",
        "decoder_count_includes_initial": True,
        "reconstruction_calls": int(compensation),
        "elapsed_search_s": time.perf_counter() - started,
        "calibration": cfg.get("calibration", "pilot_pending"),
    }
    if progress_callback:
        progress_callback({"event": "search_end", "status": outcome["status"]})
    return SearchResult(
        image,
        tuple(theta[:n_regions]),
        outcome["status"],
        metrics,
        outcome["trajectory"],
        len(outcome["trajectory"]),
        decode_count,
        outcome["cache_hits"],
        metadata,
    )
