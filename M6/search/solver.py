"""Constrained local-search core for M6. Deliberately torch-free.

Implements the frozen rules from M6/README.md sections 3.5, 4 and 5:

* state = per-region alphas from a fixed discrete level set,
* one-variable neighbourhood, strictly-improving acceptance with a frozen
  tolerance, feasibility-first comparator,
* state cache keyed by discrete level indices (never float tuples),
* a single shared evaluation engine so risk-guided, plain local and random
  baselines differ ONLY in their selection strategy and share comparator,
  budget accounting, cache and restart bookkeeping,
* default all-0.6 state counts as one evaluation and is the feasible fallback.

Anything torch-specific (images, VGG) is treated as an opaque payload: the
caller supplies ``evaluate(alphas) -> (Score, image)`` and images flow through
unchanged. Non-finite scores are a hard failure, never silently recorded.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


class ScoreFailure(RuntimeError):
    """A candidate produced a non-finite or otherwise invalid score."""


@dataclass(frozen=True)
class Score:
    """Objective/constraint values for one evaluated state.

    ``risks`` are per-region structural+coordination contributions used only
    for variable ordering; they are part of the scoring contract so that all
    search conditions share one scorer. ``source`` identifies the scorer for
    cache-identity checks.
    """

    Q: float
    T: float
    S: float
    J: float
    violation: float
    feasible: bool
    risks: tuple[float, ...] = ()
    source: str = ""

    def __post_init__(self) -> None:
        for name in ("Q", "T", "S", "J", "violation"):
            if not math.isfinite(getattr(self, name)):
                raise ScoreFailure(f"non-finite {name} in score")


@dataclass(frozen=True)
class Comparator:
    """Frozen feasibility-first comparator (README 3.5).

    Feasible states beat infeasible ones; infeasible states order by
    violation first, then J; feasible states order by J. A state only counts
    as a *strict improvement* over another when it wins the ordering AND the
    winning margin exceeds ``tolerance`` (tie-level moves are never accepted,
    preventing oscillation).
    """

    tolerance: float = 1e-9

    def _key(self, score: Score) -> tuple[int, float, float]:
        # J is MINIMISED: feasibility first, then smaller violation, then
        # smaller J. Lexicographic maximisation on the negated costs encodes
        # the frozen ordering in one comparable key.
        if score.feasible:
            return (1, 0.0, -score.J)
        return (0, -score.violation, -score.J)

    def is_better_than(self, new: Score, best: Score) -> bool:
        """Strictly better under the frozen ordering plus tolerance."""
        k_new, k_best = self._key(new), self._key(best)
        if k_new[0] != k_best[0] or k_new[1] != k_best[1]:
            return k_new > k_best
        # Same feasibility class (and same violation bucket): require the J
        # margin to clear the tolerance so ties never cycle.
        return (k_new[2] - k_best[2]) > self.tolerance

    def is_feasible_better(self, new: Score, best: Score) -> bool:
        """Improvement within the feasible archive (both must be feasible)."""
        if not new.feasible or not best.feasible:
            return False
        return self.is_better_than(new, best)


@dataclass(frozen=True)
class EvalRecord:
    """One entry of the trajectory; JSON-serialisable."""

    evaluation_index: int
    alphas: tuple[float, ...]
    parent_alphas: tuple[float, ...] | None
    selected_region: int | None
    selection_reason: str
    Q: float
    T: float
    S: float
    J: float
    violation: float
    feasible: bool
    accepted: bool
    is_new_best: bool
    cache_hit: bool
    restart_index: int


@dataclass
class SearchOutcome:
    """Solver-side result; adapter converts this into contracts.SearchResult."""

    best_alphas: tuple[float, ...]
    best_score: Score
    best_image: Any
    used_default: bool
    improved: bool
    stop_reason: str
    trajectory: list[EvalRecord] = field(default_factory=list)
    candidate_count: int = 0
    decoder_calls: int = 0
    state_cache_hits: int = 0
    restarts_used: int = 0

    @property
    def status(self) -> str:
        if self.used_default and not self.improved:
            return "no_improvement"
        return "improved"


@dataclass(frozen=True)
class SearchConfig:
    """Frozen search settings (README 4/5)."""

    levels: tuple[float, ...] = (0.3, 0.6, 0.9)
    budget: int = 40
    max_restarts: int = 2
    tolerance: float = 1e-9
    mode: str = "risk"  # "risk" | "random_permutation" | "random_states"
    seed: int = 0


class EvaluationEngine:
    """Budget + cache + trajectory bookkeeping shared by ALL search modes.

    Cache keys are tuples of discrete level indices, never floats, and the
    cache identity is bound to (pair, regions, scorer) by the caller passing a
    single engine instance per (method, pair) run — engines are never shared
    across methods, so no method gets another method's candidates for free.
    """

    def __init__(
        self,
        evaluate: Callable[[tuple[float, ...]], tuple[Score, Any]],
        config: SearchConfig,
        decoder_counter: Callable[[], int] | None = None,
    ) -> None:
        self._evaluate = evaluate
        self.config = config
        self.levels = tuple(float(v) for v in config.levels)
        self._level_index = {v: i for i, v in enumerate(self.levels)}
        self._cache: dict[tuple[int, ...], tuple[Score, Any]] = {}
        self._records: list[EvalRecord] = []
        self._budget_used = 0
        self._decoder_calls = 0
        self._cache_hits = 0
        self._external_decoder_counter = decoder_counter
        self._restart_index = 0

    # -- helpers ---------------------------------------------------------

    @property
    def key(self) -> Callable[[Sequence[float]], tuple[Any, ...]]:
        # Discrete level indices for formal-domain states; off-domain values
        # (global-scan midpoints) fall back to a tagged float so they never
        # collide with a formal state and never enter the formal archive.
        index = self._level_index

        def make(alphas: Sequence[float]) -> tuple[Any, ...]:
            parts: list[Any] = []
            for a in alphas:
                a = float(a)
                parts.append(index[a] if a in index else ("off", a))
            return tuple(parts)

        return make

    def to_alphas(self, key: Sequence[int]) -> tuple[float, ...]:
        return tuple(self.levels[i] for i in key)

    # -- bookkeeping -----------------------------------------------------

    @property
    def candidate_count(self) -> int:
        return self._budget_used

    @property
    def decoder_calls(self) -> int:
        return self._decoder_calls

    @property
    def state_cache_hits(self) -> int:
        return self._cache_hits

    @property
    def trajectory(self) -> list[EvalRecord]:
        return list(self._records)

    def budget_left(self) -> int:
        return self.config.budget - self._budget_used

    # -- core ------------------------------------------------------------

    def evaluate(self, alphas: Sequence[float], *, parent, region, reason) -> tuple[Score, Any]:
        """Evaluate one state honouring cache and budget rules (README 5.3/5.5)."""
        alphas = tuple(float(a) for a in alphas)
        key = self.key(alphas)
        if key in self._cache:
            self._cache_hits += 1
            score, image = self._cache[key]
            self._records.append(
                self._record(alphas, parent, region, reason, score, False, False, True)
            )
            return score, image
        if self._budget_used >= self.config.budget:
            raise BudgetExhausted(
                f"budget {self.config.budget} exhausted; refusing to evaluate new state"
            )
        score, image = self._evaluate(alphas)
        if not isinstance(score, Score):
            raise ScoreFailure(f"scorer returned {type(score).__name__}, expected Score")
        if self._external_decoder_counter is not None:
            self._external_decoder_counter()  # adapter counts REAL generate calls
        else:
            self._decoder_calls += 1  # standalone use: 1 decode per new state
        self._budget_used += 1
        self._cache[key] = (score, image)
        self._records.append(self._record(alphas, parent, region, reason, score, False, False, False))
        return score, image

    def _record(self, alphas, parent, region, reason, score, accepted, new_best, cache_hit):
        return EvalRecord(
            evaluation_index=len(self._records) + 1,
            alphas=tuple(alphas),
            parent_alphas=None if parent is None else tuple(parent),
            selected_region=region,
            selection_reason=reason,
            Q=score.Q,
            T=score.T,
            S=score.S,
            J=score.J,
            violation=score.violation,
            feasible=score.feasible,
            accepted=accepted,
            is_new_best=new_best,
            cache_hit=cache_hit,
            restart_index=self._restart_index,
        )

    def mark_accepted(self, index: int, new_best: bool) -> None:
        record = self._records[index]
        self._records[index] = EvalRecord(**{**record.__dict__, "accepted": True, "is_new_best": new_best})


class BudgetExhausted(RuntimeError):
    """Internal control flow: no budget left for a NEW state."""


def default_alphas(n_regions: int) -> tuple[float, ...]:
    """The agreed initial state: global 0.6 (handoff contract)."""
    return (0.6,) * n_regions


def run_search(
    *,
    evaluate: Callable[[tuple[float, ...]], tuple[Score, Any]],
    n_regions: int,
    initial: Sequence[float] | None = None,
    config: SearchConfig | None = None,
    initial_decoder_calls: int = 0,
    decoder_counter: Callable[[], None] | None = None,
) -> SearchOutcome:
    """Run one search condition under the frozen rules.

    ``evaluate`` must return ``(Score, image)`` for a full state; the engine
    adds cache/budget bookkeeping. ``decoder_counter`` (if given) is invoked
    once per REAL decode so adapters can count actual generate calls instead
    of assuming one decode per new state. ``initial_decoder_calls`` records
    decode calls already spent before the search (the provided default).
    """
    config = config or SearchConfig()
    if config.mode not in {"risk", "random_permutation", "random_states"}:
        raise ValueError(f"unknown mode {config.mode!r}")
    engine = EvaluationEngine(evaluate, config, decoder_counter=decoder_counter)
    rng = random.Random(config.seed)
    n_regions = int(n_regions)
    initial_key_alphas = tuple(initial) if initial is not None else default_alphas(n_regions)

    # README 5.1: the default state is evaluation #1 and enters the archive.
    default_score, default_image = engine.evaluate(
        initial_key_alphas, parent=None, region=None, reason="initial_default"
    )
    best_key = engine.key(initial_key_alphas)
    best_score, best_image = default_score, default_image
    current_key = best_key
    current_score = default_score
    restarts_used = 0
    stop_reason = ""  # set only on real termination causes; never pre-armed

    def archive_update(key, score, image):
        nonlocal best_key, best_score, best_image
        if score.feasible and Comparator(config.tolerance).is_feasible_better(score, best_score):
            best_key, best_score, best_image = key, score, image
            return True
        return False

    archive_update(best_key, best_score, best_image)

    def random_unseen_key() -> tuple[int, ...] | None:
        """Uniform sample over states not in the cache (README 6, random baseline)."""
        space = len(config.levels) ** n_regions
        if len(engine._cache) >= space:
            return None
        for _ in range(64 * space):
            key = tuple(rng.randrange(len(config.levels)) for _ in range(n_regions))
            if key not in engine._cache:
                return key
        return None

    if config.mode == "random_states":
        # README 6: uniform sampling of unseen states until budget ends; the
        # default still counts as one evaluation and the feasible archive is kept.
        while engine.budget_left() > 0:
            key = random_unseen_key()
            if key is None:
                stop_reason = "space_exhausted"
                break
            alphas = engine.to_alphas(key)
            score, image = engine.evaluate(alphas, parent=None, region=None, reason="random_sample")
            archive_update(key, score, image)
            current_key, current_score = key, score
        stop_reason = stop_reason if stop_reason == "space_exhausted" else "budget_exhausted"
    else:
        # risk-guided / plain local search: identical except variable ordering.
        tried: set[int] = set()
        while True:
            if config.mode == "risk":
                # Descending risk, ties by region id (README 4). Regions already
                # tried for the CURRENT state are skipped.
                risks = current_score.risks
                if len(risks) != n_regions:
                    raise ScoreFailure(
                        "risk ordering requires per-region risks from the scorer"
                    )
                order = sorted(
                    (i for i in range(n_regions) if i not in tried),
                    key=lambda i: (-risks[i], i),
                )
                reason_base = "risk_desc"
            else:
                order = [i for i in range(n_regions) if i not in tried]
                rng.shuffle(order)
                reason_base = "seeded_random_permutation"

            moved = False
            for region in order:
                if engine.budget_left() <= 0:
                    stop_reason = "budget_exhausted"
                    break
                cur_alphas = engine.to_alphas(current_key)
                cur_level = engine._level_index[cur_alphas[region]]
                others = [lv for li, lv in enumerate(config.levels) if li != cur_level]
                # README 5.3: the other two levels in ascending order; with only
                # one budget left, evaluate just the first and log truncation.
                candidates: list[tuple[tuple[int, ...], Score, Any, int]] = []
                truncated = False
                for k, level in enumerate(others):
                    if engine.budget_left() <= 0:
                        truncated = k == 0
                        break
                    nxt = list(cur_alphas)
                    nxt[region] = level
                    nxt = tuple(nxt)
                    record_index = len(engine._records)
                    score, image = engine.evaluate(
                        nxt, parent=cur_alphas, region=region,
                        reason=f"{reason_base}:move_region_{region}",
                    )
                    candidates.append((engine.key(nxt), score, image, record_index))
                if truncated:
                    engine._records.append(
                        engine._record(
                            cur_alphas, cur_alphas, region,
                            "truncated_no_budget", current_score, False, False, False,
                        )
                    )
                if not candidates:
                    tried.add(region)
                    continue
                # README 5.4: pick strictly-best among current + evaluated neighbours.
                comparator = Comparator(config.tolerance)
                pick_key, pick_score, pick_image, pick_index = current_key, current_score, None, -1
                for key, score, image, record_index in candidates:
                    if comparator.is_better_than(score, pick_score):
                        pick_key, pick_score, pick_image, pick_index = key, score, image, record_index
                if pick_key != current_key and comparator.is_better_than(pick_score, current_score):
                    new_best = archive_update(pick_key, pick_score, pick_image)
                    engine.mark_accepted(pick_index, new_best)
                    current_key, current_score = pick_key, pick_score
                    tried.clear()
                    moved = True
                    break
                tried.add(region)
            if stop_reason == "budget_exhausted":
                break
            if moved:
                continue
            # Full neighbourhood without strict improvement (README 5.6).
            if restarts_used < config.max_restarts and engine.budget_left() > 0:
                start = random_unseen_key()
                if start is None:
                    stop_reason = "space_exhausted"
                    break
                restarts_used += 1
                engine._restart_index = restarts_used
                alphas = engine.to_alphas(start)
                score, image = engine.evaluate(
                    alphas, parent=engine.to_alphas(current_key), region=None,
                    reason=f"restart_{restarts_used}",
                )
                archive_update(start, score, image)
                current_key, current_score = start, score
                tried.clear()
                continue
            stop_reason = "local_optimum"
            break

    improved = (
        best_key != engine.key(initial_key_alphas)
        and Comparator(config.tolerance).is_feasible_better(best_score, default_score)
    )
    used_default = not improved
    return SearchOutcome(
        best_alphas=engine.to_alphas(best_key),
        best_score=best_score,
        best_image=default_image if used_default else best_image,
        used_default=used_default,
        improved=improved,
        stop_reason=stop_reason,
        trajectory=engine.trajectory,
        candidate_count=engine.candidate_count,
        decoder_calls=initial_decoder_calls + engine.decoder_calls,
        state_cache_hits=engine.state_cache_hits,
        restarts_used=restarts_used,
    )
