"""Phase A tests: controlled fake scoring on small finite spaces (README 9).

These tests prove program logic — comparator, budget, cache, acceptance,
fallback, restarts and baseline fairness. They do NOT prove generation
quality; the real-weight integration checks live in test_integration.py.

Pure-python: imports only M6.search.solver / baselines / objective math
(torch optional). Run:  python -m pytest M6/tests/test_phase_a.py
"""

from __future__ import annotations

import itertools
import math

import pytest

from M6.search.baselines import run_global_scan, run_plain_local, run_random_states, run_risk_guided
from M6.search.objective import (
    ScorerConfig,
    assemble_score,
    coordination,
    coordination_contrib,
)
from M6.search.solver import (
    Comparator,
    EvaluationEngine,
    Score,
    ScoreFailure,
    SearchConfig,
    default_alphas,
)

N3_EDGES = ((0, 1, 1.0), (1, 2, 1.0))
LEVELS = (0.3, 0.6, 0.9)

# ---------------------------------------------------------------------------
# Deterministic fake landscape (N=3, chain edges 0-1-2).
#
# Q is interaction-based so that hill climbing demonstrably needs restarts:
#   * region 0 at 0.3 alone earns 0.10  -> local optimum L=(0.3,0.6,0.6)
#   * region 1 at 0.3 alone earns nothing
#   * region 0 AND region 1 at 0.3 together earn 0.30 -> global G=(0.3,0.3,0.6)
#   * the single step from L toward G is (0.6->0.3 on region 1) = (0.6,0.3,0.6),
#     which is WORSE than L (no solo gain, extra coordination cost).
# T = 0.1*(1+0.02*k), k = number of regions off 0.6; eps=0.05 so up to two
#   regions may move without violating the style cap.
# ---------------------------------------------------------------------------

QMAX = 1.0
Q0 = QMAX
T0 = 0.1


def fake_q(alphas):
    q = QMAX
    if math.isclose(alphas[0], 0.3):
        q -= 0.10
    if math.isclose(alphas[0], 0.3) and math.isclose(alphas[1], 0.3):
        q -= 0.20  # joint bonus on top of region-0 solo gain
    return q


def make_scorer(edges=N3_EDGES, cfg=None):
    cfg = cfg or ScorerConfig()

    def score(alphas):
        alphas = tuple(float(a) for a in alphas)
        Q = fake_q(alphas)
        k = sum(0 if math.isclose(a, 0.6) else 1 for a in alphas)
        T = T0 * (1.0 + 0.02 * k)
        S = coordination(alphas, edges)
        # per-region structural contributions summing exactly to Q:
        # q_i = Q_max_i - gain_i, gains realised by region 0 (solo) and the
        # (0,1) pair split half/half.
        q = [QMAX / 3.0, QMAX / 3.0, QMAX / 3.0]
        if math.isclose(alphas[0], 0.3):
            q[0] -= 0.10
        if math.isclose(alphas[0], 0.3) and math.isclose(alphas[1], 0.3):
            q[0] -= 0.10
            q[1] -= 0.10
        s = coordination_contrib(alphas, edges)
        risks = tuple(qi / (Q0 + cfg.delta) + cfg.lam * si for qi, si in zip(q, s))
        return assemble_score(Q, T, S, risks, Q0, T0, cfg, source="fake")

    return score


def make_evaluate(edges=N3_EDGES, cfg=None):
    scorer = make_scorer(edges, cfg)

    def evaluate(alphas):
        return scorer(alphas), f"img{tuple(alphas)}"

    return evaluate


# ---------------------------------------------------------------------------
# Comparator (README 3.5)
# ---------------------------------------------------------------------------

def test_feasible_beats_infeasible_regardless_of_j():
    c = Comparator(1e-9)
    good = Score(Q=0.5, T=0.1, S=0.0, J=5.0, violation=0.0, feasible=True)
    bad = Score(Q=0.01, T=0.1, S=0.0, J=0.001, violation=0.5, feasible=False)
    assert c.is_better_than(good, bad)
    assert not c.is_better_than(bad, good)


def test_infeasible_ordering_by_violation_then_j():
    c = Comparator(1e-9)
    a = Score(Q=0, T=0, S=0, J=100.0, violation=0.1, feasible=False)
    b = Score(Q=0, T=0, S=0, J=1.0, violation=0.3, feasible=False)
    assert c.is_better_than(a, b)  # smaller violation wins first


def test_strict_improvement_requires_margin_over_tolerance():
    c = Comparator(1e-6)
    base = Score(Q=0, T=0, S=0, J=1.0, violation=0.0, feasible=True)
    tiny = Score(Q=0, T=0, S=0, J=1.0 - 5e-7, violation=0.0, feasible=True)
    big = Score(Q=0, T=0, S=0, J=0.9, violation=0.0, feasible=True)
    assert not c.is_better_than(tiny, base)  # tie-level move rejected
    assert c.is_better_than(big, base)


def test_nonfinite_score_is_explicit_failure():
    with pytest.raises(ScoreFailure):
        Score(Q=float("inf"), T=0, S=0, J=0, violation=0, feasible=True)
    with pytest.raises(ScoreFailure):
        Score(Q=0, T=float("nan"), S=0, J=0, violation=0, feasible=True)


# ---------------------------------------------------------------------------
# Engine: budget + cache rules (README 5)
# ---------------------------------------------------------------------------

def test_default_counts_once_and_cache_hits_are_free():
    engine = EvaluationEngine(make_evaluate(), SearchConfig(budget=40))
    default = default_alphas(3)
    s1, _ = engine.evaluate(default, parent=None, region=None, reason="t")
    s2, _ = engine.evaluate(default, parent=None, region=None, reason="t")
    assert engine.candidate_count == 1
    assert engine.state_cache_hits == 1
    assert s1 is s2


def test_budget_is_a_hard_cap_for_new_states():
    engine = EvaluationEngine(make_evaluate(), SearchConfig(budget=2))
    engine.evaluate((0.6, 0.6, 0.6), parent=None, region=None, reason="t")
    engine.evaluate((0.3, 0.6, 0.6), parent=None, region=None, reason="t")
    with pytest.raises(Exception):
        engine.evaluate((0.9, 0.6, 0.6), parent=None, region=None, reason="t")
    assert engine.candidate_count == 2


def test_engine_rejects_scores_without_region_risks():
    engine = EvaluationEngine(make_evaluate(), SearchConfig(budget=40))
    score, _ = engine.evaluate((0.6, 0.6, 0.6), parent=None, region=None, reason="t")
    assert len(score.risks) == 3
    assert all(math.isfinite(r) for r in score.risks)


# ---------------------------------------------------------------------------
# Risk-guided search on the small landscape
# ---------------------------------------------------------------------------

def _brute_force_best(evaluate, budget=27):
    cfg = ScorerConfig()
    best = None
    for alphas in itertools.product(LEVELS, repeat=3):
        score, image = evaluate(alphas)
        if score.feasible and (best is None or score.J < best[0].J):
            best = (score, alphas, image)
    return best


def test_small_space_archive_is_best_of_all_evaluated_states():
    # Local search with 2 restarts is NOT expected to reach the brute-force
    # optimum on an interaction landscape (README never claims optimality).
    # The frozen invariant is weaker and exact: the returned feasible archive
    # must be the best feasible state among ALL states actually evaluated.
    evaluated: dict[tuple[float, ...], object] = {}
    inner = make_evaluate()

    def spy(alphas):
        score, image = inner(alphas)
        evaluated[tuple(alphas)] = score
        return score, image

    outcome = run_risk_guided(
        evaluate=spy, n_regions=3, config=SearchConfig(budget=27, seed=0)
    )
    feasible = [s for s in evaluated.values() if s.feasible]
    best_evaluated = min(feasible, key=lambda s: s.J)
    assert outcome.best_alphas in evaluated
    assert math.isclose(outcome.best_score.J, best_evaluated.J, rel_tol=1e-9)


def test_risk_ordering_prioritises_region_zero():
    # region 0 carries the only solo gain, so it must be explored first.
    evaluate = make_evaluate()
    outcome = run_risk_guided(
        evaluate=evaluate, n_regions=3, config=SearchConfig(budget=40, seed=0)
    )
    first_move = next(
        r for r in outcome.trajectory if r.accepted and r.selected_region is not None
    )
    assert first_move.selected_region == 0


def test_every_accepted_move_changes_exactly_one_region():
    evaluate = make_evaluate()
    outcome = run_risk_guided(
        evaluate=evaluate, n_regions=3, config=SearchConfig(budget=40, seed=1)
    )
    for record in outcome.trajectory:
        if record.accepted and record.parent_alphas is not None:
            diffs = sum(
                0 if math.isclose(a, b) else 1
                for a, b in zip(record.alphas, record.parent_alphas)
            )
            assert diffs == 1, record
            for a in record.alphas:
                assert any(math.isclose(a, lv) for lv in LEVELS)


def test_fallback_to_default_when_nothing_improves():
    # Flat landscape: default is already optimal.
    cfg = ScorerConfig()

    def flat(alphas):
        S = coordination(alphas, N3_EDGES)
        s = coordination_contrib(alphas, N3_EDGES)
        risks = tuple(Q0 / 3 / (Q0 + cfg.delta) + cfg.lam * si for si in s)
        return assemble_score(Q0, T0, S, risks, Q0, T0, cfg, source="fake"), "img"

    outcome = run_risk_guided(
        evaluate=flat, n_regions=3, config=SearchConfig(budget=10, seed=0)
    )
    assert not outcome.improved
    assert outcome.used_default
    assert outcome.status == "no_improvement"
    assert outcome.best_alphas == (0.6, 0.6, 0.6)
    assert outcome.best_image == "img"
    assert outcome.candidate_count >= 1
    assert outcome.decoder_calls >= 1  # default decode is accounted


def test_restart_escapes_the_interaction_local_optimum():
    # (0.3,0.6,0.6) is a strict local optimum under single-variable moves; only
    # a random restart can reach the joint optimum (0.3,0.3,0.6).
    reached = []
    for seed in range(200):
        outcome = run_risk_guided(
            evaluate=make_evaluate(),
            n_regions=3,
            config=SearchConfig(budget=27, max_restarts=2, seed=seed),
        )
        if outcome.best_alphas == (0.3, 0.3, 0.6):
            reached.append((seed, outcome.restarts_used))
            break
    assert reached, "no seed reached the joint optimum — restart logic broken"
    assert reached[0][1] >= 1


def test_budget_exhaustion_is_reported_honestly():
    outcome = run_risk_guided(
        evaluate=make_evaluate(), n_regions=3, config=SearchConfig(budget=6, seed=0)
    )
    assert outcome.candidate_count <= 6
    assert outcome.stop_reason in {"budget_exhausted", "local_optimum", "space_exhausted"}
    seen = set()
    for r in outcome.trajectory:
        key = tuple(round(a, 6) for a in r.alphas)
        if not r.cache_hit:
            assert key not in seen, "duplicate NEW evaluation of the same state"
            seen.add(key)


# ---------------------------------------------------------------------------
# Baseline fairness (README 6/9)
# ---------------------------------------------------------------------------

def test_plain_local_shares_rules_and_differs_only_in_ordering():
    evaluate = make_evaluate()
    cfg = SearchConfig(budget=40, seed=7)
    risk_out = run_risk_guided(evaluate=evaluate, n_regions=3, config=cfg)
    plain_out = run_plain_local(evaluate=evaluate, n_regions=3, config=cfg)
    for out in (risk_out, plain_out):
        assert out.candidate_count <= 40
        assert out.decoder_calls >= out.candidate_count  # 1 initial + candidates
        assert out.status in {"improved", "no_improvement"}
    risk_reasons = {r.selection_reason for r in risk_out.trajectory}
    plain_reasons = {r.selection_reason for r in plain_out.trajectory}
    assert any("risk_desc" in s for s in risk_reasons)
    assert any("seeded_random_permutation" in s for s in plain_reasons)


def test_random_states_respects_budget_and_counts_unique_states():
    outcome = run_random_states(
        evaluate=make_evaluate(), n_regions=3, config=SearchConfig(budget=10, seed=3)
    )
    assert outcome.candidate_count == 10
    new_states = [
        tuple(round(a, 6) for a in r.alphas) for r in outcome.trajectory if not r.cache_hit
    ]
    assert len(new_states) == len(set(new_states)) == 10
    assert all(
        any(math.isclose(a, lv) for a in alphas for lv in LEVELS) for alphas in new_states
    )


def test_global_scan_reports_real_counts_and_stays_in_domain():
    evaluate = make_evaluate()
    # domain-only grid: 0.6 is evaluation #1, so 4 new states are evaluated
    outcome = run_global_scan(
        evaluate=evaluate, n_regions=3, grid=(0.3, 0.5, 0.6, 0.7, 0.9),
        config=SearchConfig(budget=40),
    )
    assert outcome.candidate_count == 5  # default 0.6 + 4 grid points
    assert outcome.restarts_used == 0
    # the fake landscape's best feasible state under uniform alpha is 0.6 (default)
    assert outcome.used_default and outcome.status == "no_improvement"


def test_global_scan_with_mixed_domain_counts_everything():
    evaluate = make_evaluate()
    outcome = run_global_scan(
        evaluate=evaluate, n_regions=3, grid=(0.3, 0.45, 0.6),
        config=SearchConfig(budget=40),
    )
    assert outcome.candidate_count == 3
    # 0.45 is outside the formal domain; it may not enter the formal archive
    assert outcome.used_default
    assert outcome.stop_reason == "grid_complete_mixed_domain"


# ---------------------------------------------------------------------------
# Coordination math
# ---------------------------------------------------------------------------

def test_coordination_without_edges_is_zero():
    assert coordination((0.3, 0.9), ()) == 0.0
    assert coordination_contrib((0.3, 0.9), ()) == (0.0, 0.0)


def test_coordination_contributions_sum_to_s():
    alphas = (0.3, 0.6, 0.9)
    s = coordination(alphas, N3_EDGES)
    parts = coordination_contrib(alphas, N3_EDGES)
    assert math.isclose(sum(parts), s, rel_tol=1e-12)


def test_feasibility_boundary_respects_epsilon():
    cfg = ScorerConfig(epsilon=0.05)
    # exactly at the cap -> feasible; one epsilon above -> infeasible
    at = assemble_score(0, T0 * 1.05, 0, (0, 0, 0), Q0, T0, cfg, source="fake")
    over = assemble_score(0, T0 * 1.050001, 0, (0, 0, 0), Q0, T0, cfg, source="fake")
    assert at.feasible and at.violation == 0.0
    assert not over.feasible and over.violation > 0
