"""
Checks of the engine against independent references. No downloads needed.

    python tests/verify_engine.py

1. Highest total: equals a dense integer program over every allowed pair (random instances, with institution caps,
   experienced reviewers and minimum loads), although the solver keeps only a few candidates per paper.
2. Fair: its worst-off paper equals the dense max-min optimum; its total never beats the highest total.
3. Randomised: always valid; sampled pairs appear with their planned probabilities; none above the cap.
4. Re-plan: keeps the largest possible number of existing pairs (dense integer program as reference).
5. Rule checker: every kind of broken rule is reported (mutation test of the checker itself).
6. Conflicts, expertise, file reading, the gold-standard loss.
7. Mutation test: bugs are planted one at a time (no pricing, no keep bonus, independent rounding, ...);
   the checks above must catch every one.
"""

from __future__ import annotations

import os
import random
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import LinearConstraint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import revassign.conflicts as CF  # noqa: E402
import revassign.evaluate as EV  # noqa: E402
import revassign.expertise as EXP  # noqa: E402
from revassign import solve as S  # noqa: E402
from revassign.io import Settings, build, read_conference  # noqa: E402
from revassign.problem import Problem  # noqa: E402

FAILS = []
QUIET = [False]


def ok(cond, what):
    if not cond:
        FAILS.append(what)
        if not QUIET[0]:
            print("  FAIL", what)


def violations(pb, A):
    return EV.violations(pb, A)


def summary(pb, A):
    return EV.summary(pb, A)


def dense_total(pb):
    m = S._Model(pb, ~pb.conflict | pb.forced)
    r = S._milp(m, -pb.S[m.pi, m.ri])
    return None if r.x is None else -r.fun


def dense_maxmin(pb):
    m = S._Model(pb, ~pb.conflict | pb.forced)
    P, n = pb.shape[0], m.n
    s = pb.S[m.pi, m.ri]
    A_s = sp.csr_matrix((s, (m.pi, np.arange(n))), shape=(P, n))
    r = S._milp(m, np.r_[np.zeros(n), -1.0], [LinearConstraint(sp.hstack([A_s, -sp.csr_matrix(np.ones((P, 1)))]), 0, np.inf)],
                n_extra_vars=1)
    return -r.fun


def dense_max_kept(pb, current):
    m = S._Model(pb, ~pb.conflict | pb.forced)
    r = S._milp(m, -(current[m.pi, m.ri] & ~pb.conflict[m.pi, m.ri]).astype(float))
    return round(-r.fun)


def random_problem(rng, it, extras=True):
    P, R = int(rng.integers(5, 40)), int(rng.integers(4, 30))
    k = int(rng.integers(1, 4))
    L = int(np.ceil(P * k / R)) + int(rng.integers(0, 3))
    inst = rng.integers(0, 4, R) if extras and it % 3 == 0 else None
    sen = rng.random(R) < 0.5 if extras and it % 4 == 1 else None
    return Problem(rng.random((P, R)) ** 2, k, L, min_load=1 if extras and it % 5 == 2 else 0,
                   conflict=rng.random((P, R)) < 0.1, institution=inst, max_per_institution=1 if inst is not None else None,
                   senior=sen, min_senior=1 if sen is not None else 0)


def test_max_total(n=60, seed=0, quiet=False):
    rng = np.random.default_rng(seed)
    tested = bad = 0
    for it in range(n):
        pb = random_problem(rng, it)
        if pb.issues():
            continue
        opt = dense_total(pb)
        res = S.max_total(pb, K=3)                 # few candidates: column generation must find the rest
        if opt is None:
            ok(res.status == "infeasible", f"max_total {it}: dense infeasible but solver says {res.status}")
            continue
        tested += 1
        good = res.status == "optimal" and not violations(pb, res.A) and abs(res.objective - opt) < 1e-6
        bad += not good
        if not quiet:
            ok(good, f"max_total {it}: {res.objective} vs dense {opt}")
    return tested, bad


def test_fair(n=40):
    rng = np.random.default_rng(1)
    tested = 0
    for it in range(n):
        P, R = int(rng.integers(5, 30)), int(rng.integers(4, 25))
        k = int(rng.integers(1, 4))
        L = int(np.ceil(P * k / R)) + int(rng.integers(0, 2))
        pb = Problem(rng.random((P, R)) ** 2, k, L, conflict=rng.random((P, R)) < 0.1)
        if pb.issues():
            continue
        tested += 1
        f = S.fair(pb, K=4)
        best = S.max_total(pb)
        worst = summary(pb, f.A)["worst_paper"]
        mm = dense_maxmin(pb)
        ok(not violations(pb, f.A), f"fair {it}: invalid")
        ok(worst >= mm - 1e-6, f"fair {it}: worst {worst} < max-min {mm}")
        ok(worst <= S.floor_upper_bound(pb) + 1e-9, f"fair {it}: above the bound")
        ok(f.objective <= best.objective + 1e-9, f"fair {it}: total above the optimum")
    return tested


def test_randomized():
    rng = np.random.default_rng(2)
    bad = 0
    for it in range(25):
        P, R = int(rng.integers(5, 25)), int(rng.integers(6, 25))
        pb = Problem(rng.random((P, R)), 2, int(np.ceil(P * 2 / R)) + 1, conflict=rng.random((P, R)) < 0.1)
        if pb.issues():
            continue
        for seed in range(4):
            r = S.randomized(pb, q=0.6, seed=seed)
            if r.status == "infeasible":
                break
            bad += bool(violations(pb, r.A))
            ok(r.info["max_probability"] <= 0.6 + 1e-9, "randomized: a pair above the cap")
    ok(bad == 0, f"randomized: {bad} invalid samples")
    # marginals: frequencies match the planned probabilities
    pb = Problem(np.random.default_rng(3).random((12, 10)), 3, 4)
    r = S.randomized(pb, q=0.5, seed=0)
    pi, ri, x = r.info["prob_pairs"]
    acc = np.zeros(len(x))
    n = 4000
    for s in range(n):
        acc += S.dependent_round(pi, ri, x, random.Random(s))
    dev = np.abs(acc / n - x).max()
    ok(dev < 0.035, f"randomized marginals off by {dev:.3f}")
    return dev


def test_replan():
    rng = np.random.default_rng(4)
    tested = 0
    for it in range(30):
        pb = random_problem(rng, it, extras=it % 2 == 0)
        if pb.issues():
            continue
        cur = S.max_total(pb)
        if cur.status != "optimal":
            continue
        gone = int(rng.integers(0, pb.shape[1]))
        C = pb.conflict.copy()
        C[:, gone] = True
        mx, mn = pb.max_load.copy(), pb.min_load.copy()
        mx[gone] = mn[gone] = 0
        pb2 = pb.with_changes(conflict=C, max_load=mx, min_load=mn)
        if pb2.issues() or dense_total(pb2) is None:
            continue
        tested += 1
        r = S.replan(pb2, cur.A)
        ok(not violations(pb2, r.A), f"replan {it}: invalid")
        ok(r.info["kept"] == dense_max_kept(pb2, cur.A), f"replan {it}: kept {r.info['kept']} < {dense_max_kept(pb2, cur.A)}")
    return tested


def test_checker():
    """Break a valid assignment in every way; the checker must notice each one."""
    rng = np.random.default_rng(5)
    pb = Problem(rng.random((10, 8)), 2, 4, min_load=1, conflict=rng.random((10, 8)) < 0.2,
                 institution=np.array([0, 0, 1, 1, 2, 2, 3, 3]), max_per_institution=1,
                 senior=np.array([1, 0, 1, 0, 1, 0, 1, 0], bool), min_senior=1)
    A = S.max_total(pb).A
    ok(not violations(pb, A), "checker: valid assignment flagged")
    mutations = {}
    B = A.copy(); p = 0; B[p, np.flatnonzero(B[p])[0]] = False; mutations["missing review"] = B
    no_c = pb.with_changes(conflict=np.zeros_like(pb.conflict))
    only_conflict = []                     # swap one reviewer for a conflicted one, breaking no other rule
    for p, r in np.argwhere(A):
        for rc in np.flatnonzero(pb.conflict[p] & ~A[p]):
            B = A.copy()
            B[p, r], B[p, rc] = False, True
            if not violations(no_c, B):
                only_conflict.append(B)
    ok(len(only_conflict) > 0, "checker: no conflict-only mutation found")
    mutations["conflict"] = only_conflict[0] if only_conflict else A
    B = A.copy(); r = 0; B[:, r] = ~pb.conflict[:, r]; mutations["over load"] = B
    B = A.copy(); r = int(np.flatnonzero(A.any(axis=0))[0]); B[:, r] = False; mutations["under minimum"] = B
    B = A.copy(); p = 3; B[p] = False; B[p, [4, 5]] = True; mutations["same institution"] = B
    B = A.copy(); p = 2; B[p] = False; B[p, [1, 3]] = True; mutations["no experienced reviewer"] = B
    for name, B in mutations.items():
        ok(len(violations(pb, B)) > 0, f"checker missed: {name}")
    return len(mutations)


def test_conflicts():
    ok(CF.name_parts("Smith, John") == ("john", "smith"), "name: last, first")
    ok(CF.same_person(CF.name_parts("J. Smith"), CF.name_parts("John Smith")), "name: initial")
    ok(not CF.same_person(CF.name_parts("Jane Smith"), CF.name_parts("John Smith")), "name: different first names")
    ok(CF.same_person(CF.name_parts("José García"), CF.name_parts("Jose Garcia")), "name: accents")
    ok(CF.split_authors("A. B; C. D") == ["A. B", "C. D"], "split: semicolons")
    ok(CF.split_authors("['Ann Lee', 'Bo Chen']") == ["Ann Lee", "Bo Chen"], "split: list text")
    C, why = CF.find_conflicts(
        [["Ann Lee"], ["Bo Chen"], ["Cy Diaz"], ["Di Eng"]], ["Ann Lee", "Fay Gu", "Hal Ito", "Ivy Jo"],
        reviewer_profile_authors=[[], [["Fay Gu", "Bo Chen"]], [["Hal Ito", "Cy Diaz"], ["Hal Ito"]], []],
        reviewer_profile_years=[[], [2024], [2005, 2024], []],
        paper_emails=[[], [], [], ["di@uni.edu"]], reviewer_emails=["", "", "", "ivy@uni.edu"],
        paper_institutions=[[], [], [], []], reviewer_institutions=["", "", "", ""], coauthor_years=5)
    ok(C[0, 0] and "author" in why[(0, 0)], "conflict: author")
    ok(C[1, 1] and "co-author" in why[(1, 1)], "conflict: recent co-author")
    ok(not C[2, 2], "conflict: old co-author outside the window")
    ok(C[3, 3] and "domain" in why[(3, 3)], "conflict: e-mail domain")
    C2, _ = CF.find_conflicts([["X Y"]], ["Z W"], paper_emails=[["x@gmail.com"]], reviewer_emails=["z@gmail.com"])
    ok(not C2.any(), "conflict: public mail provider ignored")
    C3, _ = CF.find_conflicts([["X Y"]], ["Z W"], paper_institutions=[["The University of Colombo"]],
                           reviewer_institutions=["University of Colombo"])
    ok(C3.all(), "conflict: institution names normalised")


def test_expertise():
    sims = np.array([[0.1, 0.5, 0.2]])
    w = np.array([0.2, 0.3, 0.5])
    ok(np.isclose(EXP.soft_max(sims, w, 0)[0], sims @ w), "soft max: T=0 is the weighted mean")
    ok(np.isclose(EXP.soft_max(sims, w, np.inf)[0], 0.5), "soft max: T=inf is the maximum")
    ok(sims @ w < EXP.soft_max(sims, w, 5)[0] < 0.5, "soft max: in between")
    rw = EXP.recency_weights([2024, 2019, 2014], 5)
    ok(np.allclose(rw[1] / rw[0], 0.5) and np.allclose(rw[2] / rw[0], 0.25), "recency: half-life")
    m = EXP.build_model(["graph neural networks for molecules", "kalman filters for time series"],
                    [["message passing neural networks on graphs"], ["state space models and kalman filters"]])
    E = m.scores()
    ok(E[0, 0] > E[0, 1] and E[1, 1] > E[1, 0], "expertise: right reviewer ranked first")
    ok(np.allclose(E, m.score_texts(["graph neural networks for molecules", "kalman filters for time series"])),
       "expertise: pasted text scores the same")
    ok(((E >= 0) & (E <= 1)).all(), "expertise: in [0, 1]")
    ok("kalman" in " ".join(m.explain(1, 1)["terms"]), "explanation: shared term")


def test_files():
    d = os.path.join(ROOT, "sample_data")
    files = {k: (open(os.path.join(d, f"example_{k}.csv"), "rb").read(), f"example_{k}.csv")
             for k in ("papers", "reviewers", "profiles", "conflicts", "bids")}
    conf = read_conference(files)
    b = build(conf, Settings(demand=2, max_load=5))
    pb = b.problem
    ok(pb.shape == (12, 6), "files: shape")
    ok(pb.conflict[conf.papers.paper_id.tolist().index("S09"), 1], "files: declared conflict")
    r = S.max_total(pb)
    ok(r.status == "optimal" and not violations(pb, r.A), "files: solved")
    try:
        read_conference({"papers": (b"id,name\n1,x\n", "p.csv"), "reviewers": files["reviewers"]})
        ok(False, "files: missing title column not reported")
    except ValueError as e:
        ok("title" in str(e), "files: error names the missing column")


def test_loss():
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    from run_benchmark import loss_parts, pair_terms
    refs = {"r": {"a": 5, "b": 3, "c": 1}}
    t = pair_terms(refs)
    for preds, want in (({"a": 3, "b": 2, "c": 1}, 0.0), ({"a": 1, "b": 2, "c": 3}, 1.0), ({"a": 1, "b": 1, "c": 1}, 0.5)):
        n, d = loss_parts({"r": preds}, t, ["r"])
        ok(abs(n / d - want) < 1e-12, f"loss: {preds} gives {n / d}, expected {want}")


# ------------------------------------------------------------------------------------------------ 7. mutation test
def _indep_round(pi, ri, x, rng):
    return np.array([1.0 if rng.random() < v else 0.0 for v in x])


def _replan_no_bonus(pb, current, method="total", time_limit=60.0):
    res = S.max_total(pb)
    res.info["kept"] = int((res.A & current).sum())
    res.info["moved"] = int((current & ~res.A).sum())
    return res


def _surname_only(a, b):
    return a is not None and b is not None and a[1] == b[1]


def _no_conflict_check(pb, A):
    keep = pb.conflict
    pb.conflict = np.zeros_like(keep)
    try:
        return _ORIG_VIOLATIONS(pb, A)
    finally:
        pb.conflict = keep


_ORIG_VIOLATIONS = EV.violations
MUTANTS = {
    "pricing switched off (no optimality check)": (S._Model, "reduced_costs",
                                                   lambda self, w, y_eq, y_ub, ub=None: np.zeros(self.pb.shape), [lambda: test_max_total(n=60, quiet=True)]),
    "re-plan without the keep bonus": (S, "replan", _replan_no_bonus, [test_replan]),
    "independent instead of dependent rounding": (S, "dependent_round", _indep_round, [test_randomized]),
    "fair goal replaced by highest total": (S, "fair", lambda pb, K=None, time_limit=60.0: S.max_total(pb), [test_fair]),
    "soft maximum ignores recency weights": (EXP, "soft_max", None, [test_expertise]),      # set in mutation_suite
    "names matched by surname only": (CF, "same_person", _surname_only, [test_conflicts]),
    "co-author time window ignored": (CF, "find_conflicts", None, [test_conflicts]),       # set in mutation_suite
    "public mail providers not ignored": (CF, "PUBLIC_DOMAINS", set(), [test_conflicts]),
    "rule checker skips conflicts": (EV, "violations", _no_conflict_check, [test_checker]),
}


def mutation_suite():
    caught = {}
    orig_soft = EXP.soft_max
    for name, (obj, attr, new, tests) in MUTANTS.items():
        if name.startswith("soft maximum"):
            new = lambda sims, w, T, _f=orig_soft: _f(sims, np.full(len(w), 1 / max(len(w), 1)), T)
        if name.startswith("co-author time window"):
            orig = CF.find_conflicts

            def new_fc(*a, _f=orig, **kw):
                kw["coauthor_years"] = None
                return _f(*a, **kw)
            new = new_fc
        saved = getattr(obj, attr)
        setattr(obj, attr, new)
        before = len(FAILS)
        QUIET[0] = True
        try:
            for t in tests:
                r = t()
                if isinstance(r, tuple) and r[1] > 0:      # test_max_total(quiet) returns (tested, mismatches)
                    FAILS.append("mutant")
        except Exception:                                    # a crash also counts as caught
            FAILS.append("mutant crashed")
        finally:
            QUIET[0] = False
            setattr(obj, attr, saved)
        caught[name] = len(FAILS) > before
        del FAILS[before:]
    return caught


def main():
    t0 = time.time()
    tested, bad = test_max_total()
    print(f"1. highest total = dense optimum on {tested} random instances ({bad} mismatches)")
    print(f"2. fair = dense max-min on {test_fair()} random instances")
    print(f"3. randomised: valid samples, marginals within {test_randomized():.3f}")
    print(f"4. re-plan keeps the most pairs on {test_replan()} instances")
    print(f"5. rule checker caught {test_checker()} kinds of broken rule")
    test_conflicts()
    test_expertise()
    test_files()
    test_loss()
    print("6. conflicts, expertise, files, loss: checked")
    caught = mutation_suite()
    ok(all(caught.values()), "mutation: some planted bugs went unnoticed: " + ", ".join(k for k, v in caught.items() if not v))
    print(f"7. mutation test: {sum(caught.values())} of {len(caught)} planted bugs caught")
    for k, v in caught.items():
        print(f"     {'caught' if v else 'MISSED'}  {k}")
    print(f"{'ALL CHECKS PASSED' if not FAILS else f'{len(FAILS)} FAILED'} in {time.time() - t0:.0f} s")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
