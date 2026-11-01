"""
Assignment solvers.

All of them respect: each paper gets exactly its number of reviews, each reviewer stays within their load,
conflicts are never assigned, required pairs are always assigned. Optional: at most N reviewers from one
institution per paper, and at least N experienced reviewers per paper.

* max_total   Highest total expertise. A linear programme solved by column generation: start from each paper's
              best candidates, solve, then use the dual prices to find every left-out pair that could improve
              the solution, add those, and repeat until none is left. The final duals prove the answer is
              optimal over *all* pairs. With only loads and institutions the LP optimum is integral (the
              constraint matrix is totally unimodular); otherwise a MILP finishes the job.
* fair        Protect the worst-off paper (the PeerReview4All objective, Stelmakh et al. 2021): first maximise
              the smallest total expertise any paper receives, then maximise the total expertise without letting
              any paper drop below that level. Two mixed-integer programmes (HiGHS).
* randomized  Randomised assignment against manipulation (Jecmen et al. 2020): no reviewer-paper pair may be
              assigned with probability above q, so a reviewer cannot guarantee themselves a friend's paper.
              Maximises expected expertise (LP), then samples an assignment with exactly those probabilities by
              dependent rounding (Gandhi et al. 2006), which keeps every paper's demand and every load.
* greedy      The usual spreadsheet approach, as a baseline: best pairs first while capacity lasts.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .problem import Problem

TOL = 1e-7


@dataclass
class Result:
    A: np.ndarray                      # papers x reviewers, True = assigned
    method: str
    status: str                        # "optimal", "time limit", "infeasible", ...
    objective: float                   # total affinity of A
    seconds: float
    bound: float | None = None         # proven upper bound on the method's objective
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("optimal", "time limit", "sampled", "heuristic")


class Infeasible(Exception):
    pass


# ---------------------------------------------------------------------------------------------- model building
def _candidates(pb: Problem, K: int, extra=None) -> np.ndarray:
    P, R = pb.shape
    allowed = ~pb.conflict
    S2 = np.where(allowed, pb.S, -np.inf)
    mask = np.zeros((P, R), dtype=bool)
    k = min(K, R)
    top = np.argpartition(-S2, k - 1, axis=1)[:, :k]
    mask[np.arange(P)[:, None], top] = True
    kr = min(max(K // 2, 3 * int(pb.min_load.max(initial=0))), P)
    if kr > 0:
        topr = np.argpartition(-S2, kr - 1, axis=0)[:kr, :]
        mask[topr, np.arange(R)[None, :]] = True
    mask |= pb.forced
    if extra is not None:
        mask |= extra
    return mask & (allowed | pb.forced)


class _Model:
    """Rows of the LP over a set of candidate pairs. Equality rows: one per paper. Inequality rows (<=): reviewer
    maxima, reviewer minima (as -sum <= -min), institution caps per (paper, institution), experienced reviewers
    per paper (as -sum <= -min)."""

    def __init__(self, pb: Problem, mask: np.ndarray):
        self.pb = pb
        P, R = pb.shape
        self.pi, self.ri = np.nonzero(mask)
        n = self.n = len(self.pi)
        cols = np.arange(n)
        self.A_eq = sp.csr_matrix((np.ones(n), (self.pi, cols)), shape=(P, n))
        self.b_eq = pb.demand.astype(float)
        blocks, rhs = [sp.csr_matrix((np.ones(n), (self.ri, cols)), shape=(R, n))], [pb.max_load.astype(float)]
        self.row_max = 0
        self.min_rows = np.flatnonzero(pb.min_load > 0)
        self.min_row_of = np.full(R, -1)
        offset = R
        if len(self.min_rows):
            self.min_row_of[self.min_rows] = offset + np.arange(len(self.min_rows))
            sel = np.isin(self.ri, self.min_rows)
            blocks.append(sp.csr_matrix((-np.ones(sel.sum()), (np.searchsorted(self.min_rows, self.ri[sel]), cols[sel])),
                                        shape=(len(self.min_rows), n)))
            rhs.append(-pb.min_load[self.min_rows].astype(float))
            offset += len(self.min_rows)
        self.inst_row = {}
        if pb.institution is not None and pb.max_per_institution:
            g = pb.institution[self.ri]
            keys = {}
            for j in np.flatnonzero(g >= 0):
                keys.setdefault((self.pi[j], g[j]), []).append(j)
            groups = [(k, js) for k, js in keys.items() if len(js) > pb.max_per_institution]
            if groups:
                rr, cc = [], []
                for i, (k, js) in enumerate(groups):
                    self.inst_row[k] = offset + i
                    rr += [i] * len(js)
                    cc += js
                blocks.append(sp.csr_matrix((np.ones(len(cc)), (rr, cc)), shape=(len(groups), n)))
                rhs.append(np.full(len(groups), float(pb.max_per_institution)))
                offset += len(groups)
        self.senior_row0 = None
        if pb.min_senior and pb.senior is not None:
            sel = pb.senior[self.ri]
            blocks.append(sp.csr_matrix((-np.ones(sel.sum()), (self.pi[sel], cols[sel])), shape=(P, n)))
            rhs.append(np.full(P, -float(pb.min_senior)))
            self.senior_row0 = offset
            offset += P
        self.A_ub = sp.vstack(blocks).tocsr()
        self.b_ub = np.concatenate(rhs)

    def reduced_costs(self, w: np.ndarray, y_eq: np.ndarray, y_ub: np.ndarray, ub: np.ndarray | None = None) -> np.ndarray:
        """Reduced cost of every pair (minimisation of -w); negative = would improve the LP."""
        pb = self.pb
        P, R = pb.shape
        d = -w - y_eq[:, None] - y_ub[:R][None, :]
        if len(self.min_rows):
            d += np.where(self.min_row_of >= 0, y_ub[np.maximum(self.min_row_of, 0)], 0.0)[None, :]
        for (p, g), row in self.inst_row.items():
            d[p, pb.institution == g] -= y_ub[row]
        if self.senior_row0 is not None:
            d += np.outer(y_ub[self.senior_row0:self.senior_row0 + P], pb.senior.astype(float))
        return d


def _solve_lp(model: _Model, w: np.ndarray, ub_value: float | np.ndarray = 1.0):
    pb = model.pb
    c = -w[model.pi, model.ri]
    lb = pb.forced[model.pi, model.ri].astype(float)
    ub = np.maximum(np.broadcast_to(ub_value, w.shape)[model.pi, model.ri] if np.ndim(ub_value) else
                    np.full(model.n, float(ub_value)), lb)
    return linprog(c, A_ub=model.A_ub, b_ub=model.b_ub, A_eq=model.A_eq, b_eq=model.b_eq,
                   bounds=np.column_stack([lb, ub]), method="highs-ds")


def _column_generation(pb: Problem, w: np.ndarray, K: int, ub_value=1.0, max_rounds: int = 60):
    """LP optimum of max sum w*x over all allowed pairs, via column generation. Returns (model, lp, rounds)."""
    allowed = ~pb.conflict | pb.forced
    mask = _candidates(pb, K)
    for rounds in range(1, max_rounds + 1):
        model = _Model(pb, mask)
        lp = _solve_lp(model, w, ub_value)
        if lp.status == 2:                       # infeasible on the candidates: try every allowed pair once
            if mask.sum() == allowed.sum():
                raise Infeasible("The constraints cannot all be met. " + " ".join(pb.issues()))
            mask = allowed.copy()
            continue
        if lp.status != 0:
            raise RuntimeError(f"LP solver: {lp.message}")
        d = model.reduced_costs(w, lp.eqlin.marginals, lp.ineqlin.marginals)
        enter = (d < -1e-9) & allowed & ~mask
        if not enter.any():
            return model, lp, rounds
        # add, per paper, the most promising left-out pairs
        dd = np.where(enter, d, np.inf)
        k = min(K, pb.shape[1])
        best = np.argpartition(dd, k - 1, axis=1)[:, :k]
        add = np.zeros_like(mask)
        add[np.arange(pb.shape[0])[:, None], best] = True
        mask |= add & enter
    return model, lp, max_rounds


def _to_matrix(pb: Problem, model: _Model, x: np.ndarray) -> np.ndarray:
    A = np.zeros(pb.shape, dtype=bool)
    A[model.pi, model.ri] = x > 0.5
    return A


def _milp(model: _Model, c: np.ndarray, extra_rows=None, time_limit=60.0, n_extra_vars=0, lb_extra=None, ub_extra=None,
          integral_extra=False, gap=1e-6):
    pb = model.pb
    n = model.n
    lb = np.r_[pb.forced[model.pi, model.ri].astype(float), lb_extra if lb_extra is not None else np.zeros(n_extra_vars)]
    ub = np.r_[np.ones(n), ub_extra if ub_extra is not None else np.full(n_extra_vars, np.inf)]
    pad = lambda M: sp.hstack([M, sp.csr_matrix((M.shape[0], n_extra_vars))]) if n_extra_vars else M
    cons = [LinearConstraint(pad(model.A_eq), model.b_eq, model.b_eq),
            LinearConstraint(pad(model.A_ub), -np.inf, model.b_ub)]
    if extra_rows is not None:
        cons += extra_rows
    integ = np.r_[np.ones(n), np.full(n_extra_vars, 1.0 if integral_extra else 0.0)]
    return milp(c, constraints=cons, bounds=Bounds(lb, ub), integrality=integ,
                options={"time_limit": float(time_limit), "mip_rel_gap": gap})


# ---------------------------------------------------------------------------------------------- solvers
def max_total(pb: Problem, bonus: np.ndarray | None = None, K: int | None = None, time_limit: float = 60.0) -> Result:
    """Highest total affinity. `bonus` (papers x reviewers) is added to the objective, e.g. to keep a previous
    assignment when re-planning."""
    t0 = time.time()
    w = pb.S + (bonus if bonus is not None else 0.0)
    K = K or max(20, 4 * int(pb.demand.max(initial=1)))
    try:
        model, lp, rounds = _column_generation(pb, w, K)
    except Infeasible as e:
        return Result(np.zeros(pb.shape, bool), "max_total", "infeasible", 0.0, time.time() - t0, info={"message": str(e)})
    x = lp.x
    frac = int(((x > 1e-6) & (x < 1 - 1e-6)).sum())
    info = {"candidates": model.n, "rounds": rounds, "lp_bound": -lp.fun, "fractional": frac}
    status = "optimal"
    if frac:
        # rare: experienced-reviewer rows break total unimodularity; finish with a MILP on the same pairs
        m = _milp(model, -w[model.pi, model.ri], time_limit=time_limit)
        if m.x is None:
            return Result(np.zeros(pb.shape, bool), "max_total", "infeasible", 0.0, time.time() - t0, info=info)
        x = m.x
        status = "optimal" if m.status == 0 else "time limit"
    A = _to_matrix(pb, model, x)
    return Result(A, "max_total", status, float(pb.S[A].sum()), time.time() - t0, bound=-lp.fun if bonus is None else None, info=info)


def floor_upper_bound(pb: Problem) -> float:
    """No paper can do better than its own best eligible reviewers, so the worst-off paper cannot either."""
    S2 = np.where(pb.conflict, -np.inf, pb.S)
    srt = -np.sort(-S2, axis=1)
    best = np.array([srt[p, :pb.demand[p]].sum() for p in range(pb.shape[0])]) if pb.shape[0] else np.zeros(0)
    return float(best.min(initial=np.inf))


def fair(pb: Problem, K: int | None = None, time_limit: float = 60.0) -> Result:
    """Maximise the worst-off paper's total expertise, then the total expertise subject to that floor.

    The floor can never exceed `floor_upper_bound` (the weakest paper's best possible reviewers). That bound is tried
    first: when it is reachable, the floor is proven optimal without the harder max-min search."""
    t0 = time.time()
    base = max_total(pb, K=K)
    if base.status == "infeasible":
        base.method = "fair"
        return base
    base_min = float((pb.S * base.A).sum(axis=1).min(initial=0))
    ub = floor_upper_bound(pb)
    tol = lambda v: v * (1 - 1e-6) - 1e-9
    info = {"total_without_fairness": base.objective, "min_without_fairness": base_min, "floor_bound": ub}
    if base_min >= tol(ub):     # the highest-total assignment already gives the weakest paper its best reviewers
        info.update(floor=base_min, candidates=base.info.get("candidates"), how="already at the bound")
        return Result(base.A, "fair", base.status, base.objective, time.time() - t0, info=info)
    K = K or max(40, 8 * int(pb.demand.max(initial=1)))
    model = _Model(pb, _candidates(pb, K, extra=base.A))
    P = pb.shape[0]
    n = model.n
    s = pb.S[model.pi, model.ri]
    A_s = sp.csr_matrix((s, (model.pi, np.arange(n))), shape=(P, n))
    info["candidates"] = n
    # try the bound itself: if some assignment reaches it, the floor is optimal
    quick = _milp(model, -s, [LinearConstraint(A_s, tol(ub), np.inf)], min(15.0, time_limit / 4))
    if quick.x is not None:
        A = _to_matrix(pb, model, quick.x)
        info.update(floor=float((pb.S * A).sum(axis=1).min()), how="bound reached")
        return Result(A, "fair", "optimal" if quick.status == 0 else "time limit", float(pb.S[A].sum()),
                      time.time() - t0, info=info)
    left = max(time_limit - (time.time() - t0), 5.0)
    # stage 1: maximise t subject to every paper's total >= t
    floor_rows = [LinearConstraint(sp.hstack([A_s, -sp.csr_matrix(np.ones((P, 1)))]), 0, np.inf)]
    m1 = _milp(model, np.r_[np.zeros(n), -1.0], floor_rows, left / 2, n_extra_vars=1)
    if m1.x is None:
        base.method, base.status = "fair", "time limit"
        base.info.update(info, note="no fair solution found in time; showing the highest-total assignment")
        return base
    t_star = float(-m1.fun)
    if getattr(m1, "mip_dual_bound", None) is not None and np.isfinite(m1.mip_dual_bound):
        info["floor_bound"] = min(ub, float(-m1.mip_dual_bound))
    floor = tol(max(t_star, base_min))
    stage1_ok = m1.status == 0 or t_star >= tol(info["floor_bound"])
    # stage 2: maximise the total subject to every paper's total >= floor
    m2 = _milp(model, -s, [LinearConstraint(A_s, floor, np.inf)], max(time_limit - (time.time() - t0), 5.0))
    if m2.x is None:
        A = _to_matrix(pb, model, m1.x[:n])
        status = "time limit"
    else:
        A = _to_matrix(pb, model, m2.x)
        status = "optimal" if (stage1_ok and m2.status == 0) else "time limit"
    info.update(floor=float((pb.S * A).sum(axis=1).min()), how="max-min search")
    return Result(A, "fair", status, float(pb.S[A].sum()), time.time() - t0, info=info)


def dependent_round(pi: np.ndarray, ri: np.ndarray, x: np.ndarray, rng: random.Random) -> np.ndarray:
    """Round a fractional bipartite assignment so that each pair is chosen with probability x, each paper keeps
    its (integral) number of reviewers exactly and each reviewer's load ends at the floor or ceiling of its
    fractional load (Gandhi, Khuller, Parthasarathy & Srinivasan 2006)."""
    y = np.asarray(x, dtype=float).copy()
    y[y < TOL] = 0.0
    y[y > 1 - TOL] = 1.0
    frac = set(np.flatnonzero((y > 0) & (y < 1)).tolist())
    adj = {}
    for e in frac:
        adj.setdefault(("p", int(pi[e])), set()).add(e)
        adj.setdefault(("r", int(ri[e])), set()).add(e)

    def other(e, node):
        return ("r", int(ri[e])) if node[0] == "p" else ("p", int(pi[e]))

    leaves = {nd for nd, es in adj.items() if len(es) == 1}
    while frac:
        start = next(iter(leaves)) if leaves else ("p", int(pi[next(iter(frac))]))
        node, prev, edges, seen = start, None, [], {start: 0}
        while True:   # walk until the path ends (a maximal path) or comes back on itself (a cycle)
            nxt_edges = [f for f in adj[node] if f != prev]
            if not nxt_edges:
                break
            e = nxt_edges[0]
            nxt = other(e, node)
            edges.append(e)
            if nxt in seen:
                edges = edges[seen[nxt]:]
                break
            seen[nxt] = len(edges)
            node, prev = nxt, e
        m1, m2 = edges[0::2], edges[1::2]
        a = min([1 - y[e] for e in m1] + [y[e] for e in m2])
        b = min([y[e] for e in m1] + [1 - y[e] for e in m2])
        if rng.random() < b / (a + b):
            for e in m1:
                y[e] += a
            for e in m2:
                y[e] -= a
        else:
            for e in m1:
                y[e] -= b
            for e in m2:
                y[e] += b
        for e in edges:
            if y[e] < TOL or y[e] > 1 - TOL:
                y[e] = float(round(y[e]))
                frac.discard(e)
                for nd in (("p", int(pi[e])), ("r", int(ri[e]))):
                    es = adj.get(nd)
                    if es is None:
                        continue
                    es.discard(e)
                    if len(es) == 1:
                        leaves.add(nd)
                    else:
                        leaves.discard(nd)
                    if not es:
                        del adj[nd]
    return y


def randomized(pb: Problem, q: float = 0.5, seed: int = 0, K: int | None = None, samples: int = 1) -> Result:
    """Maximise expected expertise with P(pair assigned) <= q, then sample one assignment."""
    t0 = time.time()
    if pb.min_senior or (pb.institution is not None and pb.max_per_institution):
        raise ValueError("Randomised assignment supports loads, conflicts and required pairs; switch off the "
                         "institution and experienced-reviewer rules to use it.")
    if not 0 < q <= 1:
        raise ValueError("q must be between 0 and 1")
    K = K or max(20, int(np.ceil(4 * pb.demand.max(initial=1) / q)))
    ub = np.full(pb.shape, q)
    try:
        model, lp, rounds = _column_generation(pb, pb.S, K, ub_value=ub)
    except Infeasible as e:
        return Result(np.zeros(pb.shape, bool), "randomized", "infeasible", 0.0, time.time() - t0,
                      info={"message": str(e) + f" With q = {q}, every paper needs at least {pb.demand.max() / q:.0f} eligible reviewers."})
    x = np.clip(lp.x, 0, 1)
    y = dependent_round(model.pi, model.ri, x, random.Random(seed))
    A = _to_matrix(pb, model, y)
    best = max_total(pb, K=K)
    info = {"candidates": model.n, "expected_total": float(-lp.fun), "deterministic_best": best.objective,
            "q": q, "seed": seed, "max_probability": float(x[~pb.forced[model.pi, model.ri]].max(initial=0)),
            "prob_pairs": (model.pi, model.ri, x)}
    return Result(A, "randomized", "sampled", float(pb.S[A].sum()), time.time() - t0, bound=float(-lp.fun), info=info)


def greedy(pb: Problem) -> Result:
    """Baseline: required pairs first, then the best remaining pairs while papers need reviewers and reviewers
    have room. Ignores minimum loads, institution and experience rules, and can leave papers short."""
    t0 = time.time()
    P, R = pb.shape
    A = pb.forced.copy()
    need = pb.demand - A.sum(axis=1)
    room = pb.max_load - A.sum(axis=0)
    allowed = ~pb.conflict & ~A
    order = np.argsort(-np.where(allowed, pb.S, -np.inf), axis=None, kind="stable")
    for idx in order:
        p, r = divmod(int(idx), R)
        if not allowed[p, r]:
            break
        if need[p] > 0 and room[r] > 0:
            A[p, r] = True
            need[p] -= 1
            room[r] -= 1
    short = int((need > 0).sum())
    return Result(A, "greedy", "heuristic" if not short else "incomplete", float(pb.S[A].sum()), time.time() - t0,
                  info={"papers_short": short})


def replan(pb: Problem, current: np.ndarray, method: str = "total", time_limit: float = 60.0) -> Result:
    """Re-plan after a change (a reviewer drops out, new conflicts, loads change) while keeping as many of the
    current pairs as possible: each kept pair earns a bonus larger than any expertise gain from moving it."""
    keep = current & ~pb.conflict
    bonus = keep * (1.0 + 2.0 * float(pb.S.max(initial=0)) * int(pb.demand.max(initial=1)))
    res = max_total(pb, bonus=bonus, time_limit=time_limit)
    res.method = "replan"
    res.info["kept"] = int((res.A & current).sum())
    res.info["moved"] = int((current & ~res.A).sum())
    return res


SOLVERS = {"Highest total expertise": "max_total", "Fair: protect the worst-off paper": "fair",
           "Randomised against manipulation": "randomized"}


def solve(pb: Problem, method: str, **kw) -> Result:
    return {"max_total": max_total, "fair": fair, "randomized": randomized, "greedy": greedy}[method](pb, **kw)
