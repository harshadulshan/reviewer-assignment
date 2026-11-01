"""Independent checks and quality measures of an assignment (written separately from the solvers)."""

from __future__ import annotations

import numpy as np

from .problem import Problem


def violations(pb: Problem, A: np.ndarray) -> list:
    """Every rule the assignment breaks, as text. Empty list = valid."""
    A = np.asarray(A, dtype=bool)
    out = []
    got = A.sum(axis=1)
    for p in np.flatnonzero(got != pb.demand):
        out.append(f"{pb.papers[p]} has {got[p]} reviewers, needs {pb.demand[p]}")
    load = A.sum(axis=0)
    for r in np.flatnonzero(load > pb.max_load):
        out.append(f"{pb.reviewers[r]} has {load[r]} papers, maximum {pb.max_load[r]}")
    for r in np.flatnonzero(load < pb.min_load):
        out.append(f"{pb.reviewers[r]} has {load[r]} papers, minimum {pb.min_load[r]}")
    for p, r in np.argwhere(A & pb.conflict):
        out.append(f"conflict assigned: {pb.papers[p]} / {pb.reviewers[r]}")
    for p, r in np.argwhere(pb.forced & ~A):
        out.append(f"required pair missing: {pb.papers[p]} / {pb.reviewers[r]}")
    if pb.institution is not None and pb.max_per_institution:
        for p in range(A.shape[0]):
            g = pb.institution[A[p]]
            g = g[g >= 0]
            if len(g):
                vals, counts = np.unique(g, return_counts=True)
                if counts.max() > pb.max_per_institution:
                    out.append(f"{pb.papers[p]} has {counts.max()} reviewers from one institution")
    if pb.min_senior and pb.senior is not None:
        sen = (A & pb.senior[None, :]).sum(axis=1)
        for p in np.flatnonzero(sen < pb.min_senior):
            out.append(f"{pb.papers[p]} has {sen[p]} experienced reviewers, needs {pb.min_senior}")
    return out


def paper_scores(pb: Problem, A: np.ndarray) -> np.ndarray:
    return (pb.S * A).sum(axis=1)


def candidate_ranks(pb: Problem) -> np.ndarray:
    """For every pair, the reviewer's rank (1 = best) among the paper's eligible reviewers."""
    S2 = np.where(pb.conflict, -np.inf, pb.S)
    order = np.argsort(-S2, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(S2.shape[0])[:, None], order] = np.arange(1, S2.shape[1] + 1)[None, :]
    return ranks


def summary(pb: Problem, A: np.ndarray, top: int = 5) -> dict:
    A = np.asarray(A, dtype=bool)
    ps = paper_scores(pb, A)
    ranks = candidate_ranks(pb)
    best_rank = np.where(A, ranks, np.iinfo(ranks.dtype).max).min(axis=1)
    load = A.sum(axis=0)
    pairs = A.sum()
    return {
        "valid": not violations(pb, A),
        "total": float(pb.S[A].sum()),
        "mean_pair": float(pb.S[A].mean()) if pairs else 0.0,
        "worst_paper": float(ps.min()) if len(ps) else 0.0,
        "paper_p10": float(np.percentile(ps, 10)) if len(ps) else 0.0,
        "paper_median": float(np.median(ps)) if len(ps) else 0.0,
        "share_top": float((best_rank <= top).mean()) if len(ps) else 0.0,
        "mean_rank": float(ranks[A].mean()) if pairs else 0.0,
        "load_min": int(load.min()) if len(load) else 0,
        "load_max": int(load.max()) if len(load) else 0,
        "reviewers_used": int((load > 0).sum()),
    }
