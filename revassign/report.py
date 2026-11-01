"""Tables and downloads for an assignment."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .data import Conference
from .evaluate import candidate_ranks, paper_scores
from .io import Built


def pair_table(conf: Conference, b: Built, A: np.ndarray, explain: bool = True, max_explain: int = 5000) -> pd.DataFrame:
    """One row per assigned pair, with the reason it is a good match."""
    pb = b.problem
    ranks = candidate_ranks(pb)
    eligible = (~pb.conflict).sum(axis=1)
    rows = []
    pairs = np.argwhere(A)
    do_explain = explain and b.model is not None and len(pairs) <= max_explain
    for p, r in pairs:
        row = {"paper_id": conf.papers["paper_id"].iloc[p], "title": conf.papers["title"].iloc[p],
               "reviewer": pb.reviewers[r], "reviewer_id": conf.reviewers["reviewer_id"].iloc[r],
               "expertise": round(float(b.expertise[p, r]), 4), "score": round(float(pb.S[p, r]), 4),
               "rank": f"{ranks[p, r]} of {eligible[p]}", "rank_n": int(ranks[p, r])}
        if do_explain:
            e = b.model.explain(p, r)
            row["closest reviewer paper"] = e["closest"] or ""
            row["shared terms"] = ", ".join(e["terms"])
        rows.append(row)
    return pd.DataFrame(rows)


def paper_table(conf: Conference, b: Built, A: np.ndarray) -> pd.DataFrame:
    pb = b.problem
    ranks = candidate_ranks(pb)
    ps = paper_scores(pb, A)
    rows = []
    for p in range(pb.shape[0]):
        rs = np.flatnonzero(A[p])
        rs = rs[np.argsort(-pb.S[p, rs])]
        rows.append({"paper_id": conf.papers["paper_id"].iloc[p], "title": conf.papers["title"].iloc[p],
                     "reviewers": "; ".join(pb.reviewers[r] for r in rs),
                     "total expertise": round(float(ps[p]), 4),
                     "best rank": int(ranks[p, rs].min()) if len(rs) else None,
                     "ranks": ", ".join(str(int(ranks[p, r])) for r in rs),
                     "eligible reviewers": int((~pb.conflict[p]).sum())})
    return pd.DataFrame(rows)


def reviewer_table(conf: Conference, b: Built, A: np.ndarray) -> pd.DataFrame:
    pb = b.problem
    rows = []
    for r in range(pb.shape[1]):
        ps = np.flatnonzero(A[:, r])
        rows.append({"reviewer": pb.reviewers[r], "reviewer_id": conf.reviewers["reviewer_id"].iloc[r],
                     "papers": len(ps), "max": int(pb.max_load[r]),
                     "mean expertise": round(float(pb.S[ps, r].mean()), 4) if len(ps) else None,
                     "assigned": "; ".join(conf.papers["paper_id"].iloc[p] for p in ps)})
    return pd.DataFrame(rows)


def conflict_table(conf: Conference, b: Built, limit: int | None = None) -> pd.DataFrame:
    items = sorted(b.reasons.items())
    if limit:
        items = items[:limit]
    anonymous = (conf.defaults or {}).get("anonymous")   # demo: reviewers are shown without names, so are co-authors

    def show(why):
        return "co-author of an author of the submission" if anonymous and why.startswith("co-author of") else why
    return pd.DataFrame([{"paper_id": conf.papers["paper_id"].iloc[p], "reviewer": b.problem.reviewers[r], "reason": show(why)}
                         for (p, r), why in items], columns=["paper_id", "reviewer", "reason"])


def excel(conf: Conference, b: Built, A: np.ndarray, settings: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pair_table(conf, b, A).drop(columns=["rank_n"]).to_excel(w, sheet_name="Assignment", index=False)
        paper_table(conf, b, A).to_excel(w, sheet_name="By paper", index=False)
        reviewer_table(conf, b, A).to_excel(w, sheet_name="By reviewer", index=False)
        conflict_table(conf, b).to_excel(w, sheet_name="Conflicts", index=False)
        pd.DataFrame(list(settings.items()), columns=["setting", "value"]).to_excel(w, sheet_name="Settings", index=False)
    return buf.getvalue()


def csv(conf: Conference, b: Built, A: np.ndarray) -> bytes:
    t = pair_table(conf, b, A, explain=False)[["paper_id", "reviewer_id", "reviewer", "expertise", "rank"]]
    return t.to_csv(index=False).encode("utf-8")
