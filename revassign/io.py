"""Read a conference from CSV / Excel files, and build the assignment problem from it."""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .conflicts import find_conflicts, norm_institution, split_authors
from .data import Conference
from .expertise import DEFAULT_HALF_LIFE, DEFAULT_TEMPERATURE, ExpertiseModel, build_model, document
from .problem import Problem

COLUMNS = {
    "papers": {"paper_id": ["paper_id", "id", "submission_id", "submission", "paper", "number"],
               "title": ["title", "paper_title"], "abstract": ["abstract", "summary"],
               "authors": ["authors", "author_names", "author"], "emails": ["author_emails", "emails", "email"],
               "institutions": ["author_institutions", "institutions", "affiliations", "affiliation"],
               "keywords": ["keywords", "topics", "subject_areas"], "demand": ["reviews_needed", "demand", "reviews"]},
    "reviewers": {"reviewer_id": ["reviewer_id", "id", "reviewer"], "name": ["name", "full_name", "reviewer_name"],
                  "email": ["email", "e_mail"], "institution": ["institution", "affiliation", "organisation", "organization"],
                  "max_load": ["max_load", "max_papers", "capacity", "max"], "min_load": ["min_load", "min_papers", "min"],
                  "experienced": ["experienced", "senior"], "expertise": ["expertise", "interests", "keywords", "statement"]},
    "profiles": {"reviewer_id": ["reviewer_id", "reviewer", "id"], "title": ["title"], "abstract": ["abstract", "summary"],
                 "year": ["year", "publication_year"], "authors": ["authors", "author_names"]},
    "conflicts": {"paper_id": ["paper_id", "paper", "submission_id"], "reviewer_id": ["reviewer_id", "reviewer"]},
    "bids": {"paper_id": ["paper_id", "paper", "submission_id"], "reviewer_id": ["reviewer_id", "reviewer"],
             "bid": ["bid", "preference", "interest"]},
}
REQUIRED = {"papers": ["paper_id", "title"], "reviewers": ["reviewer_id"], "profiles": ["reviewer_id", "title"],
            "conflicts": ["paper_id", "reviewer_id"], "bids": ["paper_id", "reviewer_id", "bid"]}
BID_WEIGHTS = {"eager": 1.5, "willing": 1.2, "neutral": 1.0, "in a pinch": 0.8, "not willing": 0.5}
BID_ALIASES = {"yes": "eager", "high": "eager", "very high": "eager", "maybe": "willing", "medium": "willing",
               "low": "in a pinch", "no": "not willing", "none": "neutral", "": "neutral", "conflict": "conflict"}


def read_table(data: bytes, name: str) -> pd.DataFrame:
    n = name.lower()
    if n.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    text = data.decode("utf-8-sig", errors="replace")
    sep = "\t" if n.endswith(".tsv") or text.count("\t") > text.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)


def standardise(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Rename columns to the standard names (case and spacing ignored); raise if required ones are missing."""
    norm = {re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_"): c for c in df.columns}
    out = {}
    for std, options in COLUMNS[kind].items():
        for o in options:
            if o in norm and norm[o] not in out.values():
                out[std] = norm[o]
                break
    missing = [c for c in REQUIRED[kind] if c not in out]
    if missing:
        raise ValueError(f"The {kind} file needs a column for: {', '.join(missing)} (found: {', '.join(map(str, df.columns))}).")
    res = pd.DataFrame({std: df[col] for std, col in out.items()})
    for std in COLUMNS[kind]:
        if std not in res:
            res[std] = ""
    return res.fillna("")


def _num(v, default):
    try:
        f = float(str(v).strip())
        return int(f) if np.isfinite(f) else default
    except ValueError:
        return default


def _yes(v) -> bool:
    return str(v).strip().lower() in ("1", "yes", "y", "true", "senior", "experienced", "x")


def conference_from_tables(papers: pd.DataFrame, reviewers: pd.DataFrame, profiles: pd.DataFrame | None = None,
                           conflicts: pd.DataFrame | None = None, bids: pd.DataFrame | None = None,
                           name: str = "Your conference") -> Conference:
    p = standardise(papers, "papers")
    p["paper_id"] = p["paper_id"].astype(str).str.strip()
    if p["paper_id"].duplicated().any():
        raise ValueError(f"Duplicate paper ids: {', '.join(p.loc[p['paper_id'].duplicated(), 'paper_id'].head(5))}")
    p["authors"] = p["authors"].map(split_authors)
    p["emails"] = p["emails"].map(lambda s: [e.strip() for e in re.split(r"[;,|\s]+", str(s)) if "@" in e])
    p["institutions"] = p["institutions"].map(lambda s: [i.strip() for i in re.split(r"[;|]", str(s)) if i.strip()])
    p["demand"] = p["demand"].map(lambda v: _num(v, None))
    r = standardise(reviewers, "reviewers")
    r["reviewer_id"] = r["reviewer_id"].astype(str).str.strip()
    if r["reviewer_id"].duplicated().any():
        raise ValueError(f"Duplicate reviewer ids: {', '.join(r.loc[r['reviewer_id'].duplicated(), 'reviewer_id'].head(5))}")
    r["label"] = np.where(r["name"].astype(str).str.strip() != "", r["name"], r["reviewer_id"])
    r["max_load"] = r["max_load"].map(lambda v: _num(v, None))
    r["min_load"] = r["min_load"].map(lambda v: _num(v, None))
    r["experienced"] = r["experienced"].map(_yes)
    index = {rid: k for k, rid in enumerate(r["reviewer_id"])}
    by_email = {str(e).strip().lower(): k for k, e in enumerate(r["email"]) if "@" in str(e)}
    prof = [[] for _ in range(len(r))]
    unknown = 0
    if profiles is not None and len(profiles):
        f = standardise(profiles, "profiles")
        for _, row in f.iterrows():
            key = str(row["reviewer_id"]).strip()
            k = index.get(key, by_email.get(key.lower()))
            if k is None:
                unknown += 1
                continue
            prof[k].append({"title": row["title"], "abstract": row["abstract"], "year": row["year"],
                            "authors": split_authors(row["authors"])})
    for k, (_, row) in enumerate(r.iterrows()):   # an expertise statement counts as one more "paper"
        if str(row["expertise"]).strip():
            prof[k].append({"title": "Expertise statement", "abstract": str(row["expertise"]), "year": "", "authors": []})
    pid = set(p["paper_id"])
    declared = []
    if conflicts is not None and len(conflicts):
        c = standardise(conflicts, "conflicts")
        declared = [(str(a).strip(), str(b).strip()) for a, b in zip(c["paper_id"], c["reviewer_id"])
                    if str(a).strip() in pid and str(b).strip() in index]
    bid_map = {}
    if bids is not None and len(bids):
        b = standardise(bids, "bids")
        for a, rv, v in zip(b["paper_id"], b["reviewer_id"], b["bid"]):
            a, rv = str(a).strip(), str(rv).strip()
            if a in pid and rv in index:
                v = str(v).strip().lower()
                v = BID_ALIASES.get(v, v)
                if v == "conflict":
                    declared.append((a, rv))
                elif v in BID_WEIGHTS:
                    bid_map[(a, rv)] = v
    conf = Conference(name, p, r, prof, declared, bid_map,
                      description=f"{len(p)} submissions, {len(r)} reviewers, {sum(len(x) for x in prof)} reviewer papers"
                                  + (f" ({unknown} profile rows did not match a reviewer)" if unknown else ""))
    return conf


def read_conference(files: dict) -> Conference:
    """files: {'papers': (bytes, filename), 'reviewers': ..., 'profiles': ..., 'conflicts': ..., 'bids': ...}"""
    t = {k: read_table(*v) for k, v in files.items() if v is not None}
    if "papers" not in t or "reviewers" not in t:
        raise ValueError("A papers file and a reviewers file are needed.")
    return conference_from_tables(t["papers"], t["reviewers"], t.get("profiles"), t.get("conflicts"), t.get("bids"))


# ------------------------------------------------------------------------------------------------ problem building
@dataclass
class Settings:
    demand: int = 3
    max_load: int = 6
    min_load: int = 0
    temperature: float = DEFAULT_TEMPERATURE
    half_life: float | None = DEFAULT_HALF_LIFE
    use_coauthors: bool = True
    coauthor_years: int | None = None
    use_institution: bool = True
    use_email_domain: bool = True
    max_per_institution: int | None = None
    min_senior: int = 0
    use_bids: bool = True
    bid_weights: dict = field(default_factory=lambda: dict(BID_WEIGHTS))


@dataclass
class Built:
    problem: Problem
    expertise: np.ndarray              # papers x reviewers, before bids
    model: ExpertiseModel | None
    reasons: dict                      # (p, r) -> conflict reason
    warnings: list


def build(conf: Conference, st: Settings, model: ExpertiseModel | None = None) -> Built:
    P, R = len(conf.papers), len(conf.reviewers)
    warnings = []
    if conf.S is not None:
        E = np.asarray(conf.S, dtype=float)
        model = None
    else:
        if model is None or model.temperature != st.temperature or model.half_life != st.half_life \
                or model.X_papers.shape[0] != P or model.n_reviewers != R:
            texts = [document(t, a, k) for t, a, k in zip(conf.papers["title"], conf.papers["abstract"], conf.papers["keywords"])]
            docs = [[document(d["title"], d["abstract"]) for d in prof] for prof in conf.profiles]
            years = [[d.get("year") for d in prof] for prof in conf.profiles]
            titles = [[d["title"] for d in prof] for prof in conf.profiles]
            model = build_model(texts, docs, years, st.temperature, st.half_life, titles)
        E = model.scores()
        empty = [conf.reviewers["label"].iloc[k] for k in range(R) if not conf.profiles[k]]
        if empty:
            warnings.append(f"{len(empty)} reviewers have no papers or expertise statement, so their expertise is 0 "
                            f"(e.g. {', '.join(map(str, empty[:3]))}).")
    pidx = {pid: i for i, pid in enumerate(conf.papers["paper_id"])}
    ridx = {rid: j for j, rid in enumerate(conf.reviewers["reviewer_id"])}
    declared = [(pidx[a], ridx[b]) for a, b in conf.declared if a in pidx and b in ridx]
    if conf.conflict is not None:
        C = np.asarray(conf.conflict, dtype=bool).copy()
        reasons = {(int(p), int(r)): "conflict in the bid data" for p, r in np.argwhere(C)}
        for p, r in declared:
            C[p, r] = True
            reasons[(p, r)] = "declared conflict"
    else:
        C, reasons = find_conflicts(
            list(conf.papers["authors"]), list(conf.reviewers["name"]),
            reviewer_profile_authors=[[d.get("authors", []) for d in prof] for prof in conf.profiles],
            reviewer_profile_years=[[d.get("year") for d in prof] for prof in conf.profiles],
            paper_emails=list(conf.papers["emails"]), reviewer_emails=list(conf.reviewers["email"]),
            paper_institutions=list(conf.papers["institutions"]), reviewer_institutions=list(conf.reviewers["institution"]),
            declared=declared, coauthor_years=st.coauthor_years, use_coauthors=st.use_coauthors,
            use_institution=st.use_institution, use_email_domain=st.use_email_domain)
    S = E.copy()
    if st.use_bids and conf.bids:
        for (a, b), v in conf.bids.items():
            if a in pidx and b in ridx:
                S[pidx[a], ridx[b]] *= st.bid_weights.get(v, 1.0)
        S = np.clip(S, 0, None)
    demand = np.array([d if d not in (None, "") and not pd.isna(d) else st.demand
                       for d in conf.papers.get("demand", pd.Series([None] * P))], dtype=int)
    max_load = np.array([v if v not in (None, "") and not pd.isna(v) else st.max_load for v in conf.reviewers["max_load"]], dtype=int)
    min_load = np.array([v if v not in (None, "") and not pd.isna(v) else st.min_load for v in conf.reviewers["min_load"]], dtype=int)
    inst = None
    if st.max_per_institution:
        keys = [norm_institution(x) for x in conf.reviewers["institution"]]
        codes = {k: i for i, k in enumerate(sorted({k for k in keys if k}))}
        inst = np.array([codes.get(k, -1) if k else -1 for k in keys])
        if (inst < 0).all():
            warnings.append("No reviewer institutions are given, so the one-per-institution rule has no effect.")
    senior = conf.reviewers["experienced"].astype(bool).to_numpy() if st.min_senior else None
    pb = Problem(S, demand, max_load, min_load, C, None, inst, st.max_per_institution, senior, st.min_senior,
                 list(conf.papers["paper_id"]), list(conf.reviewers["label"]))
    return Built(pb, E, model, reasons, warnings)
