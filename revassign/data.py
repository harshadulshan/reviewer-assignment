"""
Data: a conference = submissions + reviewers (+ their own papers) + optional bids and declared conflicts.

Public data used (downloaded on first use from pinned commits and cached in data_cache/):

* Gold-standard reviewer expertise (Stelmakh, Wieting, Xi, Neubig & Shah, TMLR 2025): 58 researchers rated their
  expertise (1-5) for 477 papers they had read; their 20 most recent papers form their profiles.
  github.com/niharshah/goldstandard-reviewer-paper-match. Evaluations CC BY 4.0; titles, abstracts and
  author lists are Semantic Scholar metadata and are downloaded, not redistributed.
* ICLR 2018 similarity matrix (911 papers x 2,435 reviewers; Xu, Zhao, Shi & Shah, IJCAI 2019), MIT licence.
  github.com/xycforgithub/StrategyProof_Conference_Review
* Bids from 3 AI conferences (PrefLib 00039 csconf; Mattei & Walsh 2013): yes / maybe / no response, conflicts.
  github.com/PrefLib/PrefLib-Data
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .conflicts import name_parts, same_person, split_authors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data_cache")
GOLD = "https://raw.githubusercontent.com/niharshah/goldstandard-reviewer-paper-match/00296730e3ebe058c7b4cc567cd52208ca80735c/"
ICLR = ("https://raw.githubusercontent.com/xycforgithub/StrategyProof_Conference_Review/"
        "26934e04f60df433a937a5d3e86a2fc97f91cfba/iclr2018.npz")
ICLR_SHA256 = "b2bec165c95d96c06c2e7ec1fcdac82344c5bec3933ae7f7b94026c66c2ed157"
PREFLIB = ("https://raw.githubusercontent.com/PrefLib/PrefLib-Data/03f5a88df70668ea0ab30062562fd57bf721d360/"
           "datasets/00039%20-%20csconf/00039-0000000{i}.cat")
PREFLIB_SHA256 = {1: "70451344d9845a546164e05b59637a87d97c123d515052dbd3facfd29b46841d",
                  2: "a5f7c9c6c0173f8430a4b69c4f76fa2ab02d85a0917f806b18dd6cb86dc48e5a",
                  3: "970a2b132e825bac8a669803055d90118eababe303db2a06cc911cf158057718"}


@dataclass
class Conference:
    name: str
    papers: pd.DataFrame                       # paper_id, title, abstract, authors (list), emails (list), institutions (list), keywords, demand
    reviewers: pd.DataFrame                    # reviewer_id, label, name, email, institution, max_load, min_load, experienced
    profiles: list = field(default_factory=list)   # per reviewer: list of dicts title, abstract, year, authors (list)
    declared: list = field(default_factory=list)   # (paper_id, reviewer_id)
    bids: dict = field(default_factory=dict)       # (paper_id, reviewer_id) -> bid text
    S: np.ndarray | None = None                # given affinity (benchmark instances without text)
    conflict: np.ndarray | None = None         # given conflicts
    gold: dict = field(default_factory=dict)   # reviewer_id -> {paper_id: self-reported expertise 1-5}
    description: str = ""
    defaults: dict = field(default_factory=dict)

    @property
    def has_text(self) -> bool:
        return self.S is None


# ------------------------------------------------------------------------------------------------ download helpers
def _get(url: str, path: str, timeout: int = 60) -> str:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "reviewer-assignment"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def _get_many(pairs, workers: int = 8):
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda uv: _get(*uv), pairs))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(path: str, expected: str):
    if sha256(path) != expected:
        os.remove(path)
        raise ValueError(f"{os.path.basename(path)} does not match the published file (checksum); it was removed, try again.")


# ------------------------------------------------------------------------------------------------ gold standard
def gold_evaluations() -> dict:
    path = _get(GOLD + "data/evaluations.csv", os.path.join(CACHE, "gold", "evaluations.csv"))
    refs = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            r = str(row["ParticipantID"])
            refs[r] = {row[f"Paper{x}"]: float(row[f"Expertise{x}"]) for x in range(1, 11)
                       if row.get(f"Paper{x}") and row.get(f"Expertise{x}")}
    return refs


def gold_dataset(i: int = 1):
    """Submissions and reviewer profiles of the gold-standard dataset d_20_i (OpenReview format)."""
    refs = gold_evaluations()
    base = f"evaluation_datasets/d_20_{i}/"
    local = os.path.join(CACHE, "gold", f"d_20_{i}")
    jobs = [(GOLD + base + "submissions.json", os.path.join(local, "submissions.json"))]
    jobs += [(GOLD + base + f"archives/~{r}.jsonl", os.path.join(local, "archives", f"~{r}.jsonl")) for r in refs]
    _get_many(jobs)
    subs = json.load(open(jobs[0][1], encoding="utf-8"))
    arch = {}
    for r in refs:
        with open(os.path.join(local, "archives", f"~{r}.jsonl"), encoding="utf-8") as f:
            arch[r] = [json.loads(line) for line in f if line.strip()]
    return subs, arch, refs


def _infer_name(papers: list) -> str:
    """A researcher's own name = the author appearing on most of their papers."""
    counts, spelling = Counter(), {}
    for p in papers:
        seen = set()
        for a in split_authors(p["content"].get("authors")):
            key = name_parts(a)
            if key and key not in seen:
                seen.add(key)
                k = (key[0][:1], key[1])
                counts[k] += 1
                if len(a) > len(spelling.get(k, "")):
                    spelling[k] = a
    if not counts:
        return ""
    return spelling[counts.most_common(1)[0][0]]


def demo_conference(n_papers: int = 150, seed: int = 7) -> Conference:
    """A realistic workshop built from the gold-standard data: 150 real submissions, 58 real researchers as
    reviewers (shown anonymously), their own 20 papers as profiles, conflicts from real author lists."""
    subs, arch, refs = gold_dataset(1)
    ids = sorted(subs)
    random.Random(seed).shuffle(ids)
    ids = sorted(ids[:n_papers])
    rows = []
    for pid in ids:
        c = subs[pid]["content"]
        rows.append({"paper_id": pid[:10], "title": c.get("title") or "", "abstract": c.get("abstract") or "",
                     "authors": split_authors(c.get("authors")), "emails": [], "institutions": [], "keywords": "",
                     "year": c.get("year"), "full_id": pid})
    papers = pd.DataFrame(rows)
    revs, profiles = [], []
    for k, r in enumerate(sorted(refs, key=lambda x: int(x))):
        prof = [{"title": p["content"].get("title") or "", "abstract": p["content"].get("abstract") or "",
                 "year": p["content"].get("year"), "authors": split_authors(p["content"].get("authors")),
                 "id": p["id"]} for p in arch[r]]
        years = [int(p["year"]) for p in prof if str(p["year"]).isdigit()]
        revs.append({"reviewer_id": f"R{k + 1:02d}", "label": f"Reviewer {k + 1:02d}", "name": _infer_name(arch[r]),
                     "email": "", "institution": "", "max_load": None, "min_load": None,
                     "experienced": len(prof) >= 15, "source_id": r})
        profiles.append(prof)
    reviewers = pd.DataFrame(revs)
    # a reviewer whose own profile contains the submission wrote it
    declared = []
    full_to_short = dict(zip(papers["full_id"], papers["paper_id"]))
    for (_, rv), prof in zip(reviewers.iterrows(), profiles):
        for p in prof:
            if p["id"] in full_to_short:
                declared.append((full_to_short[p["id"]], rv["reviewer_id"]))
    gold = {}
    for _, rv in reviewers.iterrows():
        g = {full_to_short[p]: e for p, e in refs[rv["source_id"]].items() if p in full_to_short}
        if g:
            gold[rv["reviewer_id"]] = g
    return Conference("Demo workshop (real papers and researchers)", papers, reviewers, profiles, declared, {}, gold=gold,
                      description=f"{len(papers)} real submissions and 58 real researchers from the gold-standard reviewer "
                                  "expertise dataset (Stelmakh et al.). Reviewers are shown anonymously; their own 20 most recent "
                                  "papers are their profiles. 3 reviews per paper, at most 9 per reviewer.",
                      defaults={"demand": 3, "max_load": 9, "anonymous": True})


# ------------------------------------------------------------------------------------------------ assignment benchmarks
def iclr2018() -> Conference:
    path = _get(ICLR, os.path.join(CACHE, "iclr2018.npz"))
    _check(path, ICLR_SHA256)
    with zipfile.ZipFile(path) as z:   # read the two numeric arrays only (no pickled objects)
        S = np.load(io.BytesIO(z.read("similarity_matrix.npy")), allow_pickle=False).T
        M = np.load(io.BytesIO(z.read("mask_matrix.npy")), allow_pickle=False).T.astype(bool)
    P, R = S.shape
    return _matrix_conference("ICLR 2018 (911 papers, 2,435 reviewers)", S, M, 3, 6,
                              "Text-similarity scores between ICLR 2018 submissions and 2,435 researchers (Xu et al. 2019), "
                              "the instance used by Jecmen et al. (2020). 3 reviews per paper, at most 6 per reviewer.")


def preflib(i: int) -> Conference:
    path = _get(PREFLIB.format(i=i), os.path.join(CACHE, "preflib", f"00039-0000000{i}.cat"))
    _check(path, PREFLIB_SHA256[i])
    P = None
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.startswith("# NUMBER ALTERNATIVES"):
            P = int(line.split(":")[1])
        elif line.strip() and not line.startswith("#"):
            count, body = line.split(":", 1)
            groups = body.strip().split("},")
            row = np.full(P, -1.0)
            for value, g in zip((1.0, 0.5, 0.25), groups):
                for a in g.replace("{", "").replace("}", "").split(","):
                    if a.strip():
                        row[int(a) - 1] = value
            rows += [row] * int(count)
    S = np.array(rows).T
    M = S < 0
    S[M] = 0.0
    loads = {1: 6, 2: 7, 3: 6}[i]
    return _matrix_conference(f"AI conference {i} bids (PrefLib)", S, M, 3, loads,
                              f"Real bids from an AI conference (PrefLib 00039-{i}): yes = 1, maybe = 0.5, no response = 0.25; "
                              f"missing = conflict. {S.shape[0]} papers, {S.shape[1]} reviewers, 3 reviews per paper, at most {loads} per reviewer.")


def _matrix_conference(name, S, M, demand, max_load, description) -> Conference:
    P, R = S.shape
    papers = pd.DataFrame({"paper_id": [f"P{i + 1}" for i in range(P)], "title": [f"Paper {i + 1}" for i in range(P)],
                           "abstract": "", "authors": [[] for _ in range(P)], "emails": [[] for _ in range(P)],
                           "institutions": [[] for _ in range(P)], "keywords": ""})
    reviewers = pd.DataFrame({"reviewer_id": [f"R{j + 1}" for j in range(R)], "label": [f"Reviewer {j + 1}" for j in range(R)],
                              "name": "", "email": "", "institution": "", "max_load": None, "min_load": None, "experienced": False})
    return Conference(name, papers, reviewers, [[] for _ in range(R)], S=S, conflict=M, description=description,
                      defaults={"demand": demand, "max_load": max_load})
