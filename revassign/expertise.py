"""
Expertise: how well does a reviewer's own work match a submission?

Each reviewer is represented by their own papers (title + abstract, optionally a short expertise statement).
Submissions and reviewer papers become TF-IDF vectors over words and word pairs (log-scaled counts, so one
repeated word cannot dominate). The cosine similarity is computed between a submission and every one of the
reviewer's papers, and these are combined with a soft maximum:

    expertise = (1/T) * log( sum_d w_d * exp(T * sim_d) ),   with sum_d w_d = 1

T = 0 is the plain (weighted) mean, T = infinity is the single closest paper; in between, a reviewer with a few
very close papers scores higher than one with many loosely related ones. The weights w_d optionally favour
recent papers (half-life in years), since reviewers' interests move.

Tested on the gold-standard dataset of Stelmakh et al. (2023): see benchmarks/run_benchmark.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# chosen by cross-validation on the gold-standard dataset (benchmarks/results.json)
DEFAULT_TEMPERATURE = 5.0
DEFAULT_HALF_LIFE = 5.0       # years; None = all papers count the same
LATEX = re.compile(r"\$[^$]*\$|\\[a-zA-Z]+")
# words that say nothing about the topic: left out of explanations (they still count in the score)
GENERIC = set(ENGLISH_STOP_WORDS) | {
    "paper", "papers", "propose", "proposed", "proposes", "approach", "approaches", "method", "methods", "present",
    "presents", "show", "shows", "shown", "study", "studies", "result", "results", "new", "novel", "use", "used",
    "using", "based", "work", "problem", "problems", "different", "existing", "provide", "provides", "able",
    "demonstrate", "experiments", "experimental", "performance", "set", "large", "small", "high", "low", "way",
    "ways", "make", "makes", "including", "introduce", "evaluate", "evaluated", "compare", "compared", "test",
    "data", "model", "models", "analysis", "framework", "task", "tasks", "state", "art", "recent", "recently",
    "achieve", "achieves", "significantly", "effective", "efficient", "simple", "general", "various", "learn",
    "learning", "level", "time", "field", "real", "world", "number", "important", "common", "end", "case",
}


def clean(text: str) -> str:
    text = LATEX.sub(" ", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def document(title: str = "", abstract: str = "", keywords: str = "") -> str:
    parts = [clean(title), clean(keywords), clean(abstract)]
    return ". ".join(p for p in parts if p)


def recency_weights(years, half_life: float | None) -> np.ndarray:
    years = np.asarray(years, dtype=float)
    if len(years) == 0:
        return years
    if half_life is None or not np.isfinite(years).any():
        return np.full(len(years), 1.0 / len(years))
    known = np.isfinite(years)
    ref = np.nanmax(years)
    filled = np.where(known, years, np.nanmedian(years) if known.any() else ref)
    w = np.exp(-(ref - filled) * np.log(2) / half_life)
    return w / w.sum()


def soft_max(sims: np.ndarray, weights: np.ndarray, temperature: float) -> np.ndarray:
    """Weighted soft maximum over the columns of `sims` (one row per submission)."""
    if sims.shape[1] == 0:
        return np.zeros(sims.shape[0])
    if temperature == 0:
        return sims @ weights
    if np.isinf(temperature):
        return sims.max(axis=1)
    m = sims.max(axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(temperature * (sims - m)) @ weights) / temperature)


@dataclass
class ExpertiseModel:
    vectorizer: TfidfVectorizer
    X_papers: sparse.csr_matrix          # submissions x terms
    X_docs: sparse.csr_matrix            # reviewer papers x terms
    doc_owner: np.ndarray                # reviewer index of each reviewer paper
    doc_titles: list
    doc_years: np.ndarray
    n_reviewers: int
    temperature: float = DEFAULT_TEMPERATURE
    half_life: float | None = DEFAULT_HALF_LIFE
    _terms: np.ndarray = field(default=None, repr=False)

    def doc_slices(self):
        starts = np.searchsorted(self.doc_owner, np.arange(self.n_reviewers))
        ends = np.searchsorted(self.doc_owner, np.arange(self.n_reviewers), side="right")
        return starts, ends

    def scores(self, block: int = 4000, X=None) -> np.ndarray:
        """Expertise of every reviewer for every submission, in [0, 1] (submissions x reviewers). X: other
        texts already turned into vectors (see score_texts)."""
        X = self.X_papers if X is None else X
        P = X.shape[0]
        out = np.zeros((P, self.n_reviewers))
        starts, ends = self.doc_slices()
        r = 0
        while r < self.n_reviewers:
            r_end = r
            while r_end < self.n_reviewers and ends[r_end] - starts[r] <= block:
                r_end += 1
            r_end = max(r_end, r + 1)
            d0, d1 = starts[r], ends[r_end - 1]
            sims = (X @ self.X_docs[d0:d1].T).toarray()
            for j in range(r, r_end):
                a, b = starts[j] - d0, ends[j] - d0
                if b > a:
                    w = recency_weights(self.doc_years[starts[j]:ends[j]], self.half_life)
                    out[:, j] = soft_max(sims[:, a:b], w, self.temperature)
            r = r_end
        return np.clip(out, 0, 1)

    def score_texts(self, texts: list) -> np.ndarray:
        """Expertise of every reviewer for new texts (e.g. a pasted abstract): texts x reviewers."""
        return self.scores(X=self.vectorizer.transform([clean(t) for t in texts]).tocsr())

    def explain(self, p: int, r: int, n_terms: int = 6) -> dict:
        """The reviewer paper closest to submission p, and the terms they share."""
        return self.explain_vector(self.X_papers[p], r, n_terms)

    def explain_text(self, text: str, r: int, n_terms: int = 6) -> dict:
        return self.explain_vector(self.vectorizer.transform([clean(text)]).tocsr(), r, n_terms)

    def explain_vector(self, x, r: int, n_terms: int = 6) -> dict:
        starts, ends = self.doc_slices()
        a, b = starts[r], ends[r]
        if b <= a:
            return {"closest": None, "similarity": 0.0, "terms": []}
        sims = (self.X_docs[a:b] @ x.T).toarray().ravel()
        d = a + int(np.argmax(sims))
        contrib = x.multiply(self.X_docs[d]).tocoo()
        if self._terms is None:
            self._terms = self.vectorizer.get_feature_names_out()
        terms = []
        for i in np.argsort(-contrib.data):
            t = self._terms[contrib.col[i]]
            words = t.split()
            if any(w in ENGLISH_STOP_WORDS or w.isdigit() or len(w) < 3 for w in words) or all(w in GENERIC for w in words):
                continue                                   # 'of the', 'models for', 'data analysis' say nothing
            if len(words) == 1 and any(t in s.split() for s in terms if " " in s):
                continue                                   # already part of a chosen phrase
            if len(words) > 1:
                terms = [s for s in terms if s not in words]   # a phrase replaces its single words
            if t not in terms:
                terms.append(t)
            if len(terms) >= n_terms:
                break
        return {"closest": self.doc_titles[d], "similarity": float(sims.max()), "terms": terms}


def build_model(paper_texts: list, reviewer_docs: list, reviewer_years: list | None = None,
                temperature: float = DEFAULT_TEMPERATURE, half_life: float | None = DEFAULT_HALF_LIFE,
                titles: list | None = None) -> ExpertiseModel:
    """paper_texts: one string per submission. reviewer_docs: for each reviewer, a list of strings (their papers).
    reviewer_years: matching lists of publication years (None or NaN where unknown)."""
    owner, docs, years, doc_titles = [], [], [], []
    for r, ds in enumerate(reviewer_docs):
        for i, d in enumerate(ds):
            owner.append(r)
            docs.append(d)
            y = reviewer_years[r][i] if reviewer_years is not None and i < len(reviewer_years[r]) else None
            years.append(float(y) if y not in (None, "") and str(y).replace(".", "", 1).isdigit() else np.nan)
            doc_titles.append(titles[r][i] if titles is not None else d[:120])
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, dtype=np.float64)
    corpus = list(paper_texts) + docs
    if not any(t.strip() for t in corpus):
        raise ValueError("There is no text to compare: add titles or abstracts.")
    vec.fit(corpus)
    return ExpertiseModel(vec, vec.transform(paper_texts).tocsr(), vec.transform(docs).tocsr() if docs else
                          sparse.csr_matrix((0, len(vec.vocabulary_))), np.asarray(owner, dtype=int), doc_titles,
                          np.asarray(years, dtype=float), len(reviewer_docs), temperature, half_life)
