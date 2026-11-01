"""
Conflicts of interest: pairs that must never be assigned.

* Author:       the reviewer is an author of the submission (name or e-mail match).
* Co-author:    the reviewer has published with an author of the submission (from the author lists of the
                reviewer's own papers), optionally only within the last N years.
* Institution:  same institution, or the same e-mail domain (public mail providers are ignored).
* Declared:     pairs listed by the chair, or reviewers who bid "conflict".

Names are compared after removing accents and punctuation. When both names have a full first name, the full
names must match; when either side only has an initial ("J. Smith"), the initial and surname must match. This
errs on the side of flagging a conflict, which is the safe direction.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

import numpy as np

PUBLIC_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.uk", "hotmail.com", "outlook.com", "live.com", "msn.com",
    "icloud.com", "me.com", "aol.com", "proton.me", "protonmail.com", "gmx.com", "gmx.de", "mail.com", "qq.com",
    "163.com", "126.com", "yandex.ru", "zoho.com",
}


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


def name_parts(name: str):
    """(first, last) in lower case, or None. 'Smith, John' and 'John Q. Smith' both give ('john', 'smith')."""
    s = _ascii(name or "").strip()
    if not s:
        return None
    if "," in s:
        last, first = [p.strip() for p in s.split(",", 1)]
        s = f"{first} {last}"
    tokens = [t for t in re.split(r"[\s.]+", re.sub(r"[^A-Za-z\s.\-']", " ", s).lower()) if t]
    tokens = [t.strip("-'") for t in tokens if t.strip("-'")]
    if not tokens:
        return None
    if len(tokens) == 1:
        return ("", tokens[0])
    return (tokens[0], tokens[-1])


def same_person(a, b) -> bool:
    if a is None or b is None or a[1] != b[1]:
        return False
    fa, fb = a[0], b[0]
    if not fa or not fb:
        return False
    if len(fa) > 1 and len(fb) > 1:
        return fa == fb
    return fa[0] == fb[0]


def split_authors(value) -> list:
    """Authors from a cell: a list, or text separated by ';', '|' or ' and '. Commas inside 'Last, First' are
    kept when the text has semicolons."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
        return [t.strip().strip("'\"") for t in re.split(r"',\s*'|\",\s*\"|',\s*\"|\",\s*'", s) if t.strip().strip("'\"")]
    if ";" in s or "|" in s:
        return [t.strip() for t in re.split(r"[;|]", s) if t.strip()]
    return [t.strip() for t in re.split(r",|\band\b", s) if t.strip()]


def email_domain(email) -> str:
    e = str(email or "").strip().lower()
    if "@" not in e:
        return ""
    d = e.split("@", 1)[1]
    return "" if d in PUBLIC_DOMAINS else d


def norm_institution(s) -> str:
    s = _ascii(s or "").lower()
    s = re.sub(r"\b(the|of|and|at|dept|department|faculty|school)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


class NameIndex:
    """Fast lookup of people by surname."""

    def __init__(self):
        self.by_last = defaultdict(list)

    def add(self, name, payload):
        parts = name_parts(name)
        if parts:
            self.by_last[parts[1]].append((parts, payload))

    def match(self, name):
        parts = name_parts(name)
        if not parts:
            return []
        return [(p, pl) for p, pl in self.by_last.get(parts[1], []) if same_person(parts, p)]


def find_conflicts(paper_authors: list, reviewer_names: list, reviewer_profile_authors: list | None = None,
                   reviewer_profile_years: list | None = None, paper_emails: list | None = None,
                   reviewer_emails: list | None = None, paper_institutions: list | None = None,
                   reviewer_institutions: list | None = None, declared: list | None = None,
                   coauthor_years: int | None = None, use_coauthors: bool = True, use_institution: bool = True,
                   use_email_domain: bool = True):
    """Returns (conflict matrix papers x reviewers, {(p, r): reason}).

    paper_authors: list of author-name lists per submission. reviewer_profile_authors: for each reviewer, the
    author lists of their own papers (to find co-authors). declared: (paper index, reviewer index) pairs."""
    P, R = len(paper_authors), len(reviewer_names)
    C = np.zeros((P, R), dtype=bool)
    reasons = {}

    def flag(p, r, why):
        if not C[p, r]:
            C[p, r] = True
            reasons[(p, r)] = why

    # index every submission author once
    authors = NameIndex()
    for p, names in enumerate(paper_authors):
        for a in names:
            authors.add(a, (p, a))
    # 1. the reviewer wrote the submission
    for r, name in enumerate(reviewer_names):
        for _, (p, a) in authors.match(name):
            flag(p, r, "author of the submission")
    if paper_emails is not None and reviewer_emails is not None:
        by_email = defaultdict(set)
        for p, emails in enumerate(paper_emails):
            for e in emails:
                if e and "@" in str(e):
                    by_email[str(e).strip().lower()].add(p)
        for r, e in enumerate(reviewer_emails):
            for p in by_email.get(str(e or "").strip().lower(), ()):
                flag(p, r, "author of the submission (same e-mail)")
    # 2. co-authors
    if use_coauthors and reviewer_profile_authors is not None:
        newest = None
        if coauthor_years and reviewer_profile_years is not None:
            ys = [float(y) for yy in reviewer_profile_years for y in yy if str(y).replace(".", "", 1).isdigit()]
            newest = max(ys) if ys else None
        for r, papers in enumerate(reviewer_profile_authors):
            me = name_parts(reviewer_names[r])
            years = reviewer_profile_years[r] if reviewer_profile_years is not None else [None] * len(papers)
            for names, y in zip(papers, years):
                if newest is not None and str(y).replace(".", "", 1).isdigit() and float(y) < newest - coauthor_years:
                    continue
                for co in names:
                    if same_person(name_parts(co), me):
                        continue
                    for _, (p, a) in authors.match(co):
                        flag(p, r, f"co-author of {a}")
    # 3. institution / e-mail domain
    if use_institution and paper_institutions is not None and reviewer_institutions is not None:
        by_inst = defaultdict(set)
        for p, insts in enumerate(paper_institutions):
            for i in insts:
                k = norm_institution(i)
                if k:
                    by_inst[k].add(p)
        for r, inst in enumerate(reviewer_institutions):
            k = norm_institution(inst)
            for p in by_inst.get(k, ()) if k else ():
                flag(p, r, f"same institution ({inst})")
    if use_email_domain and paper_emails is not None and reviewer_emails is not None:
        by_dom = defaultdict(set)
        for p, emails in enumerate(paper_emails):
            for e in emails:
                d = email_domain(e)
                if d:
                    by_dom[d].add(p)
        for r, e in enumerate(reviewer_emails):
            d = email_domain(e)
            for p in by_dom.get(d, ()) if d else ():
                flag(p, r, f"same e-mail domain ({d})")
    # 4. declared by the chair or the reviewer
    for p, r in declared or []:
        flag(p, r, "declared conflict")
    return C, reasons
