"""The assignment problem: who may review what, how many reviews each paper needs, how many each reviewer takes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Problem:
    S: np.ndarray                         # papers x reviewers, affinity (expertise, possibly blended with bids)
    demand: np.ndarray                    # reviews needed per paper
    max_load: np.ndarray                  # most papers per reviewer
    min_load: np.ndarray = None           # fewest papers per reviewer
    conflict: np.ndarray = None           # papers x reviewers, True = never assign
    forced: np.ndarray = None             # papers x reviewers, True = must assign
    institution: np.ndarray = None        # reviewer -> institution code (-1 = unknown)
    max_per_institution: int | None = None
    senior: np.ndarray = None             # reviewer is experienced
    min_senior: int = 0                   # experienced reviewers needed per paper
    papers: list = field(default_factory=list)
    reviewers: list = field(default_factory=list)

    def __post_init__(self):
        self.S = np.asarray(self.S, dtype=float)
        P, R = self.S.shape
        self.demand = np.broadcast_to(np.asarray(self.demand, dtype=int), (P,)).copy()
        self.max_load = np.broadcast_to(np.asarray(self.max_load, dtype=int), (R,)).copy()
        self.min_load = np.zeros(R, dtype=int) if self.min_load is None else \
            np.broadcast_to(np.asarray(self.min_load, dtype=int), (R,)).copy()
        self.conflict = np.zeros((P, R), dtype=bool) if self.conflict is None else np.asarray(self.conflict, bool)
        self.forced = np.zeros((P, R), dtype=bool) if self.forced is None else np.asarray(self.forced, bool)
        if self.institution is not None:
            self.institution = np.asarray(self.institution, dtype=int)
        if self.senior is not None:
            self.senior = np.asarray(self.senior, dtype=bool)
        if not self.papers:
            self.papers = [f"P{i + 1}" for i in range(P)]
        if not self.reviewers:
            self.reviewers = [f"R{j + 1}" for j in range(R)]

    @property
    def shape(self):
        return self.S.shape

    @property
    def allowed(self) -> np.ndarray:
        return ~self.conflict

    def with_changes(self, **kw) -> "Problem":
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d.update(kw)
        return Problem(**d)

    def issues(self) -> list:
        """Reasons the problem cannot be solved, in plain language (empty list = no obvious obstacle)."""
        P, R = self.S.shape
        out = []
        need, cap, low = int(self.demand.sum()), int(self.max_load.sum()), int(self.min_load.sum())
        if need > cap:
            out.append(f"{need} reviews are needed but reviewers can take at most {cap}: raise the maximum load or add reviewers.")
        if low > need:
            out.append(f"Minimum loads add up to {low} reviews but only {need} are needed: lower the minimum load.")
        if np.any(self.min_load > self.max_load):
            out.append("Some reviewers have a minimum load above their maximum.")
        bad = self.forced & self.conflict
        if bad.any():
            p, r = np.argwhere(bad)[0]
            out.append(f"{bad.sum()} required pairs are also conflicts (e.g. {self.papers[p]} / {self.reviewers[r]}).")
        eligible = (~self.conflict).sum(axis=1)
        short = np.flatnonzero(eligible < self.demand)
        if len(short):
            out.append(f"{len(short)} papers have fewer eligible reviewers than reviews needed (e.g. {self.papers[short[0]]}: "
                       f"{eligible[short[0]]} eligible, {self.demand[short[0]]} needed).")
        over = np.flatnonzero(self.forced.sum(axis=0) > self.max_load)
        if len(over):
            out.append(f"{len(over)} reviewers have more required papers than their maximum load.")
        over_p = np.flatnonzero(self.forced.sum(axis=1) > self.demand)
        if len(over_p):
            out.append(f"{len(over_p)} papers have more required reviewers than reviews needed.")
        if self.min_senior and self.senior is not None:
            ok = ((~self.conflict) & self.senior[None, :]).sum(axis=1)
            few = np.flatnonzero(ok < self.min_senior)
            if len(few):
                out.append(f"{len(few)} papers have no eligible experienced reviewer.")
            if int(self.min_senior) * P > int(self.max_load[self.senior].sum()):
                out.append("Experienced reviewers cannot cover every paper: raise their maximum load or lower the requirement.")
        return out
