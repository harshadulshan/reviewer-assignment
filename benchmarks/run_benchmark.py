"""
Benchmarks (results go to benchmarks/results.json, used by the README and the app's Benchmark tab).

1. Expertise: the gold-standard dataset of Stelmakh et al. (2023). 58 researchers rated their expertise (1-5) for
   papers they had read. A method ranks each researcher's papers; the loss is the paper's weighted Kendall tau
   (wrongly ordered pairs, weighted by the gap in reported expertise; 0 = perfect, 0.5 = constant). Same 10
   datasets (reviewer profiles of 20 papers, titles + abstracts) as the paper's Table 3. The aggregation settings
   are chosen by 5-fold cross-validation over researchers, so the reported loss is on researchers not used to
   choose them. Published methods are scored from the predictions their authors released, with the same code.
2. Assignment: ICLR 2018 (911 papers x 2,435 reviewers) and real bids from three AI conferences (PrefLib), the
   instances of Jecmen et al. (2020), plus versions with scarce reviewers, and the demo workshop.

    python benchmarks/run_benchmark.py            # about 10 minutes, downloads ~110 MB on first run
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from itertools import combinations

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from revassign import data, solve  # noqa: E402
from revassign.evaluate import summary, violations  # noqa: E402
from revassign.expertise import build_model, document  # noqa: E402
from revassign.io import Settings, build  # noqa: E402
from revassign.problem import Problem  # noqa: E402

PUBLISHED = {  # name in the released predictions -> (label, kind)
    "tpms": ("TPMS (Charlin & Zemel 2013), used by many conferences", "classic"),
    "elmo": ("ELMo (OpenReview)", "neural"),
    "acl": ("ACL reviewer matcher (Neubig et al.)", "neural"),
    "specter": ("SPECTER (Cohan et al. 2020)", "neural"),
    "specter_mfr": ("SPECTER + MFR (OpenReview)", "neural"),
    "spector2": ("SPECTER2 (Singh et al. 2023)", "neural"),
    "spector2_max": ("SPECTER2, closest paper", "neural"),
    "o1mini": ("OpenAI o1-mini (LLM)", "llm"),
    "claudesonnet35": ("Claude 3.5 Sonnet (LLM)", "llm"),
    "gemini2flash": ("Gemini 2.0 Flash (LLM)", "llm"),
}
TEMPS = [0.0, 5.0, 10.0, 20.0, float("inf")]
HALF_LIVES = [None, 5.0]


# ------------------------------------------------------------------------------------------------ 1. expertise
def pair_terms(refs, reviewers=None):
    """For each reviewer: list of (paper a, paper b, true difference) over pairs rated differently."""
    out = {}
    for r in (reviewers if reviewers is not None else refs):
        out[r] = [(a, b, refs[r][a] - refs[r][b]) for a, b in combinations(list(refs[r]), 2)]
    return out


def loss_parts(preds, terms, reviewers):
    num = den = 0.0
    for r in reviewers:
        for a, b, t in terms[r]:
            d = preds[r][a] - preds[r][b]
            den += abs(t)
            if d * t == 0:
                num += abs(t) / 2
            elif d * t < 0:
                num += abs(t)
    return num, den


def expertise_predictions(subs, arch, refs, temperature, half_life):
    ids = list(subs)
    texts = [document(subs[s]["content"].get("title"), subs[s]["content"].get("abstract")) for s in ids]
    revs = list(refs)
    docs = [[document(p["content"].get("title"), p["content"].get("abstract")) for p in arch[r]] for r in revs]
    years = [[p["content"].get("year") for p in arch[r]] for r in revs]
    m = build_model(texts, docs, years, temperature, half_life)
    E = m.scores()
    col = {s: i for i, s in enumerate(ids)}
    return {r: {p: float(E[col[p], j]) for p in refs[r]} for j, r in enumerate(revs)}


def bootstrap_ci(per_dataset_preds, terms, revs, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    samples = [rng.choice(revs, len(revs), replace=True) for _ in range(n)]
    vals = []
    for s in samples:
        vals.append(np.mean([np.divide(*loss_parts(pr, terms, s)) for pr in per_dataset_preds]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def expertise_benchmark(folds=5, repeats=10):
    data_sets = [data.gold_dataset(i) for i in range(1, 11)]
    refs = data_sets[0][2]
    revs = list(refs)
    terms = pair_terms(refs)
    grid = {}
    t0 = time.time()
    for T in TEMPS:
        for h in HALF_LIVES:
            preds = [expertise_predictions(s, a, refs, T, h) for s, a, _ in data_sets]
            grid[(T, h)] = preds
            num_den = [loss_parts(p, terms, revs) for p in preds]
            print(f"  T={T:<5} half-life={h}: loss {np.mean([n / d for n, d in num_den]):.4f}", flush=True)
    print(f"  grid done in {time.time() - t0:.0f} s", flush=True)
    # cross-validation over researchers
    cv_losses, chosen = [], {}
    for rep in range(repeats):
        order = list(revs)
        random.Random(rep).shuffle(order)
        num = den = 0.0
        for f in range(folds):
            test = order[f::folds]
            train = [r for r in order if r not in test]
            best = min(grid, key=lambda cfg: np.mean([np.divide(*loss_parts(p, terms, train)) for p in grid[cfg]]))
            chosen[str(best)] = chosen.get(str(best), 0) + 1
            parts = [loss_parts(p, terms, test) for p in grid[best]]
            num += np.mean([a for a, _ in parts])
            den += np.mean([b for _, b in parts])
        cv_losses.append(num / den)
    full = {f"T={k[0]}, half-life={k[1]}": float(np.mean([np.divide(*loss_parts(p, terms, revs)) for p in v]))
            for k, v in grid.items()}
    best_all = min(grid, key=lambda cfg: np.mean([np.divide(*loss_parts(p, terms, revs)) for p in grid[cfg]]))
    from revassign.expertise import DEFAULT_HALF_LIFE, DEFAULT_TEMPERATURE
    default = (DEFAULT_TEMPERATURE, DEFAULT_HALF_LIFE)
    out = {"grid": full, "cv_loss": float(np.mean(cv_losses)), "cv_loss_sd": float(np.std(cv_losses)),
           "cv_chosen": chosen, "best_on_all": str(best_all), "default": str(default),
           "default_loss": float(np.mean([np.divide(*loss_parts(p, terms, revs)) for p in grid[default]])),
           "default_ci": bootstrap_ci(grid[default], terms, revs),
           "n_researchers": len(revs), "n_ratings": sum(len(v) for v in refs.values())}
    # published methods, scored from their released predictions with the same loss
    pub = {}
    for algo, (label, kind) in PUBLISHED.items():
        preds = []
        try:
            for i in range(1, 11):
                path = data._get(data.GOLD + f"predictions/{algo}_d_20_{i}_ta.json",
                                 os.path.join(data.CACHE, "gold", "predictions", f"{algo}_d_20_{i}_ta.json"))
                pr = json.load(open(path, encoding="utf-8"))
                preds.append({r: {p: float(pr[r][p]) for p in refs[r]} for r in revs})
        except Exception as e:   # noqa: BLE001
            print(f"  {algo}: not available ({e})")
            continue
        pub[algo] = {"label": label, "kind": kind,
                     "loss": float(np.mean([np.divide(*loss_parts(p, terms, revs)) for p in preds])),
                     "ci": bootstrap_ci(preds, terms, revs)}
        print(f"  {label:55s} {pub[algo]['loss']:.3f}", flush=True)
    out["published"] = pub
    return out


# ------------------------------------------------------------------------------------------------ 2. assignment
def run_methods(pb: Problem, q=0.5, fair_limit=120.0):
    out = {}
    for name, fn in (("greedy", lambda: solve.greedy(pb)), ("max_total", lambda: solve.max_total(pb)),
                     ("fair", lambda: solve.fair(pb, time_limit=fair_limit)),
                     ("randomized", lambda: solve.randomized(pb, q=q, seed=0))):
        t = time.time()
        r = fn()
        sec = time.time() - t
        s = summary(pb, r.A)
        row = {k: s[k] for k in ("total", "worst_paper", "paper_p10", "share_top", "mean_rank", "load_max")}
        row.update({"seconds": sec, "status": r.status, "valid": not violations(pb, r.A),
                    "papers_short": r.info.get("papers_short", 0)})
        if name == "max_total":
            row["lp_bound"] = r.bound
            row["candidates"] = r.info.get("candidates")
        if name == "fair":
            row["floor"] = r.info.get("floor")
            row["floor_bound"] = r.info.get("floor_bound")
        if name == "randomized":
            row["expected_total"] = r.info.get("expected_total")
            row["max_probability"] = r.info.get("max_probability")
        out[name] = row
        print(f"    {name:10s} {r.status:9s} total {s['total']:9.3f} worst {s['worst_paper']:.4f} "
              f"p10 {s['paper_p10']:.4f} top5 {s['share_top']:.3f} {sec:6.1f}s", flush=True)
    return out


def dense_lp_seconds(pb: Problem):
    """Time to solve the same max-total LP with every allowed pair at once (for comparison)."""
    t = time.time()
    m = solve._Model(pb, ~pb.conflict)
    lp = solve._solve_lp(m, pb.S)
    return time.time() - t, float(-lp.fun)


def assignment_benchmark():
    out = {}
    insts = [("iclr2018", data.iclr2018(), None), ("preflib1", data.preflib(1), None),
             ("preflib2", data.preflib(2), None), ("preflib3", data.preflib(3), None)]
    for key, conf, _ in insts:
        d = conf.defaults
        pb = Problem(conf.S, d["demand"], d["max_load"], conflict=conf.conflict)
        print(f"  {conf.name}: capacity {pb.max_load.sum() / pb.demand.sum():.2f}x demand", flush=True)
        out[key] = {"name": conf.name, "papers": pb.shape[0], "reviewers": pb.shape[1], "demand": int(d["demand"]),
                    "max_load": int(d["max_load"]), "capacity": float(pb.max_load.sum() / pb.demand.sum()),
                    "methods": run_methods(pb)}
        if key == "iclr2018":
            sec, val = dense_lp_seconds(pb)
            out[key]["dense_lp_seconds"], out[key]["dense_lp_total"] = sec, val
    # scarce reviewers: where fairness matters
    iclr = data.iclr2018()
    cols = np.sort(np.random.default_rng(0).choice(iclr.S.shape[1], 1000, replace=False))
    pb = Problem(iclr.S[:, cols], 3, 3, conflict=iclr.conflict[:, cols])
    print("  ICLR 2018, 1,000 reviewers x 3 papers (capacity 1.10x)", flush=True)
    out["iclr2018_scarce"] = {"name": "ICLR 2018, scarce reviewers (1,000 reviewers, at most 3 each)", "papers": 911,
                              "reviewers": 1000, "demand": 3, "max_load": 3, "capacity": 3000 / 2733,
                              "methods": run_methods(pb)}
    p3 = data.preflib(3)
    pb = Problem(p3.S, 3, 4, conflict=p3.conflict)
    print("  AI conference 3, at most 4 papers per reviewer", flush=True)
    out["preflib3_scarce"] = {"name": "AI conference 3 bids, at most 4 per reviewer", "papers": pb.shape[0],
                              "reviewers": pb.shape[1], "demand": 3, "max_load": 4,
                              "capacity": float(pb.max_load.sum() / pb.demand.sum()), "methods": run_methods(pb)}
    # demo workshop, plus the gold-standard check
    conf = data.demo_conference()
    b = build(conf, Settings(demand=3, max_load=9))
    print("  Demo workshop", flush=True)
    out["demo"] = {"name": conf.name, "papers": b.problem.shape[0], "reviewers": b.problem.shape[1], "demand": 3,
                   "max_load": 9, "capacity": float(b.problem.max_load.sum() / b.problem.demand.sum()),
                   "methods": run_methods(b.problem), "conflicts": int(b.problem.conflict.sum())}
    r = solve.max_total(b.problem)
    ridx = {rid: j for j, rid in enumerate(conf.reviewers.reviewer_id)}
    pidx = {x: i for i, x in enumerate(conf.papers.paper_id)}
    a_, n_ = [], []
    for rid, g in conf.gold.items():
        for pid, e in g.items():
            if b.problem.conflict[pidx[pid], ridx[rid]]:
                continue                     # the researcher wrote it: never assignable
            (a_ if r.A[pidx[pid], ridx[rid]] else n_).append(e)
    out["demo"]["gold"] = {"assigned_n": len(a_), "assigned_mean": float(np.mean(a_)), "other_n": len(n_),
                           "other_mean": float(np.mean(n_)), "assigned_4plus": float(np.mean(np.array(a_) >= 4)),
                           "other_4plus": float(np.mean(np.array(n_) >= 4))}
    return out


def randomized_curve():
    iclr = data.iclr2018()
    pb = Problem(iclr.S, 3, 6, conflict=iclr.conflict)
    best = solve.max_total(pb).objective
    out = []
    for q in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        r = solve.randomized(pb, q=q, seed=0)
        out.append({"q": q, "expected_total": r.info["expected_total"], "share_of_best": r.info["expected_total"] / best})
        print(f"    q={q}: {r.info['expected_total'] / best:.3f} of the best total", flush=True)
    return {"best_total": best, "points": out}


def marginals_check(n_samples=2000):
    """Sampling check: how often each pair is drawn vs its planned probability (small real instance)."""
    p1 = data.preflib(1)
    pb = Problem(p1.S, 3, 6, conflict=p1.conflict)
    r = solve.randomized(pb, q=0.5, seed=0)
    pi, ri, x = r.info["prob_pairs"]
    acc = np.zeros(len(x))
    bad = 0
    for s in range(n_samples):
        y = solve.dependent_round(pi, ri, x, random.Random(1000 + s))
        A = np.zeros(pb.shape, bool)
        A[pi, ri] = y > 0.5
        bad += bool(violations(pb, A))
        acc += y
    dev = np.abs(acc / n_samples - x)
    return {"samples": n_samples, "invalid": bad, "max_abs_deviation": float(dev.max()), "max_probability": float(x.max()),
            "pairs": int(len(x))}


def main():
    t0 = time.time()
    res = {}
    print("1. expertise (gold standard)", flush=True)
    res["expertise"] = expertise_benchmark()
    print("2. assignment", flush=True)
    res["assignment"] = assignment_benchmark()
    print("3. randomised: expected expertise vs probability cap (ICLR 2018)", flush=True)
    res["randomized_curve"] = randomized_curve()
    res["marginals"] = marginals_check()
    print("  marginals:", res["marginals"], flush=True)
    res["seconds"] = time.time() - t0
    res["machine"] = f"{os.cpu_count()} cores, Python {sys.version.split()[0]}"
    with open(os.path.join(ROOT, "benchmarks", "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote results.json in {res['seconds']:.0f} s")


if __name__ == "__main__":
    main()
