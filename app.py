"""
Reviewer Assignment
-------------------
Match submissions to reviewers by expertise, never with a conflict of interest, within every reviewer's
workload, and fairly across papers. Expertise is tested on a public gold-standard dataset; assignments are
proven optimal; re-planning after a drop-out keeps as many existing pairs as possible.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from revassign import data, solve
from revassign.evaluate import candidate_ranks, paper_scores, summary, violations
from revassign.expertise import DEFAULT_HALF_LIFE, DEFAULT_TEMPERATURE
from revassign.io import BID_WEIGHTS, COLUMNS, REQUIRED, Settings, build, read_conference
from revassign.report import conflict_table, csv, excel, pair_table, paper_table, reviewer_table

st.set_page_config(page_title="Reviewer Assignment", page_icon="🧑‍🏫", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "benchmarks", "results.json")
SAMPLES = os.path.join(HERE, "sample_data")
C_OPT, C_START, C_OK, C_BAD, C_WARN, C_BEST = "#3b82f6", "#8b949e", "#2ea043", "#f85149", "#f0883e", "#a371f7"
METHODS = {"max_total": "Highest total expertise", "fair": "Fair: protect the worst-off paper",
           "randomized": "Randomised: cap any one pair's chance", "greedy": "Greedy (baseline)"}
METHOD_COLORS = {"max_total": C_OPT, "fair": C_BEST, "randomized": C_WARN, "greedy": C_START}
BENCH = {"iclr2018": "ICLR 2018 · 911 papers × 2,435 reviewers",
         "preflib1": "AI conference 1 bids · 54 × 31",
         "preflib2": "AI conference 2 bids · 52 × 24",
         "preflib3": "AI conference 3 bids · 176 × 146"}
TEMPS = {0.0: "Average of all their papers", 2.0: "Mostly average", 5.0: "Balanced (tested best)",
         10.0: "Mostly closest papers", 20.0: "Closest papers", float("inf"): "Single closest paper"}
HALF_LIVES = {None: "All papers count the same", 3.0: "3 years", 5.0: "5 years (tested best)", 10.0: "10 years"}
PRESETS = {
    "Conference": dict(rv_demand=3, rv_max=6, rv_min=0, inst_cap=0, min_senior=0),
    "Journal special issue": dict(rv_demand=2, rv_max=3, rv_min=0, inst_cap=1, min_senior=1),
    "Thesis examiners": dict(rv_demand=2, rv_max=2, rv_min=0, inst_cap=1, min_senior=1),
}

ss = st.session_state
for key, default in {"conf": None, "conf_key": None, "built": None, "built_sig": None, "model": None,
                     "runs": {}, "active": None, "replan": None, "compare": None, "bench_live": None,
                     "rv_demand": 3, "rv_max": 6, "rv_min": 0, "c_coauth": True, "c_years": 0, "c_inst": True,
                     "c_email": True, "inst_cap": 0, "min_senior": 0, "use_bids": True,
                     "temperature": DEFAULT_TEMPERATURE, "half_life": DEFAULT_HALF_LIFE}.items():
    ss.setdefault(key, default)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fmt(n) -> str:
    return f"{n:,}"


def layout(fig, height=320, **kw):
    top = 78 if kw.get("title") else 36
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="closest", **kw)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)")
    return fig


@st.cache_resource(show_spinner="Downloading the demo data (first time only, about 2 MB)…")
def demo_conf():
    return data.demo_conference()


@st.cache_resource(show_spinner="Downloading the benchmark instance (first time only)…")
def bench_conf(key):
    return data.iclr2018() if key == "iclr2018" else data.preflib(int(key[-1]))


@st.cache_data(show_spinner=False)
def load_results(mtime):
    with open(RESULTS) as f:
        return json.load(f)


def new_conf(conf, key):
    ss.conf, ss.conf_key = conf, key
    ss.built = ss.built_sig = ss.model = None
    ss.runs, ss.active, ss.replan, ss.compare = {}, None, None, None
    d = conf.defaults or {}
    ss.rv_demand, ss.rv_max, ss.rv_min = int(d.get("demand", 3)), int(d.get("max_load", 6)), 0
    ss.inst_cap, ss.min_senior = 0, 0
    ss.temperature, ss.half_life = DEFAULT_TEMPERATURE, DEFAULT_HALF_LIFE


def apply_preset(name):
    for k, v in PRESETS[name].items():
        ss[k] = v


def current_settings() -> Settings:
    return Settings(demand=int(ss.rv_demand), max_load=int(ss.rv_max), min_load=int(ss.rv_min),
                    temperature=float(ss.temperature), half_life=ss.half_life, use_coauthors=bool(ss.c_coauth),
                    coauthor_years=int(ss.c_years) or None, use_institution=bool(ss.c_inst),
                    use_email_domain=bool(ss.c_email), max_per_institution=int(ss.inst_cap) or None,
                    min_senior=int(ss.min_senior), use_bids=bool(ss.use_bids))


def signature():
    s = asdict(current_settings())
    s.pop("bid_weights", None)
    return (ss.conf_key, tuple(sorted((k, str(v)) for k, v in s.items())))


def get_built():
    if ss.conf is None:
        return None
    sig = signature()
    if ss.built is None or ss.built_sig != sig:
        with st.spinner("Scoring expertise and finding conflicts…"):
            ss.built = build(ss.conf, current_settings(), ss.model)
        ss.model = ss.built.model
        ss.built_sig = sig
    return ss.built


def run_method(b, method, q=0.5, seed=0, time_limit=60.0):
    pb = b.problem
    if method == "randomized":
        return solve.randomized(pb, q=q, seed=seed)
    if method == "greedy":
        return solve.greedy(pb)
    if method == "fair":
        return solve.fair(pb, time_limit=time_limit)
    return solve.max_total(pb, time_limit=time_limit)


def store_run(method, res):
    ss.runs[method] = {"res": res, "sig": ss.built_sig}
    ss.active = method
    ss.replan = None


def active_run(b):
    r = ss.runs.get(ss.active) if ss.active else None
    if r is None:
        return None, False
    return r["res"], r["sig"] == ss.built_sig and r["res"].A.shape == b.problem.shape


def certificate(res, pb) -> str:
    i = res.info
    if res.status == "infeasible":
        return "❌ " + i.get("message", "No assignment satisfies every rule.")
    if res.method == "max_total":
        gap = (res.bound - res.objective) / max(res.bound, 1e-12) if res.bound else 0.0
        if res.status != "optimal":
            return (f"⏱️ **Best found within the time limit**: {res.objective:.4f}, at most {max(gap, 0):.2%} below the "
                    f"best possible ({res.bound:.4f}).")
        return (f"✅ **Proven optimal.** The linear-programming bound is {res.bound:.4f} and this assignment reaches "
                f"{res.objective:.4f} (gap {max(gap, 0):.4%}). Checked against all {fmt(int((~pb.conflict).sum()))} "
                f"allowed pairs, not only the {fmt(i.get('candidates', 0))} the solver kept in memory.")
    if res.method == "fair":
        fb = i.get("floor_bound")
        bound = f" (best possible: {fb:.4f})" if fb is not None and np.isfinite(fb) else ""
        cost = 1 - res.objective / i["total_without_fairness"] if i.get("total_without_fairness") else 0
        return (f"✅ **Worst-off paper raised** from {i['min_without_fairness']:.4f} to {i['floor']:.4f}{bound}, "
                f"costing {cost:.2%} of the total expertise. Solver status: {res.status}.")
    if res.method == "randomized":
        share = i["expected_total"] / i["deterministic_best"] if i.get("deterministic_best") else 0
        return (f"🎲 **Sampled** from a plan where no pair is chosen with probability above {i['q']:.0%}: "
                f"expected total {i['expected_total']:.4f} = {share:.1%} of the best possible. A reviewer who bids "
                f"for a friend's paper gets it at most {i['q']:.0%} of the time. Seed {i['seed']}.")
    short = i.get("papers_short", 0)
    return ("⚠️ Greedy baseline: " + (f"{short} papers were left short of reviewers." if short else
                                       "valid, but not guaranteed optimal."))


def metric_row(pb, A, res=None):
    s = summary(pb, A)
    m = st.columns(6)
    m[0].metric("Total expertise", f"{s['total']:.3f}", help="Sum of expertise over all assigned pairs.")
    m[1].metric("Worst-off paper", f"{s['worst_paper']:.3f}",
                help="Total expertise of the paper whose reviewers know it least.")
    m[2].metric("Top-5 reviewer", f"{s['share_top']:.0%}",
                help="Share of papers where at least one reviewer is among its 5 best eligible reviewers.")
    m[3].metric("Average rank", f"{s['mean_rank']:.1f}",
                help="1 = the paper's best eligible reviewer.")
    m[4].metric("Papers per reviewer", f"{s['load_min']}–{s['load_max']}", help=f"{s['reviewers_used']} reviewers used")
    m[5].metric("Rules broken", len(violations(pb, A)), help="Checked independently of the solver.")
    if res is not None:
        st.caption(f"{METHODS.get(res.method, res.method)} · {res.seconds:.2f} s")


def ecdf_fig(pb, runs: dict, height=320):
    fig = go.Figure()
    for key, res in runs.items():
        ps = np.sort(paper_scores(pb, res.A))
        y = np.arange(1, len(ps) + 1) / len(ps)
        fig.add_trace(go.Scatter(x=ps, y=y, mode="lines", line=dict(width=2, color=METHOD_COLORS[key], shape="hv"),
                                 name=METHODS[key], hovertemplate="%{y:.0%} of papers have ≤ %{x:.3f}<extra></extra>"))
    layout(fig, height, title="Expertise each paper receives (further right is better)",
           xaxis_title="total expertise of the paper's reviewers", yaxis_title="share of papers", yaxis_tickformat=".0%")
    return fig


def seriate(M, A):
    """Order reviewers so similar ones sit together (correspondence analysis), then each paper at the average
    position of its reviewers: the assignment shows as a band along the diagonal."""
    M = np.clip(np.asarray(M, float), 0, None) + 1e-12
    if min(M.shape) < 3:
        return np.arange(M.shape[0]), np.arange(M.shape[1])
    r, c = M.sum(1), M.sum(0)
    U, s, Vt = np.linalg.svd(M / np.sqrt(np.outer(r, c)), full_matrices=False)
    co = np.argsort(Vt[1] / np.sqrt(c / c.sum()))
    pos = np.empty(len(co))
    pos[co] = np.arange(len(co))
    row_ca = U[:, 1] / np.sqrt(r / r.sum())
    fallback = np.argsort(np.argsort(row_ca)) * len(co) / max(len(row_ca), 1)
    key = np.array([pos[A[i]].mean() if A[i].any() else fallback[i] for i in range(M.shape[0])])
    return np.argsort(key, kind="stable"), co


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------

st.title("🧑‍🏫 Reviewer Assignment")
st.caption("Match submissions to the reviewers who know them best, **never with a conflict of interest**, within every "
           "reviewer's **workload**, and **fairly** across papers. For conferences, journals and thesis examiners.")

tabs = st.tabs(["📥 Data", "📏 Rules", "🔎 Expertise", "⚙️ Assign", "📋 Assignment", "📊 Quality", "🧪 Benchmark",
                "ℹ️ Methodology"])

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Demo workshop**")
        st.caption("150 real submissions and 58 real researchers (shown anonymously) with their own papers, from the "
                   "gold-standard expertise dataset. Researchers rated their own expertise, so the result can be checked.")
        if st.button("Load the demo", type="primary", width="stretch"):
            try:
                new_conf(demo_conf(), "demo")
                st.rerun()
            except Exception as exc:   # network problem
                st.error(f"Could not download the demo data: {exc}")
    with c2:
        st.markdown("**Benchmark instance**")
        st.caption("ICLR 2018 text-similarity scores, or real bids from three AI conferences, as used in published "
                   "studies. No text: expertise and conflicts come with the data.")
        bk = st.selectbox("Instance", list(BENCH), format_func=BENCH.get, label_visibility="collapsed")
        if st.button("Load", width="stretch", key="load_bench"):
            try:
                new_conf(bench_conf(bk), bk)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load {bk}: {exc}")
    with c3:
        st.markdown("**Your own data**")
        st.caption("CSV or Excel files: submissions and reviewers (required); reviewers' papers, declared conflicts "
                   "and bids (optional). Templates below.")
        if st.button("Try the example files", width="stretch"):
            files = {k: (open(os.path.join(SAMPLES, f"example_{k}.csv"), "rb").read(), f"example_{k}.csv")
                     for k in ("papers", "reviewers", "profiles", "conflicts", "bids")}
            conf = read_conference(files)
            conf.name = "Example files (fictional)"
            new_conf(conf, "example")
            ss.rv_demand, ss.rv_max = 2, 5
            st.rerun()

    with st.expander("📤 Upload your own files", expanded=ss.conf is None):
        u = st.columns(5)
        ups = {}
        for col, (k, label) in zip(u, [("papers", "Submissions *"), ("reviewers", "Reviewers *"),
                                       ("profiles", "Reviewers' papers"), ("conflicts", "Declared conflicts"),
                                       ("bids", "Bids")]):
            f = col.file_uploader(label, type=["csv", "tsv", "txt", "xlsx", "xls"], key=f"up_{k}")
            ups[k] = (f.getvalue(), f.name) if f is not None else None
        if ups["papers"] and ups["reviewers"]:
            if st.button("Use these files", type="primary"):
                try:
                    conf = read_conference(ups)
                    conf.name = ups["papers"][1].rsplit(".", 1)[0]
                    new_conf(conf, f"upload:{conf.name}:{len(conf.papers)}:{len(conf.reviewers)}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        st.markdown("**Templates** (fictional people and institutions)")
        t = st.columns(5)
        for col, k in zip(t, ("papers", "reviewers", "profiles", "conflicts", "bids")):
            col.download_button(f"example_{k}.csv", open(os.path.join(SAMPLES, f"example_{k}.csv"), "rb").read(),
                                f"example_{k}.csv", "text/csv", width="stretch", key=f"tpl_{k}")
        guide = []
        for kind, cols in COLUMNS.items():
            for std, opts in cols.items():
                guide.append({"file": kind, "column": std, "required": "yes" if std in REQUIRED[kind] else "",
                              "also accepted": ", ".join(o for o in opts if o != std)})
        st.dataframe(pd.DataFrame(guide), hide_index=True, width="stretch", height=280)
        st.caption("Authors: separate with `;`. Bids: eager, willing, neutral, in a pinch, not willing, or conflict "
                   "(yes / maybe / no also work). A reviewer's **expertise** statement counts as one more of their papers. "
                   "Per-row **max_load**, **min_load** and **reviews_needed** override the rules.")

    conf = ss.conf
    if conf is None:
        st.info("Load the demo, a benchmark instance or your own files to begin.")
    else:
        b = get_built()
        pb = b.problem
        st.divider()
        st.subheader(conf.name)
        st.caption(conf.description)
        for w in b.warnings:
            st.caption(f"ℹ️ {w}")
        m = st.columns(6)
        m[0].metric("Submissions", fmt(pb.shape[0]))
        m[1].metric("Reviewers", fmt(pb.shape[1]))
        m[2].metric("Reviewer papers", fmt(sum(len(p) for p in conf.profiles)) if conf.has_text else "—",
                    help="Papers that describe each reviewer's expertise.")
        m[3].metric("Reviews needed", fmt(int(pb.demand.sum())))
        m[4].metric("Capacity", f"{pb.max_load.sum() / max(pb.demand.sum(), 1):.2f}×",
                    help="Reviews the reviewers can take, divided by reviews needed.")
        m[5].metric("Conflicts found", fmt(int(pb.conflict.sum())),
                    help=f"{pb.conflict.mean():.1%} of all pairs are ruled out")
        g1, g2 = st.columns(2)
        with g1:
            if conf.has_text:
                st.markdown("**Submissions**")
                show = conf.papers[["paper_id", "title", "abstract"]].copy()
                show["authors"] = conf.papers["authors"].map(len).astype(str) + " authors"
                st.dataframe(show, hide_index=True, width="stretch", height=300)
            else:
                st.caption("Benchmark instances carry no titles or abstracts: only the affinity scores.")
        with g2:
            st.markdown("**Reviewers**")
            rv = pd.DataFrame({"reviewer": pb.reviewers, "papers in profile": [len(p) for p in conf.profiles],
                               "max load": pb.max_load, "experienced": conf.reviewers["experienced"].astype(bool),
                               "conflicts": pb.conflict.sum(axis=0)})
            if conf.has_text and ss.conf_key not in ("demo",) and conf.reviewers["institution"].astype(str).str.strip().any():
                rv.insert(1, "institution", conf.reviewers["institution"])
            st.dataframe(rv, hide_index=True, width="stretch", height=300)

# ---------------------------------------------------------------------------
# 2. Rules
# ---------------------------------------------------------------------------
with tabs[1]:
    if ss.conf is None:
        st.info("Load data first (📥 Data).")
    else:
        text = ss.conf.has_text
        p1, p2, p3, _ = st.columns([1, 1, 1, 2])
        for col, name in zip((p1, p2, p3), PRESETS):
            col.button(name, on_click=apply_preset, args=(name,), width="stretch", key=f"preset_{name}",
                       help=", ".join(f"{k.split('_', 1)[1]}={v}" for k, v in PRESETS[name].items()))
        st.caption("Presets fill in typical numbers; every rule can then be changed.")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown("**Workload**")
            st.number_input("Reviews per paper", 1, 10, key="rv_demand",
                            help="A 'reviews_needed' column in the submissions file overrides this per paper.")
            st.number_input("Most papers per reviewer", 1, 50, key="rv_max")
            st.number_input("Fewest papers per reviewer", 0, 50, key="rv_min",
                            help="Spread the work: everyone gets at least this many.")
        with r2:
            st.markdown("**Conflicts of interest**")
            st.caption("Authors of a submission can never review it.")
            st.toggle("Co-authors of the submission's authors", key="c_coauth", disabled=not text,
                      help="Found in the author lists of the reviewer's own papers.")
            st.number_input("…only papers from the last N years (0 = any time)", 0, 30, key="c_years",
                            disabled=not (text and ss.c_coauth))
            st.toggle("Same institution", key="c_inst", disabled=not text)
            st.toggle("Same e-mail domain (gmail etc. ignored)", key="c_email", disabled=not text)
        with r3:
            st.markdown("**Balance**")
            st.number_input("Most reviewers per paper from one institution (0 = no limit)", 0, 5, key="inst_cap",
                            disabled=not text)
            st.number_input("Experienced reviewers needed per paper", 0, 3, key="min_senior",
                            help="Reviewers marked 'experienced' in the reviewers file (in the demo: 15+ papers).",
                            disabled=not text)
            st.toggle("Use reviewers' bids", key="use_bids", disabled=not ss.conf.bids,
                      help=", ".join(f"{k} ×{v}" for k, v in BID_WEIGHTS.items()))
        if text:
            st.markdown("**How expertise is measured**")
            e1, e2 = st.columns(2)
            e1.select_slider("A reviewer's expertise for a paper is…", options=list(TEMPS), format_func=TEMPS.get,
                             key="temperature",
                             help="How much a few very close papers count against many loosely related ones.")
            e2.selectbox("Older papers count half as much after", list(HALF_LIVES), format_func=HALF_LIVES.get,
                         key="half_life", help="Reviewers' interests move; recent papers say more about them.")
        b = get_built()
        pb = b.problem
        issues = pb.issues()
        if issues:
            for i in issues:
                st.error(i)
        else:
            st.success(f"✅ Feasible on a first check: {fmt(int(pb.demand.sum()))} reviews needed, room for "
                       f"{fmt(int(pb.max_load.sum()))}; every paper has at least "
                       f"{int((~pb.conflict).sum(axis=1).min())} eligible reviewers.")
        if text and b.reasons:
            kinds = pd.Series([v.split(" (")[0].split(" of ")[0] for v in b.reasons.values()]).value_counts()
            st.markdown(f"**{fmt(len(b.reasons))} conflicts of interest**: " +
                        " · ".join(f"{k} {fmt(int(v))}" for k, v in kinds.items()))
            st.dataframe(conflict_table(ss.conf, b, limit=500), hide_index=True, width="stretch", height=240)

# ---------------------------------------------------------------------------
# 3. Expertise
# ---------------------------------------------------------------------------
with tabs[2]:
    if ss.conf is None:
        st.info("Load data first (📥 Data).")
    else:
        conf = ss.conf
        b = get_built()
        pb = b.problem
        P, R = pb.shape
        gold = conf.gold or {}
        rid = list(conf.reviewers["reviewer_id"])
        st.markdown("#### Who knows this paper best?")
        opts = list(range(P))
        p = st.selectbox("Submission", opts, format_func=lambda i: f"{conf.papers['paper_id'].iloc[i]} · "
                                                                 f"{str(conf.papers['title'].iloc[i])[:100]}")
        order = np.argsort(-b.expertise[p], kind="stable")[:12]
        ranks = candidate_ranks(pb)
        rows = []
        for r in order:
            row = {"reviewer": pb.reviewers[r], "expertise": float(b.expertise[p, r])}
            g = gold.get(rid[r], {}).get(conf.papers["paper_id"].iloc[p])
            if gold:
                row["self-rated"] = f"{g:g} / 5" if g is not None else ""
            row["eligible"] = "❌ " + b.reasons.get((p, r), "conflict") if pb.conflict[p, r] else f"✅ rank {ranks[p, r]}"
            if b.model is not None:
                e = b.model.explain(p, r)
                row["shared terms"] = ", ".join(e["terms"]) or "—"
                row["closest of their papers"] = e["closest"] or ""
            rows.append(row)
        top = float(b.expertise[p, order].max()) or 1.0
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=35 * (len(rows) + 1) + 3,
                     column_config={"expertise": st.column_config.ProgressColumn("expertise", min_value=0.0,
                                                                                 max_value=top, format="%.3f", width="small"),
                                    "shared terms": st.column_config.TextColumn(width="medium")})
        if gold:
            st.caption("**self-rated** = the researcher's own rating of their expertise for this paper in the "
                       "gold-standard study (blank = not rated). It is never used by the app, only shown as a check.")

        if b.model is not None:
            st.markdown("#### Find reviewers for any abstract")
            st.caption("For an editor handling one manuscript, or a department looking for thesis examiners. "
                       "Conflicts are not checked here: no author list is given.")
            txt = st.text_area("Title and abstract", height=110, placeholder="Paste a title and abstract…")
            if txt.strip():
                sc = b.model.score_texts([txt])[0]
                top_r = np.argsort(-sc)[:10]
                rows = []
                for r in top_r:
                    e = b.model.explain_text(txt, r)
                    rows.append({"reviewer": pb.reviewers[r], "expertise": float(sc[r]),
                                 "shared terms": ", ".join(e["terms"]) or "—", "closest of their papers": e["closest"] or ""})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=35 * (len(rows) + 1) + 3,
                             column_config={"expertise": st.column_config.ProgressColumn(
                                 "expertise", min_value=0.0, max_value=max(float(sc.max()), 1e-9), format="%.3f", width="small"),
                                 "shared terms": st.column_config.TextColumn(width="medium")})

        st.markdown("#### Papers that are hard to cover")
        k = int(pb.demand.max())
        S_ok = np.where(pb.conflict, -np.inf, b.expertise)
        kth = -np.sort(-S_ok, axis=1)[:, k - 1] if R >= k else np.zeros(P)
        kth = np.where(np.isfinite(kth), kth, 0.0)
        g1, g2 = st.columns([3, 2])
        with g1:
            fig = go.Figure(go.Histogram(x=kth, nbinsx=30, marker=dict(color=C_OPT, line=dict(width=1, color="rgba(0,0,0,0)")),
                                         hovertemplate="expertise %{x}: %{y} papers<extra></extra>"))
            layout(fig, 300, title=f"Expertise of each paper's {k}{'st' if k == 1 else 'nd' if k == 2 else 'rd' if k == 3 else 'th'} "
                                   f"best eligible reviewer", xaxis_title="expertise", yaxis_title="papers", bargap=0.08)
            st.plotly_chart(fig, width="stretch")
        with g2:
            hard = np.argsort(kth)[:8]
            st.caption("Even a perfect assignment cannot give these papers expert reviewers. Recruit someone, or "
                       "look outside the committee with *Find reviewers for any abstract*.")
            st.dataframe(pd.DataFrame({"paper": [conf.papers["paper_id"].iloc[i] for i in hard],
                                       "title": [str(conf.papers["title"].iloc[i])[:70] for i in hard],
                                       f"best {k}": [round(float(kth[i]), 3) for i in hard]}),
                         hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# 4. Assign
# ---------------------------------------------------------------------------
with tabs[3]:
    if ss.conf is None:
        st.info("Load data first (📥 Data).")
    else:
        b = get_built()
        pb = b.problem
        issues = pb.issues()
        if issues:
            st.error("The rules cannot all be met: " + " ".join(issues) + " Change them in 📏 Rules.")
        a1, a2 = st.columns([2, 1])
        with a1:
            method = st.radio("Goal", ["max_total", "fair", "randomized"], format_func=METHODS.get, horizontal=True)
            st.caption({
                "max_total": "The most expertise in total. Solved exactly with linear programming and a certificate that "
                             "no better assignment exists. Used by most conference systems (TPMS, OpenReview).",
                "fair": "First make the paper that gets the least expertise as well-served as possible, then the most "
                        "total expertise with that guarantee (Stelmakh, Shah & Singh, PeerReview4All). Matters when "
                        "expert reviewers are scarce.",
                "randomized": "Plans with probabilities, no pair above a cap, then samples. A reviewer who bids for a "
                              "friend's paper cannot be sure to get it (Jecmen et al., NeurIPS 2020). Supports loads, "
                              "conflicts and required pairs.",
            }[method])
        with a2:
            q = st.slider("Highest chance for any pair", 0.2, 1.0, 0.5, 0.05, disabled=method != "randomized")
            seed = st.number_input("Random seed", 0, 10_000, 0, disabled=method != "randomized")
            tl = st.slider("Time limit (seconds)", 10, 300, 60, 10)
        b1, b2, _ = st.columns([1, 1, 2])
        go_one = b1.button("Assign reviewers", type="primary", width="stretch", disabled=bool(issues))
        go_all = b2.button("Compare all methods", width="stretch", disabled=bool(issues))
        if go_one:
            with st.spinner(f"{METHODS[method]}…"):
                try:
                    store_run(method, run_method(b, method, q, int(seed), tl))
                except ValueError as exc:
                    st.error(str(exc))
        if go_all:
            done = {}
            prog = st.progress(0.0, "Comparing…")
            keys = ["greedy", "max_total", "fair"] + ([] if (pb.min_senior or (pb.institution is not None and pb.max_per_institution)) else ["randomized"])
            for i, k in enumerate(keys):
                prog.progress(i / len(keys), f"{METHODS[k]}…")
                done[k] = run_method(b, k, q, int(seed), tl)
                ss.runs[k] = {"res": done[k], "sig": ss.built_sig}
            prog.empty()
            ss.compare = {"sig": ss.built_sig, "runs": done}
            ss.active = "max_total"

        res, fresh = active_run(b)
        if res is not None:
            if not fresh:
                st.warning("Data or rules changed since this assignment was made: run it again.")
            else:
                st.divider()
                st.markdown(certificate(res, pb))
                if res.status != "infeasible":
                    metric_row(pb, res.A, res)
        cmp = ss.compare
        if cmp and cmp["sig"] == ss.built_sig:
            st.divider()
            st.markdown("#### All methods on this data")
            rows = []
            for k, r in cmp["runs"].items():
                s = summary(pb, r.A)
                rows.append({"method": METHODS[k], "total expertise": round(s["total"], 4),
                             "worst-off paper": round(s["worst_paper"], 4), "10% of papers below": round(s["paper_p10"], 4),
                             "papers with a top-5 reviewer": f"{s['share_top']:.0%}", "average rank": round(s["mean_rank"], 2),
                             "seconds": round(r.seconds, 2), "valid": "✅" if not violations(pb, r.A) else "❌"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.plotly_chart(ecdf_fig(pb, cmp["runs"]), width="stretch")
            pick = st.radio("Show in 📋 Assignment", list(cmp["runs"]), format_func=METHODS.get, horizontal=True,
                            index=list(cmp["runs"]).index(ss.active) if ss.active in cmp["runs"] else 0)
            if pick != ss.active:
                ss.active = pick
                st.rerun()

# ---------------------------------------------------------------------------
# 5. Assignment
# ---------------------------------------------------------------------------
with tabs[4]:
    b = get_built() if ss.conf is not None else None
    res, fresh = active_run(b) if b is not None else (None, False)
    if res is None or not fresh or res.status == "infeasible":
        st.info("Make an assignment first (⚙️ Assign).")
    else:
        conf = ss.conf
        pb = b.problem
        A = res.A
        view_res = ss.replan["res"] if ss.replan and ss.replan["sig"] == ss.built_sig and ss.replan["base"] is res else None
        st.markdown(f"**{METHODS.get(res.method, res.method)}** · total expertise {res.objective:.3f} · "
                    f"{fmt(int(A.sum()))} reviews")
        v = st.segmented_control("View", ["By paper", "By reviewer", "One paper in detail"], default="By paper",
                                 label_visibility="collapsed")
        if v == "By paper":
            st.dataframe(paper_table(conf, b, A), hide_index=True, width="stretch", height=420,
                         column_config={"total expertise": st.column_config.ProgressColumn(
                             "total expertise", min_value=0.0, max_value=float(paper_scores(pb, A).max() or 1), format="%.3f")})
        elif v == "By reviewer":
            st.dataframe(reviewer_table(conf, b, A), hide_index=True, width="stretch", height=420)
        else:
            p = st.selectbox("Submission", list(range(pb.shape[0])), key="detail_paper",
                             format_func=lambda i: f"{conf.papers['paper_id'].iloc[i]} · {str(conf.papers['title'].iloc[i])[:100]}")
            load = A.sum(axis=0)
            ranks = candidate_ranks(pb)
            order = np.argsort(-np.where(pb.conflict[p], -np.inf, pb.S[p]))[:10]
            rows = []
            for r in order:
                if A[p, r]:
                    why = "✅ assigned"
                elif load[r] >= pb.max_load[r]:
                    why = f"full ({load[r]} of {pb.max_load[r]} papers)"
                else:
                    why = "more useful on other papers"
                row = {"rank": int(ranks[p, r]), "reviewer": pb.reviewers[r], "expertise": round(float(pb.S[p, r]), 4),
                       "status": why}
                if b.model is not None:
                    e = b.model.explain(p, r)
                    row["closest of their papers"] = e["closest"] or ""
                    row["shared terms"] = ", ".join(e["terms"]) or "—"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            if conf.has_text:
                with st.expander("Abstract"):
                    st.write(conf.papers["abstract"].iloc[p] or "—")

        st.markdown("#### Download")
        d1, d2, _ = st.columns([1, 1, 2])
        settings = {"method": METHODS.get(res.method, res.method), "status": res.status,
                    **{k: v for k, v in asdict(current_settings()).items() if k != "bid_weights"}}
        d1.download_button("Workbook (Excel): pairs with reasons, papers, reviewers, conflicts, settings",
                           excel(conf, b, A, settings), "reviewer_assignment.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
        d2.download_button("Pairs (CSV) for a conference system", csv(conf, b, A), "reviewer_assignment.csv",
                           "text/csv", width="stretch")

        st.markdown("#### Someone dropped out? Re-plan with the fewest changes")
        st.caption("Every pair that can stay, stays; only the affected papers get new reviewers.")
        rp1, rp2 = st.columns(2)
        gone = rp1.multiselect("Reviewers who can no longer review", list(range(pb.shape[1])),
                               format_func=lambda r: pb.reviewers[r])
        newc = rp2.multiselect("New conflicts (paper / reviewer)", [f"{conf.papers['paper_id'].iloc[p]} / {pb.reviewers[r]}"
                                                                    for p, r in np.argwhere(A)][:3000])
        if st.button("Re-plan", disabled=not (gone or newc)):
            C = pb.conflict.copy()
            maxl, minl = pb.max_load.copy(), pb.min_load.copy()
            for r in gone:
                C[:, r] = True
                maxl[r] = minl[r] = 0
            lookup = {f"{conf.papers['paper_id'].iloc[p]} / {pb.reviewers[r]}": (p, r) for p, r in np.argwhere(A)}
            for s_ in newc:
                C[lookup[s_]] = True
            pb2 = pb.with_changes(conflict=C, max_load=maxl, min_load=np.minimum(minl, maxl))
            problems = pb2.issues()
            if problems:
                st.error(" ".join(problems))
            else:
                r2 = solve.replan(pb2, A, time_limit=60)
                ss.replan = {"res": r2, "pb": pb2, "sig": ss.built_sig, "base": res}
                view_res = r2
        if view_res is not None:
            pb2 = ss.replan["pb"]
            if view_res.status == "infeasible":
                st.error("No valid re-plan: " + view_res.info.get("message", ""))
            else:
                changed = np.argwhere(A != view_res.A)
                st.success(f"Kept {view_res.info['kept']} of {int(A.sum())} pairs; {view_res.info['moved']} moved. "
                           f"Total expertise {res.objective:.3f} → {view_res.objective:.3f}. "
                           f"Rules broken: {len(violations(pb2, view_res.A))}.")
                rows = []
                for p in sorted(set(int(p) for p, _ in changed)):
                    out_ = [r for r in np.flatnonzero(A[p] & ~view_res.A[p])]
                    in_ = [r for r in np.flatnonzero(view_res.A[p] & ~A[p])]
                    rows.append({"paper": conf.papers["paper_id"].iloc[p], "title": str(conf.papers["title"].iloc[p])[:60],
                                 "leaves": ", ".join(pb.reviewers[r] for r in out_),
                                 "joins": ", ".join(pb.reviewers[r] for r in in_),
                                 "paper's expertise": f"{(pb.S[p] * A[p]).sum():.3f} → {(pb.S[p] * view_res.A[p]).sum():.3f}"})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(35 * (len(rows) + 1) + 3, 320))
                st.download_button("Re-planned pairs (CSV)", csv(conf, b, view_res.A), "reviewer_assignment_replanned.csv",
                                   "text/csv")

# ---------------------------------------------------------------------------
# 6. Quality
# ---------------------------------------------------------------------------
with tabs[5]:
    b = get_built() if ss.conf is not None else None
    res, fresh = active_run(b) if b is not None else (None, False)
    if res is None or not fresh or res.status == "infeasible":
        st.info("Make an assignment first (⚙️ Assign).")
    else:
        conf = ss.conf
        pb = b.problem
        A = res.A
        P, R = pb.shape
        st.markdown(f"Checking **{METHODS.get(res.method, res.method)}**")
        bad = violations(pb, A)
        if bad:
            st.error(f"{len(bad)} rules broken: " + "; ".join(bad[:5]))
        else:
            st.success(f"✅ Independent check: every paper has its reviews, no reviewer is over their limit, and none of "
                       f"the {fmt(int(pb.conflict.sum()))} conflicts of interest is assigned.")
        # assignment map
        rng = np.random.default_rng(0)
        rows_idx = np.arange(P) if P <= 160 else np.sort(rng.choice(P, 120, replace=False))
        cols_idx = np.arange(R) if R <= 160 else np.flatnonzero(A[rows_idx].any(axis=0))
        sub = pb.S[np.ix_(rows_idx, cols_idx)]
        ro, co = seriate(sub, A[np.ix_(rows_idx, cols_idx)])
        sub = sub[np.ix_(ro, co)]
        Asub = A[np.ix_(rows_idx[ro], cols_idx[co])]
        Csub = pb.conflict[np.ix_(rows_idx[ro], cols_idx[co])]
        pl = [str(pb.papers[i]) for i in rows_idx[ro]]
        rl = [str(pb.reviewers[j]) for j in cols_idx[co]]
        fig = go.Figure(go.Heatmap(z=sub, x=list(range(len(rl))), y=list(range(len(pl))),
                                   colorscale=[[0, "rgba(59,130,246,0.0)"], [0.6, "rgba(59,130,246,0.32)"], [1, "#3b82f6"]],
                                   showscale=True,
                                   zmin=0, zmax=float(np.percentile(sub, 99)) or None,
                                   colorbar=dict(title="expertise", thickness=10), customdata=np.array(rl)[None, :].repeat(len(pl), 0),
                                   text=np.array(pl)[:, None].repeat(len(rl), 1),
                                   hovertemplate="%{text} × %{customdata}<br>expertise %{z:.3f}<extra></extra>"))
        yy, xx = np.nonzero(Asub)
        fig.add_trace(go.Scatter(x=xx, y=yy, mode="markers", name="assigned",
                                 marker=dict(size=7, color=C_WARN, line=dict(width=1.5, color="white")),
                                 hovertemplate="assigned<extra></extra>"))
        cy, cx = np.nonzero(Csub)
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", name="conflict of interest", visible="legendonly",
                                 marker=dict(size=7, symbol="x", color=C_BAD), hovertemplate="conflict<extra></extra>"))
        layout(fig, 520, title="Assignment map: expertise (blue) and assigned pairs (orange)"
                               + ("" if P <= 160 else f" · {len(pl)} sampled papers and their reviewers"),
               xaxis_title="reviewers (similar ones side by side)", yaxis_title="submissions (similar ones side by side)")
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, autorange="reversed")
        st.plotly_chart(fig, width="stretch")

        q1, q2, q3 = st.columns(3)
        with q1:
            ps = paper_scores(pb, A)
            fig = go.Figure(go.Histogram(x=ps, nbinsx=30, marker=dict(color=C_OPT, line=dict(width=1, color="rgba(0,0,0,0)")),
                                         hovertemplate="%{x}: %{y} papers<extra></extra>"))
            layout(fig, 300, title="Expertise per paper", xaxis_title="total expertise of its reviewers",
                   yaxis_title="papers", bargap=0.08)
            st.plotly_chart(fig, width="stretch")
        with q2:
            ranks = candidate_ranks(pb)[A]
            cnt = pd.Series(np.minimum(ranks, 11)).value_counts().sort_index()
            xl = [str(i) if i <= 10 else "11+" for i in cnt.index]
            fig = go.Figure(go.Bar(x=xl, y=cnt.values, marker=dict(color=C_OPT, cornerradius=4),
                                   hovertemplate="rank %{x}: %{y} reviews<extra></extra>"))
            layout(fig, 300, title="How high each reviewer ranks for their paper", xaxis_title="rank among eligible reviewers (1 = best)",
                   yaxis_title="reviews", bargap=0.25)
            st.plotly_chart(fig, width="stretch")
        with q3:
            load = A.sum(axis=0)
            cnt = pd.Series(load).value_counts().sort_index()
            fig = go.Figure(go.Bar(x=cnt.index.astype(str), y=cnt.values, marker=dict(color=C_OPT, cornerradius=4),
                                   hovertemplate="%{x} papers: %{y} reviewers<extra></extra>"))
            layout(fig, 300, title="Workload", xaxis_title="papers per reviewer", yaxis_title="reviewers", bargap=0.25)
            st.plotly_chart(fig, width="stretch")

        gold = conf.gold or {}
        if gold:
            st.markdown("#### Check against the researchers' own ratings")
            pidx = {x: i for i, x in enumerate(conf.papers["paper_id"])}
            ridx = {x: j for j, x in enumerate(conf.reviewers["reviewer_id"])}
            got, other = [], []
            for rv_id, g in gold.items():
                for pid, e in g.items():
                    i, j = pidx[pid], ridx[rv_id]
                    if pb.conflict[i, j]:
                        continue
                    (got if A[i, j] else other).append(e)
            if got:
                fig = go.Figure()
                for vals, name, color in ((got, f"assigned pairs ({len(got)})", C_OPT),
                                          (other, f"rated but not assigned ({len(other)})", C_START)):
                    share = pd.Series(vals).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0)
                    fig.add_trace(go.Bar(x=[str(x) for x in share.index], y=share.values, name=name,
                                         marker=dict(color=color, cornerradius=4),
                                         hovertemplate="rating %{x}: %{y:.0%}<extra>" + name + "</extra>"))
                layout(fig, 300, title="Self-rated expertise (1 = no idea, 5 = expert), assigned vs other pairs",
                       xaxis_title="the researcher's own rating", yaxis_title="share of pairs", yaxis_tickformat=".0%",
                       barmode="group", bargap=0.25, bargroupgap=0.08)
                g1, g2 = st.columns([3, 2])
                g1.plotly_chart(fig, width="stretch")
                g2.metric("Average self-rating, assigned pairs", f"{np.mean(got):.2f} / 5",
                          delta=f"{np.mean(got) - np.mean(other):+.2f} vs other rated pairs")
                g2.metric("Assigned pairs rated 4 or 5", f"{np.mean(np.array(got) >= 4):.0%}",
                          delta=f"{np.mean(np.array(got) >= 4) - np.mean(np.array(other) >= 4):+.0%} vs other rated pairs")
                g2.caption("The ratings come from the researchers themselves and are never used to make the assignment, "
                           "so this is an outside check. Only some pairs were rated.")
            else:
                st.caption("None of the assigned pairs was rated by the researcher in the gold-standard study.")

# ---------------------------------------------------------------------------
# 7. Benchmark
# ---------------------------------------------------------------------------
with tabs[6]:
    if not os.path.exists(RESULTS):
        st.info("Run `python benchmarks/run_benchmark.py` to create the benchmark results.")
    else:
        R_ = load_results(os.path.getmtime(RESULTS))
        ex = R_["expertise"]
        st.markdown("#### 1 · Is the expertise score right?")
        st.caption(f"Gold standard of Stelmakh et al. (TMLR 2025): {ex['n_researchers']} researchers rated their own expertise "
                   f"for {ex['n_ratings']} papers they had read. Each method ranks every researcher's papers; the loss counts "
                   "pairs put in the wrong order, weighted by how far apart the ratings are (0 = perfect, 0.5 = no better "
                   "than chance). Other methods are scored from the predictions their authors released, with the same code.")
        rows = [(v["label"], v["loss"], v["ci"], v["kind"]) for v in ex["published"].values()]
        rows.append(("This app: TF-IDF + recency-weighted soft maximum", ex["default_loss"], ex["default_ci"], "ours"))
        rows.sort(key=lambda r: -r[1])
        col = {"ours": C_OPT, "classic": C_START, "neural": C_START, "llm": C_START}
        fig = go.Figure(go.Bar(
            y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h",
            marker=dict(color=[col[r[3]] for r in rows], cornerradius=4),
            error_x=dict(type="data", symmetric=False, array=[r[2][1] - r[1] for r in rows],
                         arrayminus=[r[1] - r[2][0] for r in rows], color="rgba(128,128,128,.6)", thickness=1.2),
            text=[f"{r[1]:.3f}" for r in rows], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: loss %{x:.3f}<extra></extra>"))
        layout(fig, 430, title="Loss on the gold standard (lower is better; bars: 95% bootstrap interval)",
               xaxis_title="weighted Kendall-tau loss", xaxis_range=[0.15, 0.42])
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Settings chosen by 5-fold cross-validation over researchers give a held-out loss of "
                   f"**{ex['cv_loss']:.3f}** (sd {ex['cv_loss_sd']:.3f}), so the score is not tuned on the people it is tested on. "
                   "It needs no GPU, no training and no paid API, and every match comes with its reason.")

        st.markdown("#### 2 · Is the assignment right?")
        asg = R_["assignment"]
        rows = []
        for key, inst in asg.items():
            m_ = inst["methods"]
            g_, t_, f_, r_ = m_["greedy"], m_["max_total"], m_["fair"], m_["randomized"]
            cost = 1 - f_["total"] / t_["total"]
            short = {"iclr2018": "ICLR 2018", "preflib1": "AI conference 1 bids", "preflib2": "AI conference 2 bids",
                     "preflib3": "AI conference 3 bids", "iclr2018_scarce": "ICLR 2018, scarce reviewers",
                     "preflib3_scarce": "AI conference 3, scarce reviewers", "demo": "Demo workshop"}
            rows.append({"instance": short.get(key, inst["name"]),
                         "size": f"{inst['papers']:,} × {inst['reviewers']:,}", "capacity": f"{inst['capacity']:.2f}×",
                         "optimal total": f"{t_['total']:.2f} in {t_['seconds']:.1f} s",
                         "greedy": f"{g_['total'] / t_['total'] - 1:+.1%}",
                         "worst-off paper: optimal → fair": f"{t_['worst_paper']:.3f} → {f_['worst_paper']:.3f}",
                         "cost of fairness": "none" if cost < 5e-5 else f"{cost:.2%}",
                         "randomised (q = 0.5)": f"{r_['total'] / t_['total']:.0%}",
                         "all valid": "✅" if all(x["valid"] and not x.get("papers_short") for x in m_.values()) else "❌"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("**optimal total**: proven optimal. **greedy**: the usual best-pair-first heuristic, relative to it. "
                   "**worst-off paper**: expertise of the least-served paper, without and with the fairness goal. "
                   "**randomised**: expected share of the optimal total when no pair may have a chance above 50%.")
        ic = asg.get("iclr2018", {})
        if ic.get("dense_lp_seconds"):
            mt = ic["methods"]["max_total"]
            st.caption(f"ICLR 2018: the optimal assignment takes {mt['seconds']:.1f} s with column generation "
                       f"({fmt(mt['candidates'])} of {fmt(911 * 2435)} pairs kept in memory) against "
                       f"{ic['dense_lp_seconds']:.1f} s for the same linear program over every pair; both reach "
                       f"{ic['dense_lp_total']:.3f}.")
        g1, g2 = st.columns(2)
        with g1:
            sc = [k for k in ("iclr2018_scarce", "preflib3_scarce") if k in asg]
            fig = go.Figure()
            for m_ in ("max_total", "fair"):
                fig.add_trace(go.Bar(x=[asg[k]["name"].split(",")[0].replace(" bids", "") + " (scarce)" for k in sc],
                                     y=[asg[k]["methods"][m_]["worst_paper"] / asg[k]["methods"]["max_total"]["worst_paper"]
                                        for k in sc],
                                     name=METHODS[m_], marker=dict(color=METHOD_COLORS[m_], cornerradius=4),
                                     hovertemplate="%{y:.2f}× the worst-off paper of the highest-total assignment<extra></extra>"))
            layout(fig, 320, title="Worst-off paper when reviewers are scarce (highest total = 1)",
                   yaxis_title="relative expertise", barmode="group", bargap=0.3)
            st.plotly_chart(fig, width="stretch")
            for k in sc:
                mt, fr = asg[k]["methods"]["max_total"], asg[k]["methods"]["fair"]
                cost = 1 - fr["total"] / mt["total"]
                st.caption(f"{asg[k]['name']}: worst-off paper {mt['worst_paper']:.3f} → {fr['worst_paper']:.3f} "
                           + ("at no cost in total expertise." if cost < 5e-5 else f"for {cost:.2%} less total expertise."))
        with g2:
            rc = R_.get("randomized_curve")
            if rc:
                fig = go.Figure(go.Scatter(x=[p["q"] for p in rc["points"]], y=[p["share_of_best"] for p in rc["points"]],
                                           mode="lines+markers", line=dict(color=C_WARN, width=2), marker=dict(size=8),
                                           hovertemplate="cap %{x:.0%}: %{y:.1%} of the best total<extra></extra>"))
                layout(fig, 320, title="Randomised: price of protection (ICLR 2018)", xaxis_title="highest chance for any pair",
                       yaxis_title="expected share of the best total", xaxis_tickformat=".0%", yaxis_tickformat=".0%")
                st.plotly_chart(fig, width="stretch")
            mg = R_.get("marginals")
            if mg:
                st.caption(f"Sampling check on AI conference 1: {fmt(mg['samples'])} samples, {mg['invalid']} invalid; "
                           f"every pair was drawn within {mg['max_abs_deviation']:.3f} of its planned probability.")
        dm = asg.get("demo", {}).get("gold")
        if dm:
            st.caption(f"Demo workshop, outside check: assigned pairs have an average self-rated expertise of "
                       f"{dm['assigned_mean']:.2f}/5 ({dm['assigned_4plus']:.0%} rated 4–5) against {dm['other_mean']:.2f}/5 "
                       f"({dm['other_4plus']:.0%}) for other rated pairs.")
        st.caption(f"Benchmark run: {R_.get('machine', '')}, {R_.get('seconds', 0) / 60:.0f} minutes. "
                   "Re-run with `python benchmarks/run_benchmark.py`.")

        st.markdown("#### Re-check the expertise score here")
        st.caption("Scores the gold standard (dataset 1 of 10) with the settings chosen in 📏 Rules. About 15 seconds.")
        if st.button("Run the check"):
            from benchmarks.run_benchmark import expertise_predictions, loss_parts, pair_terms
            with st.spinner("Scoring 58 researchers' papers…"):
                subs, arch, refs = data.gold_dataset(1)
                preds = expertise_predictions(subs, arch, refs, float(ss.temperature), ss.half_life)
                n_, d_ = loss_parts(preds, pair_terms(refs), list(refs))
                ss.bench_live = (TEMPS.get(float(ss.temperature)), HALF_LIVES.get(ss.half_life), n_ / d_)
        if ss.bench_live:
            t_, h_, l_ = ss.bench_live
            st.success(f"Loss {l_:.4f} on dataset 1 ({t_}; half-life {h_}).")

# ---------------------------------------------------------------------------
# 8. Methodology
# ---------------------------------------------------------------------------
with tabs[7]:
    st.markdown("""
#### The problem
Every conference, journal and doctoral school has to decide **who reviews what**. A chair with 1,000 submissions and
2,000 reviewers cannot do it by hand, and poor matches mean shallow reviews, wrong decisions and unhappy authors.
The rules are strict: authors, co-authors and colleagues must not review each other; nobody can take more than a few
papers; every paper needs its reviews.

#### 1 · Expertise
Each reviewer is described by their own papers (title + abstract) and, optionally, a short statement. Texts become
TF-IDF vectors over words and word pairs. A reviewer's expertise for a submission combines the similarity to each of
their papers with a **soft maximum**, so a few very close papers count more than many loose ones, and **recent
papers count more** (half-life 5 years). Both settings were chosen by cross-validation on the gold-standard dataset.
Every match shows **the reviewer's closest paper and the terms they share**, so a chair can see why.

#### 2 · Conflicts of interest
Authors (name or e-mail), co-authors (from the reviewer's own author lists, optionally only recent ones), the same
institution or e-mail domain (public providers ignored), and conflicts declared by the chair or by the reviewer's bid.
Names are compared without accents; an initial matches a full first name, which errs on the safe side.

#### 3 · Assignment
An integer program: choose reviewer–paper pairs, each paper gets its reviews, each reviewer stays within their
limits, no conflict is used, optionally no two reviewers of a paper from one institution and at least one experienced
reviewer per paper.
* **Highest total expertise.** Solved as a linear program (HiGHS). The constraint matrix is totally unimodular, so
  the optimum is whole pairs; extra rules fall back to a MILP. Only each paper's best candidates are kept in memory,
  and dual prices prove that no left-out pair could improve the result (**column generation**), so the answer is
  certified optimal over all pairs.
* **Fair.** Maximise the expertise of the worst-off paper first, then the total (PeerReview4All, Stelmakh et al. 2021).
* **Randomised.** Maximise expected expertise with no pair above probability *q*, then sample with dependent
  rounding, which keeps every load and every paper's count exact (Jecmen et al. 2020).
* **Re-planning.** Pairs that may stay earn a bonus larger than any gain from moving them, so the new optimum changes
  as few pairs as possible.

#### Limits
* Text similarity misses expertise that is not in the reviewer's papers, and needs the papers: a reviewer with no
  papers and no statement scores 0 for everything.
* Names are not identities: two people with the same name and initial are treated as one (a safe extra conflict).
* Bids can be gamed; the randomised method limits how much.

#### Data and credits
Gold-standard expertise dataset: Stelmakh, Wieting, Xi, Neubig & Shah (TMLR 2025), evaluations CC BY 4.0; paper
metadata from Semantic Scholar, downloaded when used, not redistributed. ICLR 2018 similarities: Xu et al. (2019),
via Jecmen et al. (2020). Real bids: PrefLib 00039 (Mattei & Walsh). The app's own code is MIT-licensed.
""")

# ---------------------------------------------------------------------------
# sidebar (last, so it reflects this run)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧑‍🏫 Reviewer Assignment")
    st.caption("Expert, conflict-free, fair reviewer assignment.")
    if ss.conf is None:
        st.info("Start in **📥 Data**: load the demo, a benchmark instance or your own files.")
    else:
        b = ss.built
        st.markdown(f"**Data:** {html.escape(ss.conf.name)}  \n{fmt(b.problem.shape[0])} submissions · "
                    f"{fmt(b.problem.shape[1])} reviewers")
        res, fresh = active_run(b)
        if res is None:
            st.markdown("**Assignment:** not made yet")
        else:
            ok = not violations(b.problem, res.A) if fresh else False
            st.markdown(f"**Assignment:** {METHODS.get(res.method)}  \n"
                        + (f"{'✅ valid' if ok else '❌ has problems'} · total {res.objective:.3f}" if fresh else
                           "⚠️ rules changed: run again"))
    st.divider()
    st.caption("Expertise: TF-IDF with a recency-weighted soft maximum, tested on a gold standard. Assignment: "
               "linear and integer programming (HiGHS) with an optimality certificate. See **ℹ️ Methodology**.")
