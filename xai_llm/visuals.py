"""Figures. Design rules: one message per figure, direction + strength + reliability
all visible without a legend lookup, colour never the only cue (arrows, labels,
dashed outlines), and every figure says in words how to read it."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import NullFormatter
import matplotlib.colors as mcolors

from xai_llm.explain import Explanation
from xai_llm.narrative import mult, prob_name, tok

SUPPORT = "#c0392b"   # red: pushes toward the explained token
OPPOSE = "#2b6cb0"    # blue: pushes away from it
INK = "#1d1d1f"
MUTED = "#6e6e73"
NS_FACE, NS_EDGE = "#f2f2f2", "#9a9a9a"
GOOD, WARN = "#2e8b57", "#d17b0f"


def _mix(color: str, t: float):
    """White -> color, t in [0,1]."""
    r, g, b = mcolors.to_rgb(color)
    return (1 - (1 - r) * t, 1 - (1 - g) * t, 1 - (1 - b) * t)


def _label(e: Explanation, i: int) -> str:
    return f"{e.units[i]}  (#{i + 1})"


def _simulated_note(e: Explanation) -> str:
    return "SIMULATED MODEL: demonstrates the method, says nothing about a real LLM" if e.backend_name == "simulated" else ""


# --------------------------------------------------------------------------- hero
def hero_figure(e: Explanation, width: int = 1100, dpi: int = 100) -> Figure:
    """The prompt as coloured word chips. Red ▲ = supports the prediction,
    blue ▼ = works against it, dashed grey 'n.s.' = no reliable effect."""
    fig = Figure(figsize=(width / dpi, 6), dpi=dpi)
    FigureCanvasAgg(fig)
    rend = fig.canvas.get_renderer()

    def meas(s, size, weight="normal"):
        t = fig.text(0, 0, s, fontsize=size, fontweight=weight)
        bb = t.get_window_extent(rend)
        t.remove()
        return bb.width, bb.height

    n = len(e.units)
    W, pad = width, 28
    beta, sig = e.beta, e.sig
    scale = max(float(np.max(np.abs(beta[sig]))) if sig.any() else 1.0, 1e-6)

    if e.contrast:
        title = f"Why {tok(e.target.token)} rather than {tok(e.foil.token)}?"
        sub = f"Chance of {tok(e.target.token)} over {tok(e.foil.token)}: {e.base_obs.prob:.0%}"
    else:
        title = f"Why does the model say {tok(e.target.token)}?"
        sub = f"Probability of {tok(e.target.token)} as the next token: {e.base_obs.prob:.0%}"
    sub += f"   ·   {int(sig.sum())} of {n} words have a reliable effect"
    sim = _simulated_note(e)

    word_size, lab_size = 17, 10.5
    labels = [("▲ " if beta[i] > 0 else "▼ ") + mult(beta[i]) if sig[i] else "n.s." for i in range(n)]
    wh = meas("Ag", word_size)[1]
    chip_h = wh + 18
    lab_h = meas("Ag", lab_size)[1]
    row_h = chip_h + 4 + lab_h + 24

    y = pad
    y_sim = None
    if sim:
        y_sim = y
        y += meas(sim, 11, "bold")[1] + 8
    y_title = y
    y += meas(title, 21, "bold")[1] + 8
    y_sub = y
    y += meas(sub, 12)[1] + 28

    place, x, row_top = [], pad, y
    for i in range(n):
        w_word = meas(e.units[i], word_size)[0] + 22
        w_lab = meas(labels[i], lab_size, "bold")[0] + 8
        cw = max(w_word, w_lab)
        if x + cw > W - pad and x > pad:
            x = pad
            row_top += row_h
        place.append((x, row_top, cw))
        x += cw + 9
    y = row_top + chip_h + 4 + lab_h + 30

    # legend block
    leg_top = y
    y += 6
    legend_items = [("supports the prediction", "sup"), ("works against it", "opp"), ("no reliable effect", "ns")]
    y_leg1 = y
    y += chip_h * 0.7 + 18
    pn = prob_name(e)
    expl = f"×N  =  keeping the word multiplies {('the odds' if e.contrast else 'the probability')} by N  (average over many random word-removal experiments; ×1 = no effect)"
    y_expl = y
    y += meas(expl, 11)[1] + 14
    y_bar = y
    y += 16 + meas("×1", 10)[1] + 22
    footer = (
        f"{e.n_experiments} model calls · coverage {e.coverage['overall']:.0%} · "
        f"{e.cfg.ci:.0%} bootstrap intervals · false-discovery rate {e.cfg.fdr:.0%}"
    )
    y_footer = y
    y += meas(footer, 10)[1] + pad
    H = int(math.ceil(y))

    fig.set_size_inches(W / dpi, H / dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    if sim:
        ax.text(pad, y_sim, sim, fontsize=11, fontweight="bold", color=SUPPORT, va="top")
    ax.text(pad, y_title, title, fontsize=21, fontweight="bold", color=INK, va="top")
    ax.text(pad, y_sub, sub, fontsize=12, color=MUTED, va="top")

    def chip(x0, y0, w, h, kind, text, strength=1.0):
        if kind == "ns":
            face, edge, ls, tc = NS_FACE, NS_EDGE, (0, (4, 3)), "#777777"
        else:
            base = SUPPORT if kind == "sup" else OPPOSE
            t = 0.22 + 0.78 * min(1.0, strength)
            face, edge, ls = _mix(base, t), base, "-"
            tc = "white" if t > 0.62 else INK
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=9",
                                    facecolor=face, edgecolor=edge, linewidth=1.6, linestyle=ls))
        ax.text(x0 + w / 2, y0 + h / 2, text, fontsize=word_size, color=tc, ha="center", va="center")

    for i, (cx, cy, cw) in enumerate(place):
        if sig[i]:
            kind = "sup" if beta[i] > 0 else "opp"
            chip(cx, cy, cw, chip_h, kind, e.units[i], abs(beta[i]) / scale)
            ax.text(cx + cw / 2, cy + chip_h + 4, labels[i], fontsize=lab_size, fontweight="bold",
                    color=SUPPORT if kind == "sup" else OPPOSE, ha="center", va="top")
        else:
            chip(cx, cy, cw, chip_h, "ns", e.units[i])
            ax.text(cx + cw / 2, cy + chip_h + 4, "n.s.", fontsize=lab_size, color="#8a8a8a", ha="center", va="top")

    ax.plot([pad, W - pad], [leg_top, leg_top], color="#dddddd", lw=1)
    lx = pad
    for text, kind in legend_items:
        ch = chip_h * 0.7
        ax.add_patch(FancyBboxPatch((lx, y_leg1), 34, ch, boxstyle="round,pad=0,rounding_size=6",
                                    facecolor=(NS_FACE if kind == "ns" else _mix(SUPPORT if kind == "sup" else OPPOSE, 0.75)),
                                    edgecolor=(NS_EDGE if kind == "ns" else (SUPPORT if kind == "sup" else OPPOSE)),
                                    linewidth=1.4, linestyle=((0, (4, 3)) if kind == "ns" else "-")))
        glyph = {"sup": "▲", "opp": "▼", "ns": ""}[kind]
        if glyph:
            ax.text(lx + 17, y_leg1 + ch / 2, glyph, fontsize=10, color="white", ha="center", va="center")
        ax.text(lx + 44, y_leg1 + ch / 2, text, fontsize=11, color=INK, va="center")
        lx += 44 + meas(text, 11)[0] + 34
    ax.text(pad, y_expl, expl, fontsize=11, color=INK, va="top")

    # colour-strength bar
    bar_w, steps = 300, 60
    ax.text(pad, y_bar + 4, "colour strength:", fontsize=10, color=MUTED, va="top")
    bx = pad + meas("colour strength:", 10)[0] + 12
    for s in range(steps):
        v = (s / (steps - 1)) * 2 - 1  # -1 .. 1
        col = _mix(SUPPORT if v >= 0 else OPPOSE, 0.22 + 0.78 * abs(v)) if abs(v) > 0.02 else (1, 1, 1)
        ax.add_patch(Rectangle((bx + s * bar_w / steps, y_bar), bar_w / steps + 0.6, 16, facecolor=col, edgecolor="none"))
    ax.add_patch(Rectangle((bx, y_bar), bar_w, 16, facecolor="none", edgecolor="#bbbbbb", linewidth=0.8))
    ax.text(bx, y_bar + 20, mult(-scale), fontsize=10, color=OPPOSE, ha="left", va="top")
    ax.text(bx + bar_w / 2, y_bar + 20, "×1", fontsize=10, color=MUTED, ha="center", va="top")
    ax.text(bx + bar_w, y_bar + 20, mult(scale), fontsize=10, color=SUPPORT, ha="right", va="top")
    ax.text(pad, y_footer, footer, fontsize=10, color=MUTED, va="top")
    return fig


# --------------------------------------------------------------------------- forest
def forest_figure(e: Explanation) -> Figure:
    """Effect of keeping each word, with its confidence interval, on a log scale."""
    n = len(e.units)
    order = np.argsort(-e.beta, kind="stable")
    fig = Figure(figsize=(10.5, 0.46 * n + 2.3), constrained_layout=True)
    ax = fig.subplots()

    m = np.exp(np.clip(e.beta, -12, 12))
    lo = np.exp(np.clip(e.ci_lo, -12, 12))
    hi = np.exp(np.clip(e.ci_hi, -12, 12))
    xmin = min(float(lo.min()), 0.5) * 0.8
    xmax = max(float(hi.max()), 2.0) * 1.25
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.axvspan(xmin, 1, color=OPPOSE, alpha=0.05)
    ax.axvspan(1, xmax, color=SUPPORT, alpha=0.05)
    ax.axvline(1, color=INK, lw=1)

    for rank, i in enumerate(order):
        y = n - 1 - rank
        s = bool(e.sig[i])
        col = (SUPPORT if e.beta[i] > 0 else OPPOSE) if s else "#8a8a8a"
        ax.plot([lo[i], hi[i]], [y, y], color=col, lw=2.4 if s else 1.4, solid_capstyle="round")
        ax.plot([m[i]], [y], "o", ms=8 if s else 6, mfc=col if s else "white", mec=col, mew=1.7, zorder=3)
        qtxt = "q<0.001" if e.q[i] < 0.001 else f"q={e.q[i]:.3f}"
        note = f"{mult(e.beta[i])}   {qtxt}" if s else f"{mult(e.beta[i])}   n.s."
        ax.text(1.012, y, note, transform=ax.get_yaxis_transform(), va="center", fontsize=9,
                color=col if s else "#777777", fontweight="bold" if s else "normal")

    ax.set_yticks([n - 1 - r for r in range(n)])
    ax.set_yticklabels([_label(e, i) for i in order], fontsize=10)
    cands = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    ticks = [t for t in cands if xmin <= t <= xmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"×{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-0.7, n - 0.3)
    ax.text(0.01, 1.01, "◀ works against the prediction", transform=ax.transAxes, color=OPPOSE, fontsize=10, ha="left", fontweight="bold")
    ax.text(0.99 - 0.17, 1.01, "supports the prediction ▶", transform=ax.transAxes, color=SUPPORT, fontsize=10, ha="right", fontweight="bold")
    what = "odds" if e.contrast else "probability"
    ax.set_xlabel(f"How much keeping the word multiplies the {what} (log scale, ×1 = no effect). "
                  f"Bars = {e.cfg.ci:.0%} bootstrap interval; hollow grey = no reliable effect (n.s.)", fontsize=9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Effect of each word, with uncertainty", loc="left", fontsize=14, fontweight="bold", pad=24)
    return fig


# --------------------------------------------------------------------------- counterfactuals
def counterfactual_figure(e: Explanation) -> Figure:
    """What the model does when ONE word is deleted (exact, no statistics needed)."""
    n = len(e.units)
    base = e.base_obs.prob
    fig = Figure(figsize=(10.5, 0.46 * n + 2.3), constrained_layout=True)
    ax = fig.subplots()
    for i, o in enumerate(e.loo_obs):
        y = n - 1 - i
        d = base - o.prob
        col = "#9aa0a6" if abs(d) < 0.02 else (SUPPORT if d > 0 else OPPOSE)
        ax.barh(y, o.prob, color=col, alpha=0.88, height=0.62,
                hatch=None if o.covered else "///", edgecolor="white" if o.covered else col, linewidth=0.8)
        txt = f"{o.prob:.0%}" + ("" if o.covered else " (at most)")
        if o.top1_key != e.target.key:
            txt += f"   → model's top choice becomes {tok(o.top1_token)} ({o.top1_prob:.0%})"
        ax.text(o.prob + 0.012, y, txt, va="center", fontsize=9)
    ax.axvline(base, color=INK, ls="--", lw=1.2)
    ax.text(base, n - 0.35, f"full prompt: {base:.0%}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels([_label(e, i) for i in range(n)], fontsize=10)
    changed = any(o.top1_key != e.target.key for o in e.loo_obs)
    ax.set_xlim(0, 1.85 if changed else 1.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylim(-0.7, n + 0.2)
    ax.set_xlabel(f"{'Chance of target over foil' if e.contrast else 'Probability of ' + tok(e.target.token)} "
                  "after deleting that one word.  Red = the word was helping · blue = it was hurting · grey = no change · hatched = lower bound",
                  fontsize=9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("What happens if you delete one word?", loc="left", fontsize=14, fontweight="bold", pad=16)
    return fig


# --------------------------------------------------------------------------- faithfulness
def faithfulness_figure(e: Explanation) -> Figure:
    d = e.deletion
    base = e.base_obs.prob
    ks = [0] + d["ks"]
    fig = Figure(figsize=(10.5, 5.0), constrained_layout=True)
    ax = fig.subplots()
    lo, med, hi = [base] + d["rand_lo"], [base] + d["rand_med"], [base] + d["rand_hi"]
    ax.fill_between(ks, lo, hi, color="#9a9a9a", alpha=0.25,
                    label=f"random words from outside the top {d['ks'][-1]}: 5th–95th percentile ({d['n_random']} draws)")
    ax.plot(ks, med, color="#5f5f5f", ls="--", lw=1.8, label="random words: median")
    top = [base] + d["top_prob"]
    low = [base] + d["low_prob"]
    ax.plot(ks, top, color=SUPPORT, lw=2.4, marker="o", ms=7, label="remove the words ranked MOST important")
    ax.plot(ks, low, color=GOOD, lw=2.4, marker="s", ms=6, label="remove the words ranked LEAST important")
    for series, cov, col in ((top[1:], d["top_covered"], SUPPORT), (low[1:], d["low_covered"], GOOD)):
        for k, v, c in zip(ks[1:], series, cov):
            if not c:
                ax.plot([k], [v], "o", ms=7, mfc="white", mec=col, mew=1.6, zorder=4)
    ax.set_xticks(ks)
    ax.set_xticklabels(
        ["0\n(full prompt)"] + [f"{k}\np = {p:.2f}" if p >= 0.01 else f"{k}\np < 0.01" for k, p in zip(ks[1:], d["p_values"])],
        fontsize=9,
    )
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("number of words removed from the prompt   (p = chance that random outside words hurt at least as much)")
    ax.set_ylabel(f"{'chance of target over foil' if e.contrast else 'P(' + tok(e.target.token) + ')'}")
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if not all(d["top_covered"]) or not all(d["low_covered"]):
        handles.append(Line2D([], [], marker="o", mfc="white", mec="#555555", ls="", mew=1.5))
        labels.append("hollow marker = target left the model's top predictions (value is an upper bound)")
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9, frameon=False)
    verdict = "PASS" if d["p_auc"] < 0.05 else "NOT CLEARLY BETTER THAN CHANCE"
    ax.set_title(
        f"Does the explanation predict reality?  {verdict}  (whole-curve permutation p = {d['p_auc']:.3f})\n"
        "If it is right, the RED curve falls faster than the grey band and the GREEN curve stays high.",
        loc="left", fontsize=12.5, fontweight="bold", color=GOOD if d["p_auc"] < 0.05 else WARN,
    )
    return fig


# --------------------------------------------------------------------------- trust dashboard
def trust_figure(e: Explanation) -> Figure:
    fig = Figure(figsize=(14, 4.6), constrained_layout=True)
    a, b, c = fig.subplots(1, 3)
    thr = e.cfg.min_coverage

    names = list(e.coverage)
    vals = [e.coverage[k] for k in names]
    cols = [GOOD if v >= thr else WARN for v in vals]
    a.bar(range(len(vals)), vals, color=cols)
    for i, v in enumerate(vals):
        a.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, fontweight="bold")
    a.axhline(thr, ls="--", color=INK, lw=1)
    a.set_ylim(0, 1.12)
    a.set_xticks(range(len(vals)))
    a.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=9)
    ok = e.coverage["overall"] >= thr
    a.set_title(("✔ " if ok else "⚠ ") + f"Coverage (goal ≥ {thr:.0%})\nshare of experiments where the target's\nprobability was actually observed",
                fontsize=10.5, color=GOOD if ok else WARN, loc="left")

    r2 = e.r2_cv
    ok = (not math.isnan(r2)) and r2 >= 0.5
    b.scatter(e.yhat_cv, e.y_cv, s=14, alpha=0.35, color=OPPOSE, edgecolors="none")
    lo_, hi_ = float(min(e.y_cv.min(), e.yhat_cv.min())), float(max(e.y_cv.max(), e.yhat_cv.max()))
    b.plot([lo_, hi_], [lo_, hi_], color=INK, lw=1)
    b.set_xlabel("predicted from word effects (held-out)", fontsize=9)
    b.set_ylabel("actual model score", fontsize=9)
    b.set_title(("✔ " if ok else "⚠ ") + f"Does the word-by-word story fit? (goal R² ≥ 0.5)\nheld-out R² = " + ("n/a" if math.isnan(r2) else f"{r2:.2f}"),
                fontsize=10.5, color=GOOD if ok else WARN, loc="left")

    r = e.r_methods
    ok = (not math.isnan(r)) and r >= 0.5
    c.scatter(e.loo_effect, e.beta, s=40, color=SUPPORT, alpha=0.85, edgecolors="white")
    for i, u in enumerate(e.units):
        if e.sig[i]:  # label only the words that matter; the rest cluster at (0, 0)
            c.annotate(u, (e.loo_effect[i], e.beta[i]), fontsize=9, xytext=(4, 4), textcoords="offset points")
    lo_, hi_ = float(min(e.loo_effect.min(), e.beta.min())), float(max(e.loo_effect.max(), e.beta.max()))
    c.plot([lo_, hi_], [lo_, hi_], color="#999999", lw=1, ls="--")
    c.set_xlabel("remove-one effect (nats)", fontsize=9)
    c.set_ylabel("random-removal effect (nats)", fontsize=9)
    c.set_title(("✔ " if ok else "⚠ ") + f"Do the two methods agree? (goal r ≥ 0.5)\ncorrelation r = " + ("n/a" if math.isnan(r) else f"{r:.2f}"),
                fontsize=10.5, color=GOOD if ok else WARN, loc="left")
    for ax in (a, b, c):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Can you trust this explanation?", x=0.01, ha="left", fontsize=14, fontweight="bold")
    return fig


def all_figures(e: Explanation) -> Dict[str, Figure]:
    return {
        "hero": hero_figure(e),
        "effects": forest_figure(e),
        "counterfactual": counterfactual_figure(e),
        "faithfulness": faithfulness_figure(e),
        "trust": trust_figure(e),
    }
