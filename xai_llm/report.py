"""Turn an Explanation into files: PNG figures, a self-contained HTML report, JSON, CSV."""
from __future__ import annotations

import base64
import csv
import html
import io
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from xai_llm import visuals
from xai_llm.explain import Explanation
from xai_llm.narrative import mult, summary, tok

CAPTIONS: Dict[str, str] = {
    "hero": (
        "Every word of your prompt, coloured by its effect on the model's output. "
        "Red ▲ = the word makes the output more likely, blue ▼ = less likely, dashed grey n.s. = no statistically "
        "reliable effect. ×N is the multiplier on the probability; colour strength follows the size of the effect."
    ),
    "effects": (
        "Dots are the estimated effect of KEEPING each word (versus deleting it), averaged over many random "
        "experiments; bars are the 95% bootstrap interval. A word counts as reliable only if its interval excludes ×1, "
        "its false-discovery-adjusted p-value (q) is small enough, and the effect is not negligible."
    ),
    "counterfactual": (
        "The most direct evidence: delete ONE word, ask the model again, read the new probability. No statistics "
        "involved. It can miss words that overlap (delete one and the other still carries the meaning), which is why "
        "the random-removal estimate above exists."
    ),
    "faithfulness": (
        "A test of the explanation itself, run on fresh experiments. Words are removed in the order the explanation ranks "
        "them. If it is faithful, removing its top-ranked words (red) must hurt much more than removing random words the "
        "explanation did NOT rank highly (grey band), and removing its lowest-ranked words (green) must hurt almost nothing. "
        "The p-value is a permutation test of that comparison over the whole curve."
    ),
    "trust": (
        "Three independent checks. Coverage: the server only reports its top predictions, so we track how often the "
        "target's probability was really observed. Fit: how well the word-by-word story predicts experiments it was not "
        "fitted on. Agreement: whether delete-one and the random-removal estimate tell the same story."
    ),
}

METHOD_NOTES: List[str] = [
    "Ollama exposes probabilities but no gradients, so the model is treated as a black box and studied with controlled experiments.",
    "Each experiment deletes some words from the prompt and asks the model (temperature 0) for the probability of the explained first output token.",
    "Delete-one: every word is deleted on its own. Exact, easy to read, but blind to words that overlap.",
    "Random removals: many random subsets of words are deleted. The score is regressed on which words were kept; each coefficient is that word's average effect (a Shapley-style estimate). "
    "Percentile-bootstrap intervals and p-values quantify uncertainty; Benjamini–Hochberg controls the false discovery rate.",
    "The share of words removed per experiment adapts so the target stays inside the server's top-N list (coverage). When it does not, only an upper bound on its probability is known and the experiment is marked as not covered.",
    "Word-removal test: on fresh experiments, the explanation's top-ranked words are deleted and compared with random words from outside that set (permutation test over the whole curve), plus a negative control that deletes the lowest-ranked words.",
    "Limits: units are words, not sub-word tokens; deleting a word also changes the grammar of the prompt; effects describe THIS prompt, model and settings; this is evidence about behaviour, not a view inside the network.",
]


def fig_png(fig, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    return buf.getvalue()


def _x(v: float) -> float:
    return float(math.exp(max(min(v, 12.0), -12.0)))


def explanation_table(e: Explanation) -> List[dict]:
    rows = []
    for i, u in enumerate(e.units):
        o = e.loo_obs[i]
        rows.append(
            {
                "#": i + 1,
                "word": u,
                "effect (×)": round(_x(e.beta[i]), 3),
                "interval low (×)": round(_x(e.ci_lo[i]), 3),
                "interval high (×)": round(_x(e.ci_hi[i]), 3),
                "q-value": float(f"{e.q[i]:.3g}"),
                "reliable": bool(e.sig[i]),
                "probability if deleted": round(o.prob, 4),
                "covered": bool(o.covered),
                "model's top choice if deleted": o.top1_token,
            }
        )
    return rows


def explanation_dict(e: Explanation) -> dict:
    d = e.deletion
    return {
        "prompt": e.prompt,
        "model": e.model,
        "backend": e.backend_name,
        "mode": e.mode,
        "target": e.target.token,
        "foil": e.foil.token if e.foil else None,
        "baseline_probability": e.base_obs.prob,
        "baseline_top": [{"token": t, "probability": p} for t, p in e.baseline_top],
        "words": explanation_table(e),
        "diagnostics": {
            "coverage": e.coverage,
            "cv_r2": None if math.isnan(e.r2_cv) else e.r2_cv,
            "method_agreement_r": None if math.isnan(e.r_methods) else e.r_methods,
            "noise_sd_nats": e.noise_sd,
            "drop_fraction_used": e.drop_frac_used,
            "experiments": e.n_experiments,
            "model_calls": e.n_requests,
            "cache_hits": e.n_cache_hits,
            "seconds": e.seconds,
        },
        "word_removal_test": {
            "ks": d["ks"],
            "p_whole_curve": d["p_auc"],
            "p_per_k": d["p_values"],
            "top_ranked_words_removed": d["top_words"],
            "probability_after_top_removal": d["top_prob"],
            "probability_after_lowest_removal": d["low_prob"],
            "random_median": d["rand_med"],
        },
        "settings": {k: getattr(e.cfg, k) for k in e.cfg.__dataclass_fields__},
        "warnings": e.warnings,
        "summary": summary(e),
    }


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


def build_report_html(e: Explanation, pngs: Dict[str, bytes]) -> str:
    esc = html.escape
    img = lambda k: f'<img alt="{k}" src="data:image/png;base64,{_b64(pngs[k])}">'
    lines = []
    for s in summary(e):
        if s.startswith("⚠"):
            lines.append(f'<li class="warn">{esc(s)}</li>')
        else:
            lines.append(f"<li>{esc(s)}</li>")
    rows = explanation_table(e)
    head = "".join(f"<th>{esc(k)}</th>" for k in rows[0])
    body = "".join(
        "<tr class='{}'>{}</tr>".format("rel" if r["reliable"] else "", "".join(f"<td>{esc(str(v))}</td>" for v in r.values()))
        for r in rows
    )
    notes = "".join(f"<li>{esc(n)}</li>" for n in METHOD_NOTES)
    cfg = ", ".join(f"{k} = {getattr(e.cfg, k)}" for k in e.cfg.__dataclass_fields__)
    sim = (
        '<p class="sim">SIMULATED MODEL: this report demonstrates the method on a toy model with planted effects. '
        "It says nothing about a real LLM.</p>"
        if e.backend_name == "simulated"
        else ""
    )

    def section(title, key):
        return f"<h2>{esc(title)}</h2>{img(key)}<p class='cap'>{esc(CAPTIONS[key])}</p>"

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Why did the model say {esc(tok(e.target.token))}?</title>
<style>
 body{{font:16px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1d1d1f;max-width:1100px;margin:32px auto;padding:0 20px}}
 h1{{font-size:28px;margin:0 0 4px}} h2{{margin-top:40px;font-size:21px}}
 img{{max-width:100%;height:auto;border:1px solid #e5e5e5;border-radius:8px}}
 .meta{{color:#6e6e73;margin:0 0 18px}} .cap{{color:#555;font-size:14.5px;margin-top:6px}}
 li.warn{{color:#b45f06;font-weight:600}} .sim{{color:#c0392b;font-weight:700}}
 table{{border-collapse:collapse;font-size:13.5px;width:100%}} th,td{{border-bottom:1px solid #e5e5e5;padding:6px 8px;text-align:left}}
 tr.rel td{{background:#fbf1ef;font-weight:600}} code{{background:#f2f2f2;padding:1px 5px;border-radius:4px}}
 details{{margin-top:36px}}
</style></head><body>
<h1>Why did the model say {esc(tok(e.target.token))}?</h1>
<p class="meta">Prompt: <code>{esc(e.prompt)}</code> · model: {esc(str(e.model))} · {datetime.now().strftime('%Y-%m-%d %H:%M')} · {e.n_requests} model calls in {e.seconds:.0f} s</p>
{sim}
<h2>In short</h2><ul>{''.join(lines)}</ul>
{img('hero')}<p class="cap">{esc(CAPTIONS['hero'])}</p>
{section('1 · Which words matter, and how sure are we?', 'effects')}
{section('2 · What happens if you delete one word?', 'counterfactual')}
{section('3 · Does the explanation predict reality?', 'faithfulness')}
{section('4 · Can you trust it?', 'trust')}
<h2>All numbers</h2><div style="overflow-x:auto"><table><tr>{head}</tr>{body}</table></div>
<details><summary><b>How this was computed</b></summary><ul>{notes}</ul><p class="cap">Settings: {esc(cfg)}</p></details>
</body></html>"""


def write_report(e: Explanation, outdir) -> Dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figs = visuals.all_figures(e)
    pngs = {k: fig_png(f, dpi=200 if k == "hero" else 150) for k, f in figs.items()}
    paths: Dict[str, Path] = {}
    for k, data in pngs.items():
        p = out / f"{k}.png"
        p.write_bytes(data)
        paths[k] = p
    paths["report"] = out / "report.html"
    paths["report"].write_text(build_report_html(e, pngs), encoding="utf-8")
    paths["json"] = out / "explanation.json"
    paths["json"].write_text(json.dumps(explanation_dict(e), indent=2, ensure_ascii=False), encoding="utf-8")
    rows = explanation_table(e)
    paths["csv"] = out / "words.csv"
    with paths["csv"].open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return paths
