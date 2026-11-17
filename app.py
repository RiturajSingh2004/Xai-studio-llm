"""Streamlit app:  streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from xai_llm import visuals
from xai_llm.backend import BackendError, OllamaBackend
from xai_llm.config import DEFAULT_HOST, DEFAULT_MODEL, MAX_UNITS, PRESETS
from xai_llm.explain import ExplainConfig, baseline_top, explain
from xai_llm.narrative import summary, tok
from xai_llm.report import CAPTIONS, METHOD_NOTES, build_report_html, explanation_dict, explanation_table, fig_png
from xai_llm.simulate import demo_backend

import json

st.set_page_config(page_title="XAI Studio · Why did the model say that?", page_icon="🔍", layout="wide")
st.title("🔍 XAI Studio · Why did the model say that?")
st.caption(
    "Type a prompt, pick the output you want explained, and see which words caused it, "
    "with confidence intervals and a test of whether the explanation is faithful."
)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Model")
    simulate = st.checkbox("Simulated model (no Ollama) – preview only", value=False)
    host = st.text_input("Ollama address", DEFAULT_HOST, disabled=simulate)
    model = st.text_input("Model", DEFAULT_MODEL, disabled=simulate)
    mode_label = st.radio(
        "Prompt style",
        ["Plain text continuation", "Chat answer"],
        help="Plain text: the model simply continues your text (best for 'The capital of France is'). "
        "Chat answer: your text is sent as a user message with thinking switched off.",
        disabled=simulate,
    )
    workers = st.slider("Parallel requests", 1, 8, 4, help="Raise it if your GPU has headroom (set OLLAMA_NUM_PARALLEL).")

    st.header("Thoroughness")
    preset = st.radio("Preset", list(PRESETS), index=1,
                      help="More experiments = narrower confidence intervals and a more powerful faithfulness test, but slower.")
    with st.expander("Advanced statistics"):
        fdr = st.slider("False discovery rate", 0.01, 0.30, 0.10, 0.01)
        ci = st.select_slider("Confidence level", [0.80, 0.90, 0.95, 0.99], value=0.95)
        min_effect = st.slider("Smallest effect that matters (nats)", 0.0, 0.5, 0.10, 0.01,
                               help="0.10 nats ≈ a 10% change in probability. Smaller effects are never called reliable.")
        min_cov = st.slider("Target coverage", 0.5, 1.0, 0.95, 0.01)
        seed = st.number_input("Random seed", 0, 10_000, 0)

if simulate:
    st.warning("SIMULATED MODEL: a toy model with planted word effects. It demonstrates the method and says nothing about a real LLM.")


def get_backend():
    key = (simulate, host, model, mode_label, workers)
    if st.session_state.get("backend_key") != key:
        if simulate:
            be = demo_backend()
        else:
            be = OllamaBackend(model=model, host=host, workers=workers,
                               mode="complete" if mode_label.startswith("Plain") else "chat")
        be.workers = workers
        st.session_state["backend"], st.session_state["backend_key"] = be, key
        st.session_state.pop("pre", None)
    return st.session_state["backend"]


# ------------------------------------------------------------------ step 1
prompt = st.text_area("Your prompt", value="Please tell me quickly what the capital of France is, today" if simulate else "The capital of France is",
                      height=100, help=f"Up to {MAX_UNITS} words/punctuation marks.")

if st.button("① Check the model's prediction", type="primary"):
    try:
        with st.spinner("Asking the model…"):
            top = baseline_top(get_backend(), prompt, 10)
        st.session_state["pre"] = {"prompt": prompt, "top": top}
    except (BackendError, ValueError) as err:
        st.error(str(err))

pre = st.session_state.get("pre")
if pre and pre["prompt"] != prompt:
    st.info("The prompt changed. Press ① again.")
    pre = None

# ------------------------------------------------------------------ step 2
if pre:
    top = pre["top"]
    st.markdown("**What the model would write next** (most likely first):")
    st.dataframe(
        [{"output": tok(t.token), "probability": f"{p:.1%}"} for t, p in top],
        hide_index=True, use_container_width=False,
    )
    c1, c2 = st.columns(2)
    ti = c1.selectbox("Explain this output", range(len(top)),
                      format_func=lambda i: f"{tok(top[i][0].token)}  ·  {top[i][1]:.1%}")
    fi = c2.selectbox("…compared with (optional)", [None] + list(range(len(top))),
                      format_func=lambda i: "— nothing, just explain it —" if i is None else f"{tok(top[i][0].token)}  ·  {top[i][1]:.1%}",
                      help="Pick a rival to answer 'why this rather than that?'. It also keeps the comparison observable more often.")

    if st.button("② Explain it", type="primary"):
        bar = st.progress(0.0, text="Starting…")
        cfg = ExplainConfig(**PRESETS[preset], fdr=fdr, ci=ci, min_effect=min_effect, min_coverage=min_cov, seed=int(seed))
        try:
            e = explain(
                get_backend(), prompt,
                target=top[ti][0], foil=None if fi is None else top[fi][0],
                cfg=cfg, progress=lambda f, label: bar.progress(min(max(f, 0.0), 1.0), text=label),
            )
        except (BackendError, ValueError) as err:
            bar.empty()
            st.error(str(err))
        else:
            bar.empty()
            with st.spinner("Drawing figures…"):
                pngs = {k: fig_png(f, dpi=200 if k == "hero" else 150) for k, f in visuals.all_figures(e).items()}
                st.session_state["result"] = {"e": e, "pngs": pngs, "html": build_report_html(e, pngs)}

# ------------------------------------------------------------------ results
res = st.session_state.get("result")
if res:
    e, pngs = res["e"], res["pngs"]
    st.divider()
    st.subheader("In short")
    for line in summary(e):
        if line.startswith("⚠"):
            st.warning(line[1:].strip())
        else:
            st.markdown("- " + line)

    st.image(pngs["hero"])
    st.caption(CAPTIONS["hero"])

    st.subheader("1 · Which words matter, and how sure are we?")
    st.image(pngs["effects"])
    st.caption(CAPTIONS["effects"])

    st.subheader("2 · What happens if you delete one word?")
    st.image(pngs["counterfactual"])
    st.caption(CAPTIONS["counterfactual"])

    st.subheader("3 · Does the explanation predict reality?")
    st.image(pngs["faithfulness"])
    st.caption(CAPTIONS["faithfulness"])

    st.subheader("4 · Can you trust it?")
    st.image(pngs["trust"])
    st.caption(CAPTIONS["trust"])

    st.subheader("All numbers")
    st.dataframe(explanation_table(e), hide_index=True)

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Full report (HTML)", res["html"], "report.html", "text/html")
    d2.download_button("⬇ Hero picture (PNG)", pngs["hero"], "explanation.png", "image/png")
    d3.download_button("⬇ Numbers (JSON)", json.dumps(explanation_dict(e), indent=2, ensure_ascii=False),
                       "explanation.json", "application/json")
    with st.expander("How this was computed"):
        for n in METHOD_NOTES:
            st.markdown("- " + n)
