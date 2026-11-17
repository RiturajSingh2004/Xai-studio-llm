#!/usr/bin/env python3
"""Command-line entry point.

  python run_explain.py --prompt "The capital of France is"          # real model via Ollama
  python run_explain.py --prompt "The capital of France is" --simulate   # no Ollama (toy model)
  python run_explain.py --prompt "The capital of France is" --check      # sanity-check the Ollama setup
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from xai_llm.backend import BackendError, OllamaBackend  # noqa: E402
from xai_llm.config import DEFAULT_HOST, DEFAULT_MODEL, PRESETS  # noqa: E402
from xai_llm.explain import ExplainConfig, baseline_top, explain  # noqa: E402
from xai_llm.narrative import summary, tok  # noqa: E402
from xai_llm.report import write_report  # noqa: E402
from xai_llm.simulate import demo_backend  # noqa: E402


def pick(top, text):
    """Find a token among the baseline top list by its exact text (leading space matters)."""
    for t, _ in top:
        if t.token == text:
            return t
    for t, _ in top:
        if t.token.strip() == text.strip():
            return t
    options = ", ".join(tok(t.token) for t, _ in top)
    sys.exit(f"Token {tok(text)} is not among the model's top predictions. Choose one of: {options}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--mode", choices=["complete", "chat"], default="complete",
                    help="complete = raw text continuation (default); chat = use the model's chat template, thinking off")
    ap.add_argument("--target", help='output token to explain, e.g. " Paris" (default: the most likely one)')
    ap.add_argument("--foil", help='optional rival token: explain "why target rather than foil"')
    ap.add_argument("--preset", choices=list(PRESETS), default="Standard")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests to Ollama")
    ap.add_argument("--out", default="outputs/run")
    ap.add_argument("--simulate", action="store_true", help="use a toy model with planted effects (no Ollama)")
    ap.add_argument("--check", action="store_true", help="print the model's top predictions twice and exit")
    args = ap.parse_args()

    be = demo_backend() if args.simulate else OllamaBackend(
        model=args.model, host=args.host, mode=args.mode, workers=args.workers)

    try:
        if args.check:
            a = be.query(args.prompt, use_cache=False)
            b = be.query(args.prompt, use_cache=False)
            print(f"Top predictions after: {args.prompt!r}")
            for k, t, lp in zip(a.keys[:20], a.tokens[:20], a.logprobs[:20]):
                print(f"  {tok(t):>18}  {np.exp(lp):7.2%}")
            common = [k for k in a.keys if k in b.as_dict()]
            drift = max((abs(a.as_dict()[k] - b.as_dict()[k]) for k in common), default=float("nan"))
            print(f"\nlisted tokens: {len(a.keys)} · probability mass listed: {np.exp(a.logprobs).sum():.1%}")
            print(f"max log-prob difference between two identical calls: {drift:.2e}")
            return

        top = baseline_top(be, args.prompt, 10)
        target = pick(top, args.target) if args.target else None
        foil = pick(top, args.foil) if args.foil else None

        def progress(frac, label):
            sys.stderr.write(f"\r[{frac:6.1%}] {label:<60}")
            sys.stderr.flush()

        e = explain(be, args.prompt, target=target, foil=foil, cfg=ExplainConfig(**PRESETS[args.preset]), progress=progress)
        sys.stderr.write("\r" + " " * 80 + "\r")
    except (BackendError, ValueError) as err:
        sys.exit(f"Error: {err}")

    paths = write_report(e, args.out)
    print("\n".join(summary(e)))
    print(f"\nOpen {paths['report']}  (figures, table and JSON are next to it)")


if __name__ == "__main__":
    main()
