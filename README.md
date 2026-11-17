# XAI Studio · LLM word attribution for Ollama

Type a prompt, pick the output you want explained, and see **which words caused it**, with confidence
intervals, a faithfulness test and trust checks. Built for a local **Qwen3.5 4B served by Ollama**.

Ollama exposes probabilities but no gradients, so the model is treated as a black box and studied with
controlled experiments (delete words, ask again, test the results statistically).

## Quick start

```bash
# 1. model
ollama pull qwen3.5:4b           # then keep Ollama running

# 2. python env
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. check the setup (prints top predictions twice, shows how deterministic the server is)
python run_explain.py --prompt "The capital of France is" --check

# 4a. app
streamlit run app.py

# 4b. or command line -> outputs/run/report.html
python run_explain.py --prompt "The capital of France is"
python run_explain.py --prompt "The capital of France is" --target " Paris" --foil " London"   # "why Paris rather than London?"
```

No Ollama yet? Tick **Simulated model** in the app sidebar or pass `--simulate` to preview everything on a toy model
(clearly labelled; it says nothing about a real LLM).

Ollama must be recent enough to return log-probabilities (`logprobs` on `/api/generate`). If it is not, the app tells you.

## What you get

| Figure | The question it answers |
|---|---|
| **Hero** (word chips) | Which words matter? Red ▲ ×N = raises the probability N-fold, blue ▼ = lowers it, dashed grey *n.s.* = no reliable effect |
| **Effects with intervals** | How big is each effect and how sure are we? (95 % bootstrap interval, log scale) |
| **Delete one word** | What literally happens if this word is removed? (exact, no statistics) |
| **Faithfulness test** | Does the explanation predict reality? Removing its top words must hurt far more than removing other words (permutation test) |
| **Trust dashboard** | Coverage, held-out fit and agreement between two estimators, each with a pass/fail goal |

Every run also writes a plain-English summary, a CSV/JSON of all numbers and a self-contained `report.html`.

## How it works

1. **Remove-one:** each word is deleted alone → exact counterfactual probabilities.
2. **Random removals:** hundreds of random subsets of words are deleted; the score (log P of the target, or log-odds target vs foil) is regressed on which words were kept. Each coefficient is that word's average effect (a Shapley-style estimate). Bootstrap intervals + p-values, Benjamini–Hochberg false-discovery control, and a minimum effect size (≥ 0.10 nats and ≥ 3× the model's own run-to-run noise) decide what is called *reliable*.
3. **Coverage:** the server only returns its top-N tokens. If the target drops out of that list its probability is only known to be ≤ the smallest listed value. Such experiments are flagged, counted as *coverage*, and the share of words removed per experiment **adapts automatically** to keep coverage ≥ 95 %. Contrast mode ("A rather than B") stays observable far more often.
4. **Faithfulness:** on fresh experiments, the explanation's top-ranked words are removed and compared with random words from outside that set; a negative control removes its lowest-ranked words.
5. **Trust checks:** coverage, 5-fold cross-validated R² of the word-by-word model, and Pearson agreement of the two estimators. Warnings appear automatically when a goal is missed.

Presets (model calls grow with prompt length; identical prompts are cached): Quick ≈ 150–400 · Standard ≈ 250–650 · Thorough ≈ 450–1,500. A short prompt on a 4B model typically takes seconds to a few minutes; raise *Parallel requests* and set `OLLAMA_NUM_PARALLEL` if your GPU has headroom. The faithfulness test always uses at least 39 random draws, because with fewer the smallest possible p-value is above 0.05 and the test could never pass.

## Tips for clear results

- Prompts of **8–40 words** work best. Very short prompts leave few words to compare.
- Use **Plain text continuation** for sentence-completion prompts ("The capital of France is"); use **Chat answer** for questions (thinking is switched off so the first token is the answer, not a reasoning preamble).
- Tokens keep their leading space: `" Paris"` and `"Paris"` are different outputs.
- If coverage is low, pick a likelier output or use the **compared with** option.

## Honest limits

- Units are **words**, not sub-word tokens (Ollama accepts text).
- Deleting a word also changes grammar; effects describe *this* prompt, model and settings.
- Heavily overlapping words (synonyms, repeated phrases) can hide each other in remove-one; the random-removal estimate and the *held-out fit* check show when this happens.
- This is evidence about behaviour, not a view inside the network.

## Tests

```bash
pytest          # 37 tests: segmentation, statistics, Ollama parsing/errors, planted-effect recovery, figures, report
```

On the simulated model with planted effects (12 seeds × 3 noise levels, 10 filler words each) both planted words
were found in every run, with **0 false positives in 360 filler-word tests** and the faithfulness test passed every time.
That validates the method, not Qwen: real models have interactions the toy model lacks, and the trust dashboard exists to show when that matters.

## Layout

```
app.py                 Streamlit app
run_explain.py         CLI (report, --check, --simulate)
xai_llm/
  backend.py           Ollama client (parallel, cached, retries, clear errors)
  segment.py           prompt -> words, and back
  scoring.py           log-prob / log-odds + coverage bookkeeping
  explain.py           the attribution engine
  stats.py             OLS, bootstrap, BH-FDR, cross-validation (numpy only)
  visuals.py           figures        narrative.py  plain-English summary
  report.py            HTML/JSON/CSV  simulate.py   toy model with planted effects
tests/
```
