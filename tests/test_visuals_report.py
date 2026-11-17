import io
import json

import numpy as np
import pytest
from PIL import Image

from xai_llm import visuals
from xai_llm.explain import ExplainConfig, baseline_top, explain
from xai_llm.narrative import summary
from xai_llm.report import fig_png, write_report
from xai_llm.simulate import demo_backend

PROMPT = "Please tell me quickly what the capital of France is, today"
CFG = ExplainConfig(n_samples=120, n_random=15, n_boot=150)


@pytest.fixture(scope="module")
def e():
    return explain(demo_backend(), PROMPT, cfg=CFG)


def test_all_figures_render_to_png(e):
    for name, fig in visuals.all_figures(e).items():
        png = fig_png(fig, dpi=80)
        assert png[:8] == b"\x89PNG\r\n\x1a\n", name


def test_hero_has_fixed_pixel_width_and_is_not_blank(e):
    png = fig_png(visuals.hero_figure(e), dpi=100)
    im = Image.open(io.BytesIO(png)).convert("RGB")
    assert im.size[0] == 1100 and 250 < im.size[1] < 900
    colours = np.unique(np.asarray(im).reshape(-1, 3), axis=0)
    assert len(colours) > 50                    # chips, text and gradient are really drawn


def test_hero_wraps_long_prompts_and_contrast_mode_renders():
    long_prompt = " ".join(["The capital of France is"] * 1) + ", " + "please answer " * 6 + "now"
    be = demo_backend()
    e2 = explain(be, long_prompt, cfg=ExplainConfig(n_samples=80, n_random=10, n_boot=100))
    h1 = visuals.hero_figure(e2).get_size_inches()[1]
    assert h1 > visuals.hero_figure(explain(be, "The capital of France is", cfg=ExplainConfig(n_samples=60, n_random=10, n_boot=100))).get_size_inches()[1]
    top = {t.token: t for t, _ in baseline_top(be, PROMPT, 10)}
    e3 = explain(be, PROMPT, target=top[" Paris"], foil=top[" London"], cfg=CFG)
    assert fig_png(visuals.hero_figure(e3), dpi=60)[:4] == b"\x89PNG"


def test_summary_is_plain_english_and_mentions_the_key_words(e):
    text = " ".join(summary(e))
    assert "France" in text and "capital" in text and "permutation" in text and "×" in text


def test_write_report_creates_every_file(e, tmp_path):
    paths = write_report(e, tmp_path / "out")
    for key in ("hero", "effects", "counterfactual", "faithfulness", "trust", "report", "json", "csv"):
        assert paths[key].exists() and paths[key].stat().st_size > 0, key
    html = paths["report"].read_text(encoding="utf-8")
    assert "Why did the model say" in html and "SIMULATED MODEL" in html and "data:image/png;base64" in html
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["target"] == " Paris" and len(data["words"]) == len(e.units)
    assert data["backend"] == "simulated"
