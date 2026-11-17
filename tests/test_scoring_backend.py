import io
import math
import urllib.error

import numpy as np
import pytest

from xai_llm.backend import Backend, BackendError, OllamaBackend, make_topk, parse_ollama_logprobs, token_key
from xai_llm.scoring import observe


def tk(**lp):
    return make_topk([(t, v, None) for t, v in lp.items()])


A, B, C = token_key("A"), token_key("B"), token_key("C")


def test_observe_single_covered_and_missing():
    t = tk(A=math.log(0.6), B=math.log(0.3), C=math.log(0.1))
    o = observe(t, A)
    assert o.covered and abs(o.prob - 0.6) < 1e-9 and o.top1_token == "A"
    o = observe(t, token_key("Z"))           # not listed: bounded by the smallest listed value
    assert not o.covered and abs(o.prob - 0.1) < 1e-9


def test_observe_contrast_cases():
    t = tk(A=math.log(0.6), B=math.log(0.3), C=math.log(0.1))
    o = observe(t, A, B)
    assert o.covered and abs(o.score - math.log(2)) < 1e-9 and abs(o.prob - 2 / 3) < 1e-9
    o = observe(t, token_key("Z"), B)        # target missing -> upper bound
    assert not o.covered and o.score == pytest.approx(math.log(0.1) - math.log(0.3))
    o = observe(t, A, token_key("Z"))        # foil missing -> lower bound
    assert not o.covered and o.score == pytest.approx(math.log(0.6) - math.log(0.1))


def test_parse_documented_ollama_structure():
    resp = {
        "response": " Paris",
        "logprobs": [{
            "token": " Paris", "logprob": -0.1, "bytes": [32, 80, 97, 114, 105, 115],
            "top_logprobs": [
                {"token": " Paris", "logprob": -0.1, "bytes": [32, 80, 97, 114, 105, 115]},
                {"token": " the", "logprob": -3.0, "bytes": [32, 116, 104, 101]},
            ],
        }],
    }
    t = parse_ollama_logprobs(resp)
    assert t.tokens == [" Paris", " the"]
    assert t.keys[0] == bytes([32, 80, 97, 114, 105, 115]).hex()
    assert len(t.keys) == 2                         # sampled token is not duplicated
    assert np.all(np.diff(t.logprobs) <= 0)         # sorted by probability


def test_parse_adds_sampled_token_when_missing_from_top_list():
    t = parse_ollama_logprobs({"logprobs": [{"token": "X", "logprob": -9.0, "top_logprobs": []}]})
    assert t.tokens == ["X"]


def test_missing_logprobs_gives_actionable_error():
    with pytest.raises(BackendError, match="Update Ollama"):
        parse_ollama_logprobs({"response": "x"})


def _http_error(code, body=b""):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body))


def test_ollama_404_tells_you_to_pull_the_model():
    be = OllamaBackend(model="qwen3.5:4b")
    be._post = lambda payload: (_ for _ in ()).throw(_http_error(404))
    with pytest.raises(BackendError, match="ollama pull qwen3.5:4b"):
        be.query("hi")


def test_ollama_unreachable_message():
    be = OllamaBackend()
    be._post = lambda payload: (_ for _ in ()).throw(urllib.error.URLError("refused"))
    with pytest.raises(BackendError, match="ollama serve"):
        be.query("hi")


def test_ollama_reduces_top_logprobs_when_server_limits_it():
    be = OllamaBackend(top_logprobs=20)
    seen = []

    def post(payload):
        seen.append(payload["top_logprobs"])
        if payload["top_logprobs"] > 5:
            raise _http_error(400, b'{"error":"top_logprobs must be <= 5"}')
        return {"logprobs": [{"token": "A", "logprob": -0.1, "top_logprobs": []}]}

    be._post = post
    assert be.query("hi").tokens == ["A"]
    assert seen == [20, 10, 5]


def test_payload_uses_raw_for_complete_and_think_false_for_chat():
    p = OllamaBackend(mode="complete")._payload("x", 20)
    assert p["raw"] is True and "think" not in p and p["options"]["num_predict"] == 1 and p["logprobs"] is True
    p = OllamaBackend(mode="chat")._payload("x", 20)
    assert p["think"] is False and "raw" not in p


class Counting(Backend):
    def _fetch(self, prompt):
        return tk(A=-0.1, B=-3.0)


def test_cache_and_parallel_query_many_dedupe_requests():
    be = Counting(workers=4)
    seen = []
    out = be.query_many(["a", "b", "a", "c", "b"], on_progress=lambda k: seen.append(k))
    assert len(out) == 5 and be.requests == 3
    assert sum(seen) == 5                       # progress counts every requested prompt
    be.query_many(["a", "b"])
    assert be.requests == 3 and be.cache_hits >= 2
