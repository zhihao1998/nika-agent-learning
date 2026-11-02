import time

import httpx
import pytest

from nika_learning.backend import NikaBackend


@pytest.mark.parametrize("expected", [[123], ["123"]])
def test_job_id_encoding_does_not_reject_the_owned_backend(
    tmp_path, monkeypatch, expected
):
    original = httpx.Client

    def handle(request):
        data = (
            {"data": [{"id": "fixed"}]}
            if request.url.path == "/v1/models"
            else {"backends": [{"model": "fixed", "job_id": "123", "ready": True}]}
        )
        return httpx.Response(200, json=data)

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(transport=httpx.MockTransport(handle), **kw),
    )
    backend = NikaBackend.__new__(NikaBackend)
    backend.config = {
        "model_api": "chat_completions",
        "base_url": "http://serving/v1",
        "model": "fixed",
        "serving_job_ids": expected,
    }

    def reached(project):
        raise RuntimeError("Reached registration after valid serving preflight")

    backend.config.update({"agent_type": "meta-harness"})
    # Stop after preflight, before any native subprocess starts.
    monkeypatch.setattr("nika_learning.backend.write", lambda *args: reached(None))
    with pytest.raises(RuntimeError, match="Reached registration"):
        backend.run(tmp_path, {}, tmp_path / "attempt", time.time() + 10)


@pytest.mark.parametrize(
    "backends",
    [
        [{"model": "fixed", "job_id": "124", "ready": True}],
        [{"model": "fixed", "job_id": "123", "ready": False}],
        [],
    ],
)
def test_unusable_backend_is_rejected_before_creating_an_attempt(
    tmp_path, monkeypatch, backends
):
    original = httpx.Client

    def handle(request):
        data = (
            {"data": [{"id": "fixed"}]}
            if request.url.path == "/v1/models"
            else {"backends": backends}
        )
        return httpx.Response(200, json=data)

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(transport=httpx.MockTransport(handle), **kw),
    )
    backend = NikaBackend.__new__(NikaBackend)
    backend.config = {
        "model_api": "chat_completions",
        "base_url": "http://serving/v1",
        "model": "fixed",
        "serving_job_ids": [123],
    }
    with pytest.raises(RuntimeError, match="unexpected backend"):
        backend.run(tmp_path, {}, tmp_path / "attempt", time.time() + 10)
    assert not (tmp_path / "attempt").exists()
