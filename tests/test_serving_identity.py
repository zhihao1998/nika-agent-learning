import time

import httpx
import pytest

from nika_learning.backend import NikaBackend
from nika_learning.store import write


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


@pytest.mark.parametrize("active,waiting,allowed", [(0, 0, 0), (1, 0, 1)])
def test_exclusive_serving_accepts_idle_or_own_request(
    monkeypatch, active, waiting, allowed
):
    from nika_learning.backend import serving_identity

    original = httpx.Client

    def handle(request):
        return httpx.Response(
            200,
            json=(
                {"data": [{"id": "fixed"}]}
                if request.url.path == "/v1/models"
                else {
                    "backends": [
                        {
                            "model": "fixed",
                            "job_id": "123",
                            "ready": True,
                            "active_requests": active,
                            "waiting": waiting,
                        }
                    ]
                }
            ),
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(transport=httpx.MockTransport(handle), **kw),
    )
    serving_identity(
        {
            "model_api": "chat_completions",
            "base_url": "http://serving/v1",
            "model": "fixed",
            "serving_job_ids": [123],
            "serving_exclusive": True,
        },
        active_limit=allowed,
    )


@pytest.mark.parametrize(
    "telemetry",
    [{"active_requests": 16, "waiting": 0}, {"active_requests": 0, "waiting": 5}, {}],
)
def test_busy_or_unobservable_service_does_not_consume_an_attempt(
    tmp_path, monkeypatch, telemetry
):
    original = httpx.Client

    def handle(request):
        return httpx.Response(
            200,
            json=(
                {"data": [{"id": "fixed"}]}
                if request.url.path == "/v1/models"
                else {
                    "backends": [
                        {"model": "fixed", "job_id": "123", "ready": True, **telemetry}
                    ]
                }
            ),
        )

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
        "serving_exclusive": True,
    }
    with pytest.raises(RuntimeError, match="busy|telemetry"):
        backend.run(tmp_path, {}, tmp_path / "attempt-01", time.time() + 10)
    assert not (tmp_path / "attempt-01").exists()


@pytest.mark.parametrize(
    "infra,expected,score", [(True, "infra_failed", None), (False, "agent_failed", 0.0)]
)
def test_native_infrastructure_flag_controls_retry_without_rewriting_native_state(
    tmp_path, infra, expected, score
):
    trial = tmp_path / "nika/native/trials/trial"
    native = {
        "outcome": "agent_failed",
        "session_id": "session",
        "agent_error": "503" if infra else "timeout",
    }
    write(trial / "run.json", native)
    write(
        trial / "eval_metrics.json",
        {"validity_flags": ["infra_error"] if infra else ["empty_report"]},
    )
    feedback = NikaBackend.collect(tmp_path)
    assert feedback["outcome"] == expected and feedback["score"] == score
    assert feedback["native_outcome"] == "agent_failed"
    assert __import__("json").loads((trial / "run.json").read_text()) == native


@pytest.mark.parametrize("already_exited", [False, True])
def test_mid_trial_contention_interrupts_and_cleans_without_recording_quality_zero(
    tmp_path, monkeypatch, already_exited
):
    import signal
    import subprocess
    from pathlib import Path
    from nika_learning.store import inventory, read

    cfg = read(Path(__file__).parents[1] / "configs/service.json")
    cfg["serving_exclusive"] = True
    backend = NikaBackend.__new__(NikaBackend)
    backend.config, backend.env, backend.root = cfg, {}, tmp_path
    backend.checkout, backend.python = tmp_path, Path("python")
    calls, killed, cleaned = [], [], []

    def serving(config, *, active_limit=0):
        calls.append(active_limit)
        if active_limit:
            raise RuntimeError("Shared serving is busy: active=16, waiting=0")
        return {"backends": []}

    class Process:
        pid, returncode = 12345, None

        def wait(self, timeout=None):
            if not killed:
                raise subprocess.TimeoutExpired("native", timeout)
            self.returncode = -signal.SIGTERM
            return self.returncode

    def cleanup(attempt):
        cleaned.append(attempt)
        write(attempt / "cleanup.json", [])
        return []

    monkeypatch.setattr("nika_learning.backend.serving_identity", serving)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: Process())

    def terminate(pid, sig):
        killed.append((pid, sig))
        if already_exited:
            raise ProcessLookupError

    monkeypatch.setattr("nika_learning.backend.os.killpg", terminate)
    monkeypatch.setattr("nika_learning.backend.wait_process_group", lambda pid: None)
    monkeypatch.setattr(backend, "cleanup", cleanup)
    attempt = tmp_path / "attempt-01"
    with pytest.raises(RuntimeError, match="busy"):
        backend.run(tmp_path / "project", {}, attempt, time.time() + 60)
    assert calls == [0, 1]
    assert killed == [(12345, signal.SIGTERM)] and cleaned == [attempt]
    assert (attempt / "interrupted.json").exists()
    assert (attempt / "serving-failure.json").exists()
    assert not (attempt / "feedback.json").exists()
    assert inventory(attempt) == read(attempt / "artifacts.json")
