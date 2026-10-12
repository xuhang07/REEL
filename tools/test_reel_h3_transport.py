from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from reel_h3_transport import (
    H3RenderSettings,
    H3TransportError,
    H3VideoClient,
    atomic_json,
    file_sha256,
    retry_delay,
)
from storymem_adapter.models import sha256_json, sha256_text


JOB_ID = "video-test-h3"


def request_for(count=3):
    return {
        "model": "MiniMax-H3-Sol", "task": "ref2va", "prompt": "A grounded multi-reference video.",
        "conditions": [{"type": "image", "uri": f"file:///pfs/test/reference-{index}.png"} for index in range(count)],
        "seconds": 5, "num_inference_steps": 30, "seed": 42,
        "target": {"short_edge": 768, "aspect_ratio": "16:9"},
    }


class FakeClock:
    def __init__(self):
        self.value = 1000.0
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.value += delay


class Response:
    def __init__(self, status=200, body=None, headers=None, content=b"test H3 video bytes"):
        self.status_code = status
        self.body = body
        self.headers = headers or {}
        self.text = json.dumps(body)
        self.content = content

    def json(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, **kwargs):
        yield self.content


class Session:
    def __init__(self, responses=None, downloads=None):
        self.responses = list(responses or [
            Response(body={"id": JOB_ID, "status": "queued"}),
            Response(body={"id": JOB_ID, "status": "completed"}),
        ])
        self.downloads = list(downloads or [Response()])
        self.calls = []
        self.trust_env = True

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def get(self, url, **kwargs):
        self.calls.append(("DOWNLOAD", url, kwargs))
        response = self.downloads.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def media_ok(path, request):
    return {"sha256": file_sha256(path), "complete_av_decode": True, "size_bytes": path.stat().st_size}


def client_for(root, session=None, clock=None, **kwargs):
    session = session or Session()
    clock = clock or FakeClock()
    return H3VideoClient(root=root, session_factory=lambda: session, media_validator=media_ok,
                         sleep=clock.sleep, clock=clock.now, **kwargs), session, clock


def state_path(root, key="candidate-1"):
    return root / "jobs" / sha256_text(key).removeprefix("sha256:") / "state.json"


def post_count(session):
    return sum(method == "POST" for method, *_ in session.calls)


def test_success_is_durable_and_replayed_without_network(tmp_path):
    client, session, _ = client_for(tmp_path)
    first = client.generate(generation_key="candidate-1", request=request_for())
    replay = client.generate(generation_key="candidate-1", request=request_for())
    assert first == replay
    assert first["h3_job_id"] == JOB_ID
    assert first["backend"] == "h3"
    assert post_count(session) == 1
    assert len(session.calls) == 3
    assert session.trust_env is False
    assert json.loads((tmp_path / "admission.json").read_text()) == {}


def test_concurrent_same_key_makes_exactly_one_post(tmp_path):
    client, session, _ = client_for(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        receipts = list(executor.map(lambda _: client.generate(generation_key="candidate-1", request=request_for()), range(8)))
    assert all(receipt == receipts[0] for receipt in receipts)
    assert post_count(session) == 1


def test_503_with_id_is_accepted_not_reposted(tmp_path):
    session = Session([Response(503, {"detail": {"id": JOB_ID}}), Response(body={"id": JOB_ID, "status": "completed"})])
    client, _, _ = client_for(tmp_path, session)
    assert client.generate(generation_key="candidate-1", request=request_for())["h3_job_id"] == JOB_ID
    assert post_count(session) == 1


def test_429_retries_with_bounded_delay(tmp_path):
    session = Session([
        Response(429, {"detail": "full"}, {"Retry-After": "3"}),
        Response(body={"id": JOB_ID, "status": "queued"}),
        Response(body={"id": JOB_ID, "status": "completed"}),
    ])
    client, _, clock = client_for(tmp_path, session)
    client.generate(generation_key="candidate-1", request=request_for())
    assert post_count(session) == 2
    assert 3 <= sum(clock.sleeps) <= 4
    assert all(0 <= delay <= 30 for delay in clock.sleeps)


@pytest.mark.parametrize("response", [
    requests.ReadTimeout("ambiguous"), Response(503, {"detail": "unknown"}),
    Response(200, {"status": "queued"}), Response(500, {"detail": "error"}),
])
def test_unknown_post_is_never_retried_even_on_new_client(tmp_path, response):
    session = Session([response])
    for _ in range(2):
        client, _, _ = client_for(tmp_path, session)
        with pytest.raises(H3TransportError) as failure:
            client.generate(generation_key="candidate-1", request=request_for())
        assert failure.value.code == "submit.unknown"
    assert post_count(session) == 1
    assert len(json.loads((tmp_path / "admission.json").read_text())) == 1


def test_crash_after_post_response_recovers_saved_id(tmp_path):
    client, session, _ = client_for(tmp_path)
    client.generate(generation_key="candidate-1", request=request_for())
    path = state_path(tmp_path)
    state = json.loads(path.read_text())
    state.update(status="submitting")
    state.pop("job_id")
    atomic_json(path, state)
    session.responses.append(Response(body={"id": JOB_ID, "status": "completed"}))
    session.downloads.append(Response())
    assert client.generate(generation_key="candidate-1", request=request_for())["h3_job_id"] == JOB_ID
    assert post_count(session) == 1


def test_crash_before_durable_response_is_unknown(tmp_path):
    client, session, _ = client_for(tmp_path)
    path = state_path(tmp_path)
    atomic_json(path, {"status": "submitting", "submit_attempt": 1, "api": client.api,
                       "request_sha256": sha256_json(request_for())})
    with pytest.raises(H3TransportError) as failure:
        client.generate(generation_key="candidate-1", request=request_for())
    assert failure.value.code == "submit.unknown"
    assert post_count(session) == 0


def test_rejected_request_releases_slot_but_does_not_retry(tmp_path):
    session = Session([Response(422, {"detail": "bad condition"})])
    client, _, _ = client_for(tmp_path, session)
    for _ in range(2):
        with pytest.raises(H3TransportError) as failure:
            client.generate(generation_key="candidate-1", request=request_for())
        assert failure.value.code == "submit.rejected"
    assert post_count(session) == 1
    assert json.loads((tmp_path / "admission.json").read_text()) == {}


def test_poll_timeout_keeps_slot_and_resumes_same_id(tmp_path):
    session = Session([Response(body={"id": JOB_ID}), Response(body={"id": JOB_ID, "status": "queued"})])
    client, _, _ = client_for(tmp_path, session, task_timeout=1)
    with pytest.raises(H3TransportError) as failure:
        client.generate(generation_key="candidate-1", request=request_for())
    assert failure.value.code == "task.timeout"
    session.responses.append(Response(body={"id": JOB_ID, "status": "completed"}))
    resumed, _, _ = client_for(tmp_path, session)
    assert resumed.generate(generation_key="candidate-1", request=request_for())["h3_job_id"] == JOB_ID
    assert post_count(session) == 1


def test_unknown_job_retains_slot_against_new_work(tmp_path):
    session = Session([requests.ReadTimeout()])
    client, _, _ = client_for(tmp_path, session, max_inflight=1)
    with pytest.raises(H3TransportError):
        client.generate(generation_key="candidate-1", request=request_for())
    with pytest.raises(H3TransportError) as failure:
        client.generate(generation_key="candidate-2", request=request_for())
    assert failure.value.code == "capacity.local_full"
    assert post_count(session) == 1


def test_get_failure_retries_get_only(tmp_path):
    session = Session([
        Response(body={"id": JOB_ID}), requests.ReadTimeout(), Response(503),
        Response(body={"id": JOB_ID, "status": "completed"}),
    ])
    client, _, clock = client_for(tmp_path, session)
    client.generate(generation_key="candidate-1", request=request_for())
    assert post_count(session) == 1
    assert max(clock.sleeps) <= 30


def test_three_missing_statuses_are_not_new_generations(tmp_path):
    session = Session([Response(body={"id": JOB_ID}), *[Response(404) for _ in range(3)]])
    client, _, _ = client_for(tmp_path, session)
    with pytest.raises(H3TransportError) as failure:
        client.generate(generation_key="candidate-1", request=request_for())
    assert failure.value.code == "status.missing"
    assert failure.value.job_id == JOB_ID
    assert post_count(session) == 1


def test_server_failure_is_terminal_and_releases_slot(tmp_path):
    session = Session([Response(body={"id": JOB_ID}), Response(body={"id": JOB_ID, "status": "failed", "error": "test failure"})])
    client, _, _ = client_for(tmp_path, session)
    with pytest.raises(H3TransportError) as failure:
        client.generate(generation_key="candidate-1", request=request_for())
    assert failure.value.code == "task.failed"
    assert json.loads((tmp_path / "admission.json").read_text()) == {}


def test_download_failure_resumes_without_post_or_poll(tmp_path):
    session = Session(downloads=[requests.ConnectionError(), Response()])
    client, _, _ = client_for(tmp_path, session)
    with pytest.raises(requests.ConnectionError):
        client.generate(generation_key="candidate-1", request=request_for())
    assert client.generate(generation_key="candidate-1", request=request_for())["h3_job_id"] == JOB_ID
    assert post_count(session) == 1
    assert sum(method == "GET" for method, *_ in session.calls) == 1


def test_changed_request_cannot_reuse_generation_key(tmp_path):
    client, session, _ = client_for(tmp_path)
    client.generate(generation_key="candidate-1", request=request_for())
    changed = copy.deepcopy(request_for())
    changed["seed"] = 43
    with pytest.raises(H3TransportError) as failure:
        client.generate(generation_key="candidate-1", request=changed)
    assert failure.value.code == "request.mismatch"
    assert post_count(session) == 1


def test_twelve_conditions_are_preserved(tmp_path):
    client, session, _ = client_for(tmp_path)
    client.generate(generation_key="candidate-1", request=request_for(12))
    assert len(session.calls[0][2]["json"]["conditions"]) == 12


@pytest.mark.parametrize("count", [0, 13])
def test_invalid_reference_counts_are_rejected_before_network(tmp_path, count):
    client, session, _ = client_for(tmp_path)
    with pytest.raises(ValueError):
        client.generate(generation_key="candidate-1", request=request_for(count))
    assert not session.calls


@pytest.mark.parametrize("seconds", [4, 0, 16, float("nan"), True])
def test_invalid_duration_is_rejected(seconds):
    with pytest.raises(ValueError):
        H3RenderSettings(seconds=seconds).validate()


@pytest.mark.parametrize("header", ["120", "nan", "bad", "-4", "Wed, 23 Sep 2026 00:00:00 GMT"])
def test_retry_delay_is_always_bounded(header):
    assert 1 <= retry_delay({"Retry-After": header}, 99, now=1000, jitter=0.8) <= 30


def test_echo_output_roots_are_rejected(tmp_path):
    atomic_json(tmp_path / "pipeline_run.json", {})
    with pytest.raises(ValueError, match="independent output root"):
        client_for(tmp_path / "h3")


def test_h3_default_concurrency_is_32_and_cannot_be_exceeded(tmp_path):
    client, session, clock = client_for(tmp_path)
    assert client.max_inflight == 32
    with pytest.raises(ValueError, match="at most 32"):
        client_for(tmp_path / "oversized", max_inflight=33)


def test_shared_admission_caps_multiple_batches_at_32(tmp_path):
    admission = tmp_path / "shared_admission"
    for index in range(33):
        client, session, clock = client_for(tmp_path / f"batch-{index}", Session([requests.ReadTimeout()]), admission_root=admission)
        with pytest.raises(H3TransportError) as failure:
            client.generate(generation_key="candidate-1", request=request_for())
        if index < 32:
            assert failure.value.code == "submit.unknown"
            assert post_count(session) == 1
        else:
            assert failure.value.code == "capacity.local_full"
            assert post_count(session) == 0
    assert len(json.loads((admission / "admission.json").read_text())) == 32
