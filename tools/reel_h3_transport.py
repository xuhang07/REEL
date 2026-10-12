"""Durable H3 R2AV client, independent of the sealed Echo production stack."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import re
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import quote, urlparse

import requests

from reel_h3_adapter import (
    H3PreparedPrompt,
    H3PromptCompilation,
    MAX_REFERENCES,
    validate_h3_prompt,
    validate_observation,
)
from storymem_adapter.models import canonical_json, sha256_bytes, sha256_json, sha256_text


VERSION = "reel.h3.transport.v1"
DEFAULT_API = os.environ.get("REEL_H3_API_URL", "http://127.0.0.1:3826")
MAX_INFLIGHT = 32
TERMINAL_STATES = frozenset({"completed", "server_completed", "failed", "rejected", "media_invalid"})
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class H3TransportError(RuntimeError):
    def __init__(self, code: str, message: str, *, job_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.job_id = job_id


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def durable_lock(path: Path):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(str(path), threading.RLock())
    with thread_lock, path.open("a+b") as stream:
        fcntl.lockf(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.lockf(stream.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class H3RenderSettings:
    seconds: float = 5.0
    steps: int = 30
    short_edge: int = 768
    aspect_ratio: str = "16:9"
    seed: int = 42

    def validate(self) -> None:
        if isinstance(self.seconds, bool) or not 5 <= self.seconds <= 15:
            raise ValueError("H3 seconds must be within [5, 15]; 4 seconds fails in the model")
        if type(self.steps) is not int or not 2 <= self.steps <= 100:
            raise ValueError("H3 steps must be an integer within [2, 100]")
        if type(self.short_edge) is not int or self.short_edge <= 0 or self.short_edge % 32:
            raise ValueError("H3 short_edge must be a positive multiple of 32")
        if type(self.seed) is not int:
            raise ValueError("H3 seed must be an integer")
        try:
            width, height = map(float, self.aspect_ratio.split(":"))
            if not all(math.isfinite(value) and value > 0 for value in (width, height)):
                raise ValueError()
        except (TypeError, AttributeError, ValueError) as error:
            raise ValueError("H3 aspect_ratio must be positive W:H") from error


def stage_request(
    compilation: H3PromptCompilation,
    *,
    spool_root: Path,
    settings: H3RenderSettings,
) -> tuple[dict, dict]:
    settings.validate()
    document = compilation.document
    unsigned = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if document.get("artifact_sha256") != sha256_json(unsigned):
        raise H3TransportError("pe.integrity", "H3 PE artifact hash does not match")
    if document.get("prompt_sha256") != sha256_text(document["prompt"]):
        raise H3TransportError("pe.integrity", "H3 PE prompt hash does not match")
    prepared = H3PreparedPrompt(canonical_json(document["prepared"]), ())
    validate_observation(canonical_json(document["observation"]), prepared)
    validate_h3_prompt(document["prompt"], prepared)
    references = document["prepared"]["reference_lineage"]
    if not 1 <= len(references) <= MAX_REFERENCES:
        raise H3TransportError("reference.count", "H3 R2AV requires one through twelve images")
    spool_root = spool_root.resolve()
    if not spool_root.is_relative_to("/pfs"):
        raise H3TransportError("reference.shared_path", "H3 workers mount /pfs, not /pfs2 or /tmp")
    staged = []
    conditions = []
    for index, reference in enumerate(references, 1):
        if reference["label"] != f"<Image {index}>":
            raise H3TransportError("reference.order", "Image labels must follow condition order")
        source = Path(reference["content_path"])
        payload = source.read_bytes()
        digest = sha256_bytes(payload)
        if digest != reference["content_sha256"] or len(payload) != reference["size_bytes"]:
            raise H3TransportError("reference.integrity", "Authorized reference bytes changed")
        suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(reference["mime_type"])
        if suffix is None:
            raise H3TransportError("reference.mime", "Only verified image references are supported")
        filename = digest.removeprefix("sha256:") + suffix
        destination = spool_root / filename[:2] / filename
        with durable_lock(destination.with_suffix(".lock")):
            if destination.exists():
                if file_sha256(destination) != digest:
                    raise H3TransportError("reference.spool_corrupt", "Shared input bytes changed")
            else:
                descriptor, temporary = tempfile.mkstemp(prefix=".image-", dir=destination.parent)
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.chmod(temporary, 0o644)
                    os.replace(temporary, destination)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
        uri = destination.as_uri()
        if uri.removeprefix("file://") != str(destination):
            raise H3TransportError("reference.shared_path", "H3 shared paths must not require URI unescaping")
        conditions.append({"type": "image", "uri": uri})
        staged.append({**reference, "shared_path": str(destination), "uri": uri})
    request = {
        "model": "MiniMax-H3-Sol",
        "task": "ref2va",
        "prompt": document["prompt"],
        "conditions": conditions,
        "seconds": settings.seconds,
        "num_inference_steps": settings.steps,
        "seed": settings.seed,
        "target": {"short_edge": settings.short_edge, "aspect_ratio": settings.aspect_ratio},
    }
    lineage = {
        "pe_artifact_path": str(compilation.artifact_path),
        "pe_artifact_sha256": document["artifact_sha256"],
        "prompt_sha256": document["prompt_sha256"],
        "references": staged,
        "temporal_roles_enforcement": "prompt_semantics_only_not_pixel_locked",
    }
    return request, lineage


def retry_delay(headers: Mapping[str, str], attempt: int, *, now: float, jitter: float) -> float:
    value = next((value for key, value in headers.items() if key.lower() == "retry-after"), "0")
    try:
        advised = float(value)
    except (ValueError, TypeError):
        try:
            advised = parsedate_to_datetime(value).timestamp() - now
        except (ValueError, TypeError, OverflowError):
            advised = 0.0
    if not math.isfinite(advised):
        advised = 0.0
    return min(30.0, max(0.0, advised, 2.0 ** min(max(attempt - 1, 0), 5)) + jitter)


def validate_media(path: Path, request: Mapping[str, Any]) -> dict:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    media = json.loads(probe.stdout)
    videos = [stream for stream in media.get("streams", []) if stream.get("codec_type") == "video"]
    audio = [stream for stream in media.get("streams", []) if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or not audio:
        raise H3TransportError("media.invalid", "H3 output must contain one video stream and generated audio")
    video = videos[0]
    requested_frames = round(float(request["seconds"]) * 24)
    expected_frames = 5 + 17 * math.ceil((requested_frames - 5) / 17)
    if int(video.get("nb_frames", 0)) != expected_frames:
        raise H3TransportError("media.invalid", "H3 output frame count differs from requested VAE-aligned duration")
    numerator, denominator = map(float, video["r_frame_rate"].split("/"))
    if denominator <= 0 or abs(numerator / denominator - 24) > 0.001:
        raise H3TransportError("media.invalid", "H3 output must use 24 fps")
    width, height = int(video["width"]), int(video["height"])
    if min(width, height) != request["target"]["short_edge"]:
        raise H3TransportError("media.invalid", "H3 output resolution differs from requested short edge")
    aspect_width, aspect_height = map(float, request["target"]["aspect_ratio"].split(":"))
    if abs((width / height) / (aspect_width / aspect_height) - 1) > 0.05:
        raise H3TransportError("media.invalid", "H3 output aspect ratio differs from request")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-threads", "1", "-i", str(path),
         "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
        check=True, capture_output=True, timeout=120,
    )
    return {"probe": media, "complete_av_decode": True, "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


class H3VideoClient:
    def __init__(
        self,
        *,
        root: Path,
        api: str = DEFAULT_API,
        max_inflight: int = MAX_INFLIGHT,
        admission_root: Path | None = None,
        http_concurrency: int = 32,
        task_timeout: float = 3600,
        submit_timeout: float = 120,
        poll_interval: float = 5,
        session_factory: Callable = requests.Session,
        media_validator: Callable = validate_media,
        sleep: Callable = time.sleep,
        clock: Callable = time.time,
    ) -> None:
        parsed = urlparse(api)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("H3 API must be an HTTP(S) origin without embedded credentials")
        if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("H3 API must not contain a path, query, or fragment")
        if min(max_inflight, http_concurrency, task_timeout, submit_timeout, poll_interval) <= 0:
            raise ValueError("H3 concurrency and timeout settings must be positive")
        if type(max_inflight) is not int or max_inflight > MAX_INFLIGHT:
            raise ValueError("H3 permits at most 32 unfinished jobs, including queued and uncertain submissions")
        self.root = root.resolve()
        if any((parent / "pipeline_run.json").exists() for parent in (self.root, *self.root.parents)):
            raise ValueError("H3 transport needs an independent output root, not an Echo production tree")
        self.api = api.rstrip("/")
        self.max_inflight = max_inflight
        self.admission_root = admission_root.resolve() if admission_root is not None else self.root
        self.task_timeout = task_timeout
        self.submit_timeout = submit_timeout
        self.poll_interval = min(poll_interval, 30)
        self.session_factory = session_factory
        self.media_validator = media_validator
        self.sleep = sleep
        self.clock = clock
        self.local = threading.local()
        self.http_gate = threading.BoundedSemaphore(http_concurrency)
        self.media_gate = threading.BoundedSemaphore(min(http_concurrency, 8))

    def _session(self):
        if not hasattr(self.local, "session"):
            self.local.session = self.session_factory()
            self.local.session.trust_env = False
        return self.local.session

    def _json_request(self, method: str, route: str, payload: dict | None = None) -> dict:
        with self.http_gate:
            with self._session().request(
                method, self.api + route, json=payload, timeout=(5, 60), allow_redirects=False,
            ) as response:
                try:
                    body = response.json()
                except ValueError:
                    body = None
                return {"http_status": response.status_code, "headers": dict(response.headers),
                        "body": body, "text": response.text[:16000] if body is None else None}

    def capacity(self) -> dict:
        response = self._json_request("GET", "/v1/capacity")
        if response["http_status"] != 200 or not isinstance(response["body"], dict):
            raise H3TransportError("capacity.unavailable", "H3 capacity probe failed")
        body = response["body"]
        if body.get("pool_role") != "generator" or body.get("admission_backend") != "shared-two-pool-v1":
            raise H3TransportError("capacity.wrong_backend", "Expected the documented H3 generator pool")
        return body

    def _save(self, directory: Path, state: dict, status: str | None = None, **fields) -> None:
        state.update(fields)
        if status is not None:
            state["status"] = status
        state["updated_at"] = self.clock()
        atomic_json(directory / "state.json", state)

    def _slot(self, key: str, *, release: bool = False) -> None:
        state_path = self.root / "jobs" / key / "state.json"
        slot_key = sha256_text(str(state_path)).removeprefix("sha256:")
        with durable_lock(self.admission_root / ".admission.lock"):
            path = self.admission_root / "admission.json"
            holds = json.loads(path.read_text()) if path.exists() else {}
            for held in list(holds):
                held_path = Path(holds[held].get("state_path") or self.root / "jobs" / held / "state.json")
                if held_path.exists() and json.loads(held_path.read_text()).get("status") in TERMINAL_STATES:
                    holds.pop(held)
            if release:
                holds.pop(slot_key, None)
            elif slot_key not in holds:
                owned = sum(entry.get("root", str(self.root)) == str(self.root) for entry in holds.values())
                if len(holds) >= MAX_INFLIGHT or owned >= self.max_inflight:
                    raise H3TransportError("capacity.local_full", "Unfinished H3 jobs retain their slots; resume them before adding work")
                holds[slot_key] = {"reserved_at": self.clock(), "state_path": str(state_path), "root": str(self.root)}
            atomic_json(path, holds)

    @staticmethod
    def _accepted_id(response: Mapping[str, Any]) -> str | None:
        body = response.get("body")
        if not isinstance(body, dict):
            return None
        if response["http_status"] == 200:
            job_id = body.get("id")
        elif response["http_status"] == 503 and isinstance(body.get("detail"), dict):
            job_id = body["detail"].get("id")
        else:
            return None
        return job_id if isinstance(job_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", job_id) else None

    def _apply_submit_response(self, directory: Path, state: dict, response: dict) -> None:
        job_id = self._accepted_id(response)
        if job_id:
            self._save(directory, state, "accepted", job_id=job_id, accepted_at=self.clock())
        elif response["http_status"] == 429:
            pause = retry_delay(response["headers"], state["submit_attempt"], now=self.clock(), jitter=random.random())
            self._save(directory, state, "not_admitted", retry_at=self.clock() + pause)
        elif response["http_status"] in {400, 401, 403, 404, 405, 409, 413, 415, 422}:
            self._save(directory, state, "rejected", error=response["body"] or response["text"])
        else:
            self._save(directory, state, "submit_unknown", error="POST response did not establish whether a job was accepted")

    def _submit(self, directory: Path, state: dict, request: dict) -> None:
        key = directory.name
        if state["status"] == "submitting":
            saved_response = directory / f"submit_response_{state['submit_attempt']:04d}.json"
            if saved_response.exists():
                self._apply_submit_response(directory, state, json.loads(saved_response.read_text()))
            else:
                self._save(directory, state, "submit_unknown", error="Interrupted POST has no durable response; do not resubmit")
        if state["status"] == "submit_unknown":
            raise H3TransportError("submit.unknown", state["error"])
        if state["status"] == "rejected":
            self._slot(key, release=True)
            raise H3TransportError("submit.rejected", str(state.get("error")))
        deadline = self.clock() + self.submit_timeout
        while state["status"] in {"ready", "not_admitted"}:
            pause = max(0.0, state.get("retry_at", 0) - self.clock())
            if self.clock() + pause >= deadline:
                raise H3TransportError("submit.capacity_timeout", "Only 429 responses were received; request remains resumable")
            if pause:
                self.sleep(min(30, pause))
                continue
            self._slot(key)
            attempt = state.get("submit_attempt", 0) + 1
            self._save(directory, state, "submitting", submit_attempt=attempt, submit_started_at=self.clock())
            try:
                response = self._json_request("POST", "/v1/videos", request)
            except (requests.RequestException, OSError, TimeoutError) as error:
                self._save(directory, state, "submit_unknown", error=f"{type(error).__name__}: {error}")
                raise H3TransportError("submit.unknown", "POST outcome is unknown; saved intent must not be automatically resubmitted") from error
            atomic_json(directory / f"submit_response_{attempt:04d}.json", response)
            self._apply_submit_response(directory, state, response)
            if state["status"] == "submit_unknown":
                raise H3TransportError("submit.unknown", state["error"])
            if state["status"] == "rejected":
                self._slot(key, release=True)
                raise H3TransportError("submit.rejected", str(state.get("error")))

    def _poll(self, directory: Path, state: dict) -> dict:
        if state["status"] == "failed":
            raise H3TransportError("task.failed", str(state.get("error")), job_id=state["job_id"])
        if state["status"] in {"server_completed", "completed", "media_invalid"}:
            return state["server_status"]
        job_id = state["job_id"]
        deadline = self.clock() + self.task_timeout
        missing = 0
        failures = 0
        while self.clock() < deadline:
            try:
                response = self._json_request("GET", "/v1/videos/" + quote(job_id, safe=""))
            except (requests.RequestException, OSError, TimeoutError) as error:
                failures += 1
                self._save(directory, state, last_poll_error=f"{type(error).__name__}: {error}")
                self.sleep(min(30, 2 ** min(failures, 5), max(0, deadline - self.clock())))
                continue
            body = response["body"]
            status = response["http_status"]
            if status == 200 and isinstance(body, dict):
                if body.get("id") != job_id:
                    raise H3TransportError("status.id_mismatch", "H3 status returned a different job ID", job_id=job_id)
                missing = failures = 0
                self._save(directory, state, server_status=body)
                if body.get("status") == "failed":
                    self._save(directory, state, "failed", error=body.get("error"))
                    self._slot(directory.name, release=True)
                    raise H3TransportError("task.failed", str(body.get("error")), job_id=job_id)
                if body.get("status") == "completed":
                    self._save(directory, state, "server_completed", server_completed_at=self.clock())
                    self._slot(directory.name, release=True)
                    return body
                if body.get("status") not in {"queued", "in_progress"}:
                    raise H3TransportError("status.unknown", "Unknown H3 job state; retain ID for recovery", job_id=job_id)
                delay = self.poll_interval + random.random()
            elif status == 404:
                missing += 1
                if missing >= 3:
                    raise H3TransportError("status.missing", "Three H3 status lookups returned 404; retain ID, never resubmit", job_id=job_id)
                delay = self.poll_interval
            elif status == 429 or status >= 500:
                failures += 1
                delay = retry_delay(response["headers"], failures, now=self.clock(), jitter=random.random())
            else:
                raise H3TransportError("status.invalid", f"Unexpected H3 status response HTTP {status}", job_id=job_id)
            self.sleep(min(30, delay, max(0, deadline - self.clock())))
        raise H3TransportError("task.timeout", "Polling budget expired; H3 job is not cancelled and must resume with the same ID", job_id=job_id)

    def _download(self, directory: Path, state: dict, request: dict) -> dict:
        destination = directory / "result.mp4"
        if state["status"] == "completed":
            if not destination.is_file() or file_sha256(destination) != state["media"]["sha256"]:
                raise H3TransportError("media.integrity", "Completed H3 output bytes changed", job_id=state["job_id"])
            return state["media"]
        temporary = directory / "result.partial.mp4"
        try:
            with self.http_gate:
                with self._session().get(
                    self.api + "/v1/videos/" + quote(state["job_id"], safe="") + "/content",
                    timeout=(5, 120), stream=True, allow_redirects=False,
                ) as response:
                    response.raise_for_status()
                    if response.status_code != 200:
                        raise H3TransportError("download.invalid", "H3 download did not return HTTP 200", job_id=state["job_id"])
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            stream.write(chunk)
                        stream.flush()
                        os.fsync(stream.fileno())
            with self.media_gate:
                media = self.media_validator(temporary, request)
            os.replace(temporary, destination)
        except (H3TransportError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as error:
            self._save(directory, state, "media_invalid", media_error=f"{type(error).__name__}: {error}")
            raise
        self._save(directory, state, "completed", media=media, output_path=str(destination), completed_at=self.clock())
        return media

    def generate(self, *, generation_key: str, request: dict, lineage: dict | None = None) -> dict:
        if not isinstance(generation_key, str) or not generation_key:
            raise ValueError("A stable generation key is required for durable resume")
        request = json.loads(canonical_json(request))
        H3RenderSettings(
            seconds=request["seconds"], steps=request["num_inference_steps"],
            short_edge=request["target"]["short_edge"], aspect_ratio=request["target"]["aspect_ratio"], seed=request["seed"],
        ).validate()
        if request.get("model") != "MiniMax-H3-Sol" or request.get("task") != "ref2va":
            raise ValueError("H3 R2AV requires MiniMax-H3-Sol/ref2va")
        conditions = request.get("conditions")
        if not isinstance(conditions, list) or not 1 <= len(conditions) <= MAX_REFERENCES:
            raise ValueError("H3 R2AV requires one through twelve image conditions")
        if any(condition.get("type") != "image" or not condition.get("uri", "").startswith("file:///pfs/") for condition in conditions):
            raise ValueError("H3 R2AV inputs must be image conditions in shared /pfs")
        key = sha256_text(generation_key).removeprefix("sha256:")
        directory = self.root / "jobs" / key
        manifest = {"schema_version": VERSION, "api": self.api, "generation_key": generation_key,
                    "request": request, "request_sha256": sha256_json(request), "lineage": lineage or {}}
        with durable_lock(directory / ".lock"):
            manifest_path = directory / "request.json"
            if manifest_path.exists():
                if json.loads(manifest_path.read_text()) != manifest:
                    raise H3TransportError("request.mismatch", "Generation key is already bound to different immutable inputs")
            else:
                atomic_json(manifest_path, manifest)
            state_path = directory / "state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state.get("request_sha256") != manifest["request_sha256"] or state.get("api") != self.api:
                    raise H3TransportError("state.integrity", "H3 job ledger does not match its request manifest")
            else:
                state = {"schema_version": VERSION, "api": self.api, "generation_key": generation_key,
                         "request_sha256": manifest["request_sha256"], "created_at": self.clock()}
                self._save(directory, state, "ready")
            self._submit(directory, state, request)
            server_status = self._poll(directory, state)
            media = self._download(directory, state, request)
            receipt = {"schema_version": VERSION, "backend": "h3", "model": request["model"],
                       "generation_key": generation_key, "h3_job_id": state["job_id"], "api": self.api,
                       "request_sha256": manifest["request_sha256"], "output_path": state["output_path"],
                       "output_sha256": media["sha256"], "media": media, "server_status": server_status,
                       "accepted_at": state["accepted_at"], "completed_at": state["completed_at"],
                       "ledger_path": str(state_path)}
            atomic_json(directory / "receipt.json", receipt)
            return receipt
