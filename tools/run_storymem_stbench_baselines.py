#!/usr/bin/env python3
"""Generate reproducible no-memory and last-frame baselines on ST-Bench.

The runner intentionally contains no VLM, policy model, prompt rewriter, or
semantic evaluator.  It sends the normalized ST-Bench prompts to the existing
Echo 1.5 ``/r2v`` service unchanged.  The two supported controls are:

``no_memory``
    Every shot is text-to-video: no condition image and no memory slots.

``last_frame``
    For a StoryMem ``cut=false`` transition, condition the current shot on the
    exact decoded tail frame of the preceding generated shot.  Hard cuts use no
    condition.  Memory slots are always empty.

The canonical run is 30 stories / 300 shots per baseline at 832x480, 121
frames, and 25 fps.  Eight endpoint workers process stories round-robin.  Each
story stays on one endpoint so the paired baselines do not acquire an endpoint
confound.  Successful shot records and deterministic server job IDs make an
interrupted run resumable without blindly submitting a duplicate job.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from echo15_r2v_client import Echo15R2VClient


RUNNER_VERSION = "storymem.stbench.echo15-baselines.v1"
CANONICAL_STORIES = 30
CANONICAL_SHOTS = 300
CANONICAL_WIDTH = 832
CANONICAL_HEIGHT = 480
CANONICAL_FRAMES = 121
CANONICAL_FPS = 25.0
BASELINES = ("no_memory", "last_frame")
DEFAULT_ENDPOINTS = tuple(f"http://127.0.0.1:{port}" for port in range(8321, 8329))
_PRINT_LOCK = threading.Lock()


class BenchmarkContractError(ValueError):
    """Raised when an input or generated artifact violates the benchmark."""


@dataclasses.dataclass(frozen=True)
class StorySpec:
    story_id: str
    title: str
    prompt_path: Path
    prompts: tuple[str, ...]
    cuts: tuple[bool, ...]
    global_shot_start: int
    prompt_file_sha256: str


@dataclasses.dataclass(frozen=True)
class WorkUnit:
    story: StorySpec
    baseline: str
    endpoint: str
    endpoint_index: int


@dataclasses.dataclass(frozen=True)
class RunOptions:
    output_root: Path
    endpoints: tuple[str, ...]
    baselines: tuple[str, ...]
    base_seed: int
    width: int = CANONICAL_WIDTH
    height: int = CANONICAL_HEIGHT
    num_frames: int = CANONICAL_FRAMES
    fps: float = CANONICAL_FPS
    timeout: float = 7200.0
    poll_seconds: float = 5.0
    attempts_per_shot: int = 2
    resume: bool = True
    ffprobe: str = "ffprobe"
    ffmpeg: str = "ffmpeg"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, values: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(canonical_json(value) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json({"time_utc": utc_now(), **event}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkContractError(f"expected a JSON object: {path}")
    return value


def _resolve_record_path(manifest: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    for root in (manifest.parent, *manifest.parents):
        resolved = (root / candidate).resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"cannot resolve benchmark record path {value!r} from {manifest}")


def _flatten_cuts(payload: dict[str, Any], *, story_id: str) -> tuple[bool, ...]:
    metadata = payload.get("benchmark_metadata")
    scenes = metadata.get("raw_scenes") if isinstance(metadata, dict) else None
    if not isinstance(scenes, list) or not scenes:
        raise BenchmarkContractError(f"{story_id}: benchmark_metadata.raw_scenes is missing")
    cuts: list[bool] = []
    for scene_index, scene in enumerate(scenes):
        values = scene.get("cut") if isinstance(scene, dict) else None
        if not isinstance(values, list) or not all(isinstance(item, bool) for item in values):
            raise BenchmarkContractError(f"{story_id}: scene {scene_index} has invalid cut flags")
        cuts.extend(values)
    if not cuts or cuts[0] is not True:
        raise BenchmarkContractError(f"{story_id}: the first shot must be a hard cut")
    return tuple(cuts)


def load_stbench(
    manifest: Path,
    *,
    require_canonical_size: bool = True,
) -> list[StorySpec]:
    manifest = manifest.expanduser().resolve()
    payload = read_json(manifest)
    if payload is None:
        raise FileNotFoundError(manifest)
    records = payload.get("records")
    if not isinstance(records, list):
        raise BenchmarkContractError("manifest.records must be a list")
    if require_canonical_size:
        if payload.get("benchmark") != "storymem_stbench":
            raise BenchmarkContractError("manifest is not normalized StoryMem ST-Bench")
        if payload.get("num_items") != CANONICAL_STORIES or len(records) != CANONICAL_STORIES:
            raise BenchmarkContractError("canonical ST-Bench must contain exactly 30 stories")
        if payload.get("num_shots") != CANONICAL_SHOTS:
            raise BenchmarkContractError("canonical ST-Bench must declare exactly 300 shots")

    stories: list[StorySpec] = []
    seen: set[str] = set()
    global_shot_start = 0
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise BenchmarkContractError(f"manifest record {record_index} is not an object")
        story_id = str(record.get("story_id") or "").strip()
        if not story_id or story_id in seen:
            raise BenchmarkContractError(f"invalid or duplicate story_id: {story_id!r}")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", story_id):
            raise BenchmarkContractError(f"unsafe story_id: {story_id!r}")
        seen.add(story_id)
        prompt_path = _resolve_record_path(manifest, str(record.get("path") or ""))
        story_payload = read_json(prompt_path)
        assert story_payload is not None
        prompts_value = story_payload.get("prompts")
        if (
            not isinstance(prompts_value, list)
            or not prompts_value
            or not all(isinstance(item, str) and item and item == item.strip() for item in prompts_value)
        ):
            raise BenchmarkContractError(f"{story_id}: prompts must be non-blank trimmed strings")
        prompts = tuple(prompts_value)
        cuts = _flatten_cuts(story_payload, story_id=story_id)
        if len(cuts) != len(prompts):
            raise BenchmarkContractError(
                f"{story_id}: {len(cuts)} cut flags do not match {len(prompts)} prompts"
            )
        declared = record.get("num_shots")
        if declared != len(prompts):
            raise BenchmarkContractError(
                f"{story_id}: manifest num_shots={declared!r}, prompt count={len(prompts)}"
            )
        stories.append(
            StorySpec(
                story_id=story_id,
                title=str(story_payload.get("title") or record.get("title") or story_id),
                prompt_path=prompt_path,
                prompts=prompts,
                cuts=cuts,
                global_shot_start=global_shot_start,
                prompt_file_sha256=sha256_file(prompt_path),
            )
        )
        global_shot_start += len(prompts)
    if require_canonical_size and global_shot_start != CANONICAL_SHOTS:
        raise BenchmarkContractError(
            f"canonical ST-Bench resolved to {global_shot_start} shots, expected 300"
        )
    return stories


def normalize_endpoints(values: Iterable[str] | None) -> tuple[str, ...]:
    expanded: list[str] = []
    for value in values or ():
        expanded.extend(part.strip() for part in str(value).split(",") if part.strip())
    if not expanded:
        environment = os.getenv("STORYMEM_ECHO_ENDPOINTS", "")
        expanded = [part.strip() for part in environment.split(",") if part.strip()]
    if not expanded:
        expanded = list(DEFAULT_ENDPOINTS)
    normalized: list[str] = []
    for endpoint in expanded:
        endpoint = endpoint.rstrip("/")
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise BenchmarkContractError(f"invalid Echo endpoint: {endpoint!r}")
        if endpoint in normalized:
            raise BenchmarkContractError(f"duplicate Echo endpoint: {endpoint}")
        normalized.append(endpoint)
    return tuple(normalized)


def build_work_units(
    stories: Sequence[StorySpec],
    baselines: Sequence[str],
    endpoints: Sequence[str],
) -> list[WorkUnit]:
    if not endpoints:
        raise BenchmarkContractError("at least one endpoint is required")
    unknown = set(baselines) - set(BASELINES)
    if unknown:
        raise BenchmarkContractError(f"unknown baselines: {sorted(unknown)}")
    units: list[WorkUnit] = []
    for story_index, story in enumerate(stories):
        endpoint_index = story_index % len(endpoints)
        for baseline in baselines:
            units.append(
                WorkUnit(
                    story=story,
                    baseline=baseline,
                    endpoint=endpoints[endpoint_index],
                    endpoint_index=endpoint_index,
                )
            )
    return units


def build_run_config(
    *,
    manifest: Path,
    stories: Sequence[StorySpec],
    options: RunOptions,
) -> dict[str, Any]:
    work_units = build_work_units(stories, options.baselines, options.endpoints)
    body: dict[str, Any] = {
        "schema_version": RUNNER_VERSION,
        "benchmark": "storymem_stbench",
        "manifest_path": str(manifest.expanduser().resolve()),
        "manifest_sha256": sha256_file(manifest.expanduser().resolve()),
        "story_count": len(stories),
        "shots_per_baseline": sum(len(story.prompts) for story in stories),
        "total_generation_requests": sum(len(unit.story.prompts) for unit in work_units),
        "baselines": list(options.baselines),
        "last_frame_rule": "condition_on_previous_exact_tail_only_when_current_cut_is_false",
        "prompt_policy": "normalized_stbench_prompt_bytes_unchanged",
        "memory_slots": "always_empty",
        "generation": {
            "width": options.width,
            "height": options.height,
            "num_frames": options.num_frames,
            "fps": options.fps,
            "base_seed": options.base_seed,
            "seed_rule": "base_seed_plus_global_shot_index_shared_across_baselines",
        },
        "endpoints": list(options.endpoints),
        "endpoint_assignment_rule": "story_manifest_index_mod_endpoint_count; paired_baselines_share_endpoint",
        "stories": [
            {
                "story_id": story.story_id,
                "title": story.title,
                "num_shots": len(story.prompts),
                "global_shot_start": story.global_shot_start,
                "prompt_path": str(story.prompt_path),
                "prompt_file_sha256": story.prompt_file_sha256,
                "cut_flags": list(story.cuts),
            }
            for story in stories
        ],
        "work_units": [
            {
                "story_id": unit.story.story_id,
                "baseline": unit.baseline,
                "endpoint": unit.endpoint,
                "endpoint_index": unit.endpoint_index,
            }
            for unit in work_units
        ],
    }
    return {**body, "run_fingerprint": sha256_text(canonical_json(body))}


def ensure_run_config(path: Path, config: dict[str, Any]) -> None:
    existing = read_json(path)
    if existing is None:
        atomic_write_json(path, config)
        return
    if existing != config:
        raise BenchmarkContractError(
            f"output directory is sealed to a different run configuration: {path}"
        )


def _fraction(value: Any) -> float:
    text = str(value or "")
    numerator, separator, denominator = text.partition("/")
    try:
        return float(numerator) / float(denominator) if separator else float(numerator)
    except (ValueError, ZeroDivisionError) as exc:
        raise BenchmarkContractError(f"invalid video frame rate: {text!r}") from exc


def probe_video(path: Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    av_error: Exception | None = None
    try:
        import av

        container = av.open(str(path))
        try:
            if len(container.streams.video) != 1:
                raise BenchmarkContractError(
                    f"video has no unique primary video stream: {path}"
                )
            stream = container.streams.video[0]
            fps_value = stream.average_rate or stream.base_rate or stream.guessed_rate
            fps = float(fps_value) if fps_value is not None else 0.0
            frame_count = sum(1 for _ in container.decode(stream))
            return {
                "width": int(stream.width),
                "height": int(stream.height),
                "num_frames": frame_count,
                "fps": fps,
                "probe_backend": "pyav",
                "pyav_version": str(av.__version__),
            }
        finally:
            container.close()
    except ImportError as exc:
        av_error = exc
    except Exception as exc:  # ffprobe can support codecs unavailable to PyAV
        av_error = exc

    executable = shutil.which(ffprobe) or (ffprobe if Path(ffprobe).is_file() else None)
    if executable is None:
        raise FileNotFoundError(
            f"PyAV could not probe {path} ({av_error!r}) and ffprobe was not found: {ffprobe}"
        )
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise BenchmarkContractError(
            f"ffprobe failed for {path}: {(completed.stderr or completed.stdout)[-2000:]}"
        )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or len(streams) != 1:
        raise BenchmarkContractError(f"video has no unique primary video stream: {path}")
    stream = streams[0]
    frame_value = stream.get("nb_read_frames") or stream.get("nb_frames")
    try:
        frame_count = int(frame_value)
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkContractError(f"video metadata is incomplete: {path}") from exc
    fps = _fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    return {
        "width": width,
        "height": height,
        "num_frames": frame_count,
        "fps": fps,
        "probe_backend": "ffprobe",
        "ffprobe_stream": stream,
    }


def validate_video_contract(metadata: dict[str, Any], options: RunOptions) -> None:
    expected = (options.width, options.height, options.num_frames)
    actual = (
        metadata.get("width"),
        metadata.get("height"),
        metadata.get("num_frames"),
    )
    if actual != expected:
        raise BenchmarkContractError(
            f"generated video shape/frame contract mismatch: actual={actual}, expected={expected}"
        )
    if abs(float(metadata.get("fps", 0.0)) - options.fps) > 1e-3:
        raise BenchmarkContractError(
            f"generated video fps={metadata.get('fps')!r}, expected={options.fps}"
        )


def extract_tail_frame(
    video: Path,
    destination: Path,
    *,
    frame_count: int,
    ffmpeg: str = "ffmpeg",
) -> None:
    executable = shutil.which(ffmpeg) or (ffmpeg if Path(ffmpeg).is_file() else None)
    if executable is None:
        try:
            import imageio_ffmpeg

            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            executable = None
    if executable is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {ffmpeg}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.tmp.png")
    try:
        completed = subprocess.run(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video),
                "-vf",
                f"select=eq(n\\,{frame_count - 1})",
                "-frames:v",
                "1",
                "-an",
                "-update",
                "1",
                str(temporary),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
            raise BenchmarkContractError(
                f"tail-frame extraction failed for {video}: {(completed.stderr or completed.stdout)[-2000:]}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_symlink(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.symlink_to(source)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _shot_dir(options: RunOptions, baseline: str, story_id: str, shot_index: int) -> Path:
    return options.output_root / baseline / story_id / "shots" / f"shot_{shot_index:03d}"


def _shot_seed(options: RunOptions, story: StorySpec, shot_index: int) -> int:
    return options.base_seed + story.global_shot_start + shot_index


def _load_success(path: Path, *, resume: bool) -> dict[str, Any] | None:
    record = read_json(path)
    if record is None:
        return None
    if not resume:
        raise BenchmarkContractError(f"shot result already exists and --no-resume was used: {path}")
    if record.get("status") != "succeeded":
        return None
    video = Path(str((record.get("output") or {}).get("video_path") or ""))
    if not video.is_file():
        return None
    return record


def _condition_for_shot(
    unit: WorkUnit,
    shot_index: int,
    options: RunOptions,
    *,
    tail_extractor: Callable[..., None],
) -> tuple[Path | None, str | None]:
    if unit.baseline != "last_frame" or unit.story.cuts[shot_index]:
        return None, None
    previous_dir = _shot_dir(options, unit.baseline, unit.story.story_id, shot_index - 1)
    previous = read_json(previous_dir / "result.json")
    if previous is None or previous.get("status") != "succeeded":
        raise BenchmarkContractError("previous shot is unavailable for cut=false continuation")
    video_path = Path(str((previous.get("output") or {}).get("video_path") or ""))
    if not video_path.is_file():
        raise BenchmarkContractError("previous shot video is unavailable for cut=false continuation")
    tail_path = previous_dir / "tail.png"
    if not tail_path.is_file():
        tail_extractor(
            video_path,
            tail_path,
            frame_count=options.num_frames,
            ffmpeg=options.ffmpeg,
        )
    return tail_path.resolve(), sha256_file(tail_path)


def _shot_fingerprint(
    *,
    unit: WorkUnit,
    shot_index: int,
    prompt: str,
    seed: int,
    condition_sha256: str | None,
    options: RunOptions,
) -> str:
    return sha256_text(
        canonical_json(
            {
                "runner_version": RUNNER_VERSION,
                "story_id": unit.story.story_id,
                "baseline": unit.baseline,
                "shot_index": shot_index,
                "prompt_sha256": sha256_text(prompt),
                "scene_cut": unit.story.cuts[shot_index],
                "condition_sha256": condition_sha256,
                "memory_slots": [],
                "seed": seed,
                "width": options.width,
                "height": options.height,
                "num_frames": options.num_frames,
                "fps": options.fps,
            }
        )
    )


def _next_attempt(
    path: Path,
    *,
    shot_fingerprint: str,
    endpoint: str,
) -> tuple[int, str, bool]:
    prior = read_json(path)
    if (
        prior is not None
        and prior.get("shot_fingerprint") == shot_fingerprint
        and prior.get("endpoint") == endpoint
        and prior.get("status") in {"inflight", "unknown"}
    ):
        return int(prior["attempt_index"]), str(prior["job_id"]), True
    attempt_index = int((prior or {}).get("attempt_index", -1)) + 1
    identity = f"{shot_fingerprint}:{attempt_index}"
    job_id = "stbench-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:40]
    return attempt_index, job_id, False


def _provider_receipt(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in ("version_id", "task_id", "video_url", "metadata")
        if result.get(key) is not None
    }


def _log_line(value: dict[str, Any]) -> None:
    with _PRINT_LOCK:
        print(canonical_json(value), flush=True)


def run_shot(
    client: Any,
    unit: WorkUnit,
    shot_index: int,
    options: RunOptions,
    *,
    run_fingerprint: str,
    media_probe: Callable[..., dict[str, Any]] = probe_video,
    tail_extractor: Callable[..., None] = extract_tail_frame,
) -> dict[str, Any]:
    shot_dir = _shot_dir(options, unit.baseline, unit.story.story_id, shot_index)
    result_path = shot_dir / "result.json"
    event_path = shot_dir / "events.jsonl"
    prior_success = _load_success(result_path, resume=options.resume)
    prompt = unit.story.prompts[shot_index]
    seed = _shot_seed(options, unit.story, shot_index)
    if prior_success is not None:
        if (
            prior_success.get("run_fingerprint") != run_fingerprint
            or prior_success.get("baseline") != unit.baseline
            or prior_success.get("story_id") != unit.story.story_id
            or prior_success.get("shot_index") != shot_index
        ):
            raise BenchmarkContractError(
                f"successful shot is sealed to different run inputs: {result_path}"
            )
        append_event(
            event_path,
            {
                "event": "resume_skip",
                "shot_fingerprint": prior_success.get("shot_fingerprint"),
            },
        )
        _log_line(
            {
                "event": "resume_skip",
                "baseline": unit.baseline,
                "story_id": unit.story.story_id,
                "shot_index": shot_index,
            }
        )
        return prior_success

    try:
        condition_path, condition_sha256 = _condition_for_shot(
            unit,
            shot_index,
            options,
            tail_extractor=tail_extractor,
        )
    except BenchmarkContractError as exc:
        blocked = {
            "schema_version": RUNNER_VERSION,
            "status": "blocked_dependency",
            "baseline": unit.baseline,
            "story_id": unit.story.story_id,
            "shot_index": shot_index,
            "scene_cut": unit.story.cuts[shot_index],
            "endpoint": unit.endpoint,
            "seed": seed,
            "error": str(exc),
            "updated_at_utc": utc_now(),
        }
        atomic_write_json(result_path, blocked)
        append_event(event_path, {"event": "blocked_dependency", "error": str(exc)})
        _log_line({"event": "blocked", "baseline": unit.baseline, "story_id": unit.story.story_id, "shot_index": shot_index})
        return blocked

    shot_fingerprint = _shot_fingerprint(
        unit=unit,
        shot_index=shot_index,
        prompt=prompt,
        seed=seed,
        condition_sha256=condition_sha256,
        options=options,
    )

    attempt_path = shot_dir / "attempt.json"
    last_error: Exception | None = None
    for local_attempt in range(options.attempts_per_shot):
        attempt_index, job_id, reused_job = _next_attempt(
            attempt_path,
            shot_fingerprint=shot_fingerprint,
            endpoint=unit.endpoint,
        )
        attempt = {
            "schema_version": RUNNER_VERSION,
            "status": "inflight",
            "attempt_index": attempt_index,
            "job_id": job_id,
            "reused_job": reused_job,
            "shot_fingerprint": shot_fingerprint,
            "endpoint": unit.endpoint,
            "started_at_utc": utc_now(),
        }
        atomic_write_json(attempt_path, attempt)
        request = {
            "work_id": f"stbench-{run_fingerprint[7:19]}-{unit.baseline}-{unit.story.story_id}",
            "shot_id": f"shot_{shot_index:03d}",
            "job_id": job_id,
            "prompt": prompt,
            "condition_img": str(condition_path) if condition_path else None,
            "memory_slots": [],
            "seed": seed,
            "width": options.width,
            "height": options.height,
            "num_frames": options.num_frames,
            "expected_fps": options.fps,
        }
        append_event(
            event_path,
            {
                "event": "generation_start",
                "attempt_index": attempt_index,
                "job_id": job_id,
                "reused_job": reused_job,
                "request": request,
            },
        )
        started = time.monotonic()
        try:
            result = client.generate_one(
                work_id=request["work_id"],
                shot_id=request["shot_id"],
                job_id=job_id,
                prompt=prompt,
                memory_paths=[],
                condition_image_path=str(condition_path) if condition_path else None,
                reference_shot_ids=[],
                seed=seed,
                width=options.width,
                height=options.height,
                num_frames=options.num_frames,
                output_dir=shot_dir / "download",
            )
            source_video = Path(str(result.get("output_path") or result.get("video") or ""))
            if not source_video.is_file():
                raise BenchmarkContractError("Echo response does not resolve to a local video")
            media = media_probe(source_video, ffprobe=options.ffprobe)
            validate_video_contract(media, options)
            video_path = shot_dir / "video.mp4"
            atomic_symlink(source_video, video_path)
            video_sha256 = sha256_file(video_path)
            tail: dict[str, Any] | None = None
            if unit.baseline == "last_frame":
                tail_path = shot_dir / "tail.png"
                tail_extractor(
                    video_path,
                    tail_path,
                    frame_count=options.num_frames,
                    ffmpeg=options.ffmpeg,
                )
                tail = {"path": str(tail_path.resolve()), "sha256": sha256_file(tail_path)}
            elapsed = time.monotonic() - started
            receipt = _provider_receipt(result)
            record = {
                "schema_version": RUNNER_VERSION,
                "status": "succeeded",
                "run_fingerprint": run_fingerprint,
                "shot_fingerprint": shot_fingerprint,
                "baseline": unit.baseline,
                "story_id": unit.story.story_id,
                "title": unit.story.title,
                "shot_index": shot_index,
                "shot_number": shot_index + 1,
                "global_shot_index": unit.story.global_shot_start + shot_index,
                "scene_cut": unit.story.cuts[shot_index],
                "endpoint": unit.endpoint,
                "endpoint_index": unit.endpoint_index,
                "seed": seed,
                "request": request,
                "trajectory": [
                    {
                        "step": 1,
                        "action": "generate_shot",
                        "arguments": request,
                        "observation": {"status": "succeeded", **receipt},
                    }
                ],
                "protocol_metrics": {
                    "prompt_bytes_preserved": True,
                    "memory_slots_empty": True,
                    "condition_policy_correct": (
                        condition_path is not None
                        if unit.baseline == "last_frame" and not unit.story.cuts[shot_index]
                        else condition_path is None
                    ),
                    "media_contract_valid": True,
                },
                "input": {
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "condition_path": str(condition_path) if condition_path else None,
                    "condition_sha256": condition_sha256,
                },
                "output": {
                    "video_path": str(video_path.absolute()),
                    "provider_video_path": str(source_video.resolve()),
                    "video_sha256": video_sha256,
                    "tail_frame": tail,
                    "media": media,
                    "provider_receipt": receipt,
                },
                "attempt_index": attempt_index,
                "elapsed_seconds": round(elapsed, 3),
                "completed_at_utc": utc_now(),
            }
            atomic_write_json(result_path, record)
            atomic_write_json(
                attempt_path,
                {**attempt, "status": "succeeded", "completed_at_utc": utc_now()},
            )
            append_event(
                event_path,
                {
                    "event": "generation_succeeded",
                    "attempt_index": attempt_index,
                    "job_id": job_id,
                    "video_sha256": video_sha256,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
            _log_line(
                {
                    "event": "shot_succeeded",
                    "baseline": unit.baseline,
                    "story_id": unit.story.story_id,
                    "shot_index": shot_index,
                    "endpoint_index": unit.endpoint_index,
                    "conditioned": condition_path is not None,
                    "elapsed_seconds": round(elapsed, 3),
                }
            )
            return record
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # keep other stories moving; persist exact failure
            last_error = exc
            uncertain = isinstance(exc, (TimeoutError, requests.RequestException))
            attempt_status = "unknown" if uncertain else "failed"
            atomic_write_json(
                attempt_path,
                {
                    **attempt,
                    "status": attempt_status,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_at_utc": utc_now(),
                },
            )
            append_event(
                event_path,
                {
                    "event": "generation_error",
                    "attempt_index": attempt_index,
                    "job_id": job_id,
                    "uncertain_server_state": uncertain,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            _log_line(
                {
                    "event": "shot_attempt_failed",
                    "baseline": unit.baseline,
                    "story_id": unit.story.story_id,
                    "shot_index": shot_index,
                    "attempt": local_attempt + 1,
                    "uncertain_server_state": uncertain,
                    "error": str(exc),
                }
            )
            if local_attempt + 1 < options.attempts_per_shot:
                continue

    assert last_error is not None
    attempt_state = read_json(attempt_path) or {}
    status = "pending_recovery" if attempt_state.get("status") == "unknown" else "failed"
    failed = {
        "schema_version": RUNNER_VERSION,
        "status": status,
        "run_fingerprint": run_fingerprint,
        "shot_fingerprint": shot_fingerprint,
        "baseline": unit.baseline,
        "story_id": unit.story.story_id,
        "shot_index": shot_index,
        "scene_cut": unit.story.cuts[shot_index],
        "endpoint": unit.endpoint,
        "seed": seed,
        "request": request,
        "error_type": type(last_error).__name__,
        "error": str(last_error),
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(result_path, failed)
    return failed


def run_work_unit(
    client: Any,
    unit: WorkUnit,
    options: RunOptions,
    *,
    run_fingerprint: str,
    media_probe: Callable[..., dict[str, Any]] = probe_video,
    tail_extractor: Callable[..., None] = extract_tail_frame,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for shot_index in range(len(unit.story.prompts)):
        records.append(
            run_shot(
                client,
                unit,
                shot_index,
                options,
                run_fingerprint=run_fingerprint,
                media_probe=media_probe,
                tail_extractor=tail_extractor,
            )
        )
    return records


def _default_client_factory(endpoint: str, options: RunOptions) -> Echo15R2VClient:
    return Echo15R2VClient(
        endpoint,
        timeout=options.timeout,
        poll_seconds=options.poll_seconds,
        max_memory_slots=7,
    )


def preflight_endpoints(
    endpoints: Sequence[str],
    options: RunOptions,
    *,
    client_factory: Callable[[str, RunOptions], Any],
) -> list[dict[str, Any]]:
    def check(index_endpoint: tuple[int, str]) -> dict[str, Any]:
        index, endpoint = index_endpoint
        started = time.monotonic()
        health = client_factory(endpoint, options).health()
        if not isinstance(health, dict):
            raise BenchmarkContractError(f"{endpoint} returned a non-object health response")
        explicit_ok = health.get("status") == "ok" or health.get("ok") is True
        if not explicit_ok:
            raise BenchmarkContractError(f"{endpoint} is not healthy: {health}")
        return {
            "endpoint_index": index,
            "endpoint": endpoint,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "status": "ok",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
        futures = [executor.submit(check, pair) for pair in enumerate(endpoints)]
        return [future.result() for future in futures]


def collect_results(options: RunOptions, stories: Sequence[StorySpec]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for baseline in options.baselines:
        for story in stories:
            for shot_index in range(len(story.prompts)):
                path = _shot_dir(options, baseline, story.story_id, shot_index) / "result.json"
                record = read_json(path)
                if record is None:
                    record = {
                        "schema_version": RUNNER_VERSION,
                        "status": "missing",
                        "baseline": baseline,
                        "story_id": story.story_id,
                        "shot_index": shot_index,
                    }
                values.append(record)
    return values


def summarize_results(
    records: Sequence[dict[str, Any]],
    *,
    options: RunOptions,
    stories: Sequence[StorySpec],
    worker_errors: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    by_baseline: dict[str, dict[str, Any]] = {}
    for baseline in options.baselines:
        rows = [row for row in records if row.get("baseline") == baseline]
        counts: dict[str, int] = {}
        for row in rows:
            status = str(row.get("status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
        by_baseline[baseline] = {
            "expected": sum(len(story.prompts) for story in stories),
            "status_counts": counts,
            "succeeded": counts.get("succeeded", 0),
            "conditioned_shots": sum(
                bool((row.get("input") or {}).get("condition_path"))
                for row in rows
                if row.get("status") == "succeeded"
            ),
            "protocol_valid": sum(
                all((row.get("protocol_metrics") or {}).values())
                for row in rows
                if row.get("status") == "succeeded"
            ),
        }
    expected_total = sum(item["expected"] for item in by_baseline.values())
    succeeded_total = sum(item["succeeded"] for item in by_baseline.values())
    return {
        "schema_version": RUNNER_VERSION,
        "updated_at_utc": utc_now(),
        "story_count": len(stories),
        "shots_per_baseline": sum(len(story.prompts) for story in stories),
        "expected_total": expected_total,
        "succeeded_total": succeeded_total,
        "complete": succeeded_total == expected_total and not worker_errors,
        "by_baseline": by_baseline,
        "worker_errors": list(worker_errors),
    }


@contextlib.contextmanager
def exclusive_run_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another benchmark runner owns {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def execute_benchmark(
    *,
    manifest: Path,
    stories: Sequence[StorySpec],
    options: RunOptions,
    plan_only: bool = False,
    skip_health_check: bool = False,
    client_factory: Callable[[str, RunOptions], Any] = _default_client_factory,
    media_probe: Callable[..., dict[str, Any]] = probe_video,
    tail_extractor: Callable[..., None] = extract_tail_frame,
) -> dict[str, Any]:
    if options.width != CANONICAL_WIDTH or options.height != CANONICAL_HEIGHT:
        raise BenchmarkContractError("ST-Bench baseline resolution is fixed at 832x480")
    if options.num_frames != CANONICAL_FRAMES or options.fps != CANONICAL_FPS:
        raise BenchmarkContractError("ST-Bench baseline duration is fixed at 121 frames / 25 fps")
    if options.attempts_per_shot < 1:
        raise BenchmarkContractError("attempts_per_shot must be positive")
    options.output_root.mkdir(parents=True, exist_ok=True)
    config = build_run_config(manifest=manifest, stories=stories, options=options)
    ensure_run_config(options.output_root / "run_config.json", config)
    if plan_only:
        return {
            "plan_only": True,
            "run_fingerprint": config["run_fingerprint"],
            "story_count": config["story_count"],
            "shots_per_baseline": config["shots_per_baseline"],
            "total_generation_requests": config["total_generation_requests"],
            "endpoint_count": len(options.endpoints),
            "continuity_transitions": sum(
                not cut for story in stories for cut in story.cuts
            ),
            "run_config": str((options.output_root / "run_config.json").resolve()),
        }

    with exclusive_run_lock(options.output_root / ".runner.lock"):
        health = (
            []
            if skip_health_check
            else preflight_endpoints(
                options.endpoints,
                options,
                client_factory=client_factory,
            )
        )
        if health:
            atomic_write_json(options.output_root / "endpoint_health.json", health)
        units = build_work_units(stories, options.baselines, options.endpoints)
        buckets: list[list[WorkUnit]] = [[] for _ in options.endpoints]
        for unit in units:
            buckets[unit.endpoint_index].append(unit)

        def run_bucket(endpoint_index: int) -> list[dict[str, Any]]:
            client = client_factory(options.endpoints[endpoint_index], options)
            rows: list[dict[str, Any]] = []
            for unit in buckets[endpoint_index]:
                rows.extend(
                    run_work_unit(
                        client,
                        unit,
                        options,
                        run_fingerprint=config["run_fingerprint"],
                        media_probe=media_probe,
                        tail_extractor=tail_extractor,
                    )
                )
            return rows

        worker_errors: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(options.endpoints)) as executor:
            future_map = {
                executor.submit(run_bucket, index): index
                for index, bucket in enumerate(buckets)
                if bucket
            }
            for future in concurrent.futures.as_completed(future_map):
                endpoint_index = future_map[future]
                try:
                    future.result()
                except Exception as exc:  # preserve other endpoint workers
                    error = {
                        "endpoint_index": endpoint_index,
                        "endpoint": options.endpoints[endpoint_index],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    worker_errors.append(error)
                    _log_line({"event": "endpoint_worker_failed", **error})

        records = collect_results(options, stories)
        summary = summarize_results(
            records,
            options=options,
            stories=stories,
            worker_errors=worker_errors,
        )
        atomic_write_jsonl(options.output_root / "results.jsonl", records)
        atomic_write_json(options.output_root / "summary.json", summary)
        return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "benchmarks/external/normalized/storymem_stbench/manifest.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "inference_result/storymem_stbench_baselines_832x480x121",
    )
    parser.add_argument("--baseline", action="append", choices=BASELINES)
    parser.add_argument("--story", action="append", help="Run only this story ID; repeatable")
    parser.add_argument("--endpoint", action="append", help="Echo base URL; repeatable or comma-separated")
    parser.add_argument("--expected-endpoints", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--attempts-per-shot", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--skip-health-check", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--require-canonical-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the official normalized 30-story/300-shot manifest",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    endpoints = normalize_endpoints(args.endpoint)
    if args.expected_endpoints > 0 and len(endpoints) != args.expected_endpoints:
        raise BenchmarkContractError(
            f"expected {args.expected_endpoints} unique endpoints, received {len(endpoints)}"
        )
    stories = load_stbench(
        args.manifest,
        require_canonical_size=args.require_canonical_size,
    )
    if args.story:
        requested = set(args.story)
        known = {story.story_id for story in stories}
        unknown = requested - known
        if unknown:
            raise BenchmarkContractError(f"unknown story IDs: {sorted(unknown)}")
        stories = [story for story in stories if story.story_id in requested]
    baselines = tuple(dict.fromkeys(args.baseline or BASELINES))
    options = RunOptions(
        output_root=args.output_root.expanduser().resolve(),
        endpoints=endpoints,
        baselines=baselines,
        base_seed=args.base_seed,
        attempts_per_shot=args.attempts_per_shot,
        resume=args.resume,
        timeout=args.timeout,
        poll_seconds=args.poll_seconds,
        ffprobe=args.ffprobe,
        ffmpeg=args.ffmpeg,
    )
    summary = execute_benchmark(
        manifest=args.manifest,
        stories=stories,
        options=options,
        plan_only=args.plan_only,
        skip_health_check=args.skip_health_check,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if args.plan_only or summary.get("complete") else 1


if __name__ == "__main__":
    sys.exit(main())
