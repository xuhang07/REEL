from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse
from urllib.request import urlretrieve

import requests


_SUCCEEDED = {"succeeded", "completed", "done", "success"}
_FAILED = {"failed", "error", "cancelled", "canceled"}


def _raise_for_status_with_body(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:2000] if response.text else ""
        raise requests.HTTPError(f"{exc}; body={body}", response=response) from exc


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _absolute_existing_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    for item in paths:
        path = Path(str(item)).expanduser()
        if path.is_file():
            out.append(str(path.resolve()))
    return out


def _concat_videos(output_path: Path, video_paths: list[str]) -> str | None:
    if not video_paths:
        return None
    if len(video_paths) == 1:
        return video_paths[0]
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return video_paths[-1]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_path.with_suffix(".txt")
    list_path.write_text(
        "".join(f"file '{Path(path).resolve()}'\n" for path in video_paths),
        encoding="utf-8",
    )
    for command in (
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(output_path)],
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and output_path.is_file():
            return str(output_path.resolve())
    return video_paths[-1]


class Echo15R2VClient:
    """Small synchronous client for the Echo 1.5 `/r2v` server.

    The migrated StoryMem training tools are synchronous: a tool call must return
    a finished video path. Echo 1.5's server is asynchronous, so this client
    submits an R2V job and polls `/version/{version_id}` until completion. In the
    StoryMem path, saved memory-bank images are passed as `memory_slots`, while a
    separately selected external keyframe is passed as `condition_img`.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: int | float | None = None,
        poll_seconds: int | float | None = None,
        max_memory_slots: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout if timeout is not None else os.getenv("STORYMEM_ROLLOUT_TIMEOUT_SECONDS", "7200"))
        self.poll_seconds = float(poll_seconds if poll_seconds is not None else os.getenv("ECHO15_R2V_POLL_SECONDS", "5"))
        self.max_memory_slots = int(max_memory_slots if max_memory_slots is not None else os.getenv("ECHO15_MAX_MEMORY_SLOTS", "7"))

    @classmethod
    def from_env(cls, fallback_url: str | None = None, timeout: int | float | None = None) -> "Echo15R2VClient":
        url = (
            os.getenv("ECHO15_R2V_SERVER_URL")
            or os.getenv("ECHO_R2V_SERVER_URL")
            or fallback_url
            or os.getenv("ROLLOUT_SERVER_URL")
            or ""
        ).rstrip("/")
        if not url:
            raise RuntimeError("ECHO15_R2V_SERVER_URL or ROLLOUT_SERVER_URL is required")
        return cls(url, timeout=timeout)

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=30)
        response.raise_for_status()
        return response.json()

    def rollout(
        self,
        *,
        prompts: list[str],
        initial_memory_paths: list[str],
        condition_image_path: str | None = None,
        seed: int,
        run_name: str,
        output_root: str,
        width: int | None = None,
        height: int | None = None,
        num_frames: int | None = None,
    ) -> dict[str, Any]:
        if not prompts:
            raise ValueError("prompts must be a non-empty list")
        if os.getenv("ECHO15_FORCE_LEGACY_ROLLOUT", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
            return self._legacy_rollout(
                prompts=prompts,
                initial_memory_paths=initial_memory_paths,
                condition_image_path=condition_image_path,
                seed=seed,
                run_name=run_name,
                output_root=output_root,
            )

        try:
            return self._r2v_rollout(
                prompts=prompts,
                initial_memory_paths=initial_memory_paths,
                condition_image_path=condition_image_path,
                seed=seed,
                run_name=run_name,
                output_root=output_root,
                width=width,
                height=height,
                num_frames=num_frames,
            )
        except requests.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            if status_code != 404:
                raise
            return self._legacy_rollout(
                prompts=prompts,
                initial_memory_paths=initial_memory_paths,
                condition_image_path=condition_image_path,
                seed=seed,
                run_name=run_name,
                output_root=output_root,
            )

    def _legacy_rollout(
        self,
        *,
        prompts: list[str],
        initial_memory_paths: list[str],
        condition_image_path: str | None,
        seed: int,
        run_name: str,
        output_root: str,
    ) -> dict[str, Any]:
        payload = {
            "request_id": uuid.uuid4().hex,
            "run_name": run_name,
            "prompts": prompts,
            "initial_memory_paths": initial_memory_paths,
            "seed": seed,
            "output_root": output_root,
        }
        response = requests.post(f"{self.base_url}/rollout", json=payload, timeout=self.timeout)
        _raise_for_status_with_body(response)
        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(data)
        return data

    def _r2v_rollout(
        self,
        *,
        prompts: list[str],
        initial_memory_paths: list[str],
        condition_image_path: str | None,
        seed: int,
        run_name: str,
        output_root: str,
        width: int | None,
        height: int | None,
        num_frames: int | None,
    ) -> dict[str, Any]:
        output_dir = Path(output_root).expanduser().resolve() / "server_runs" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        work_id = self._safe_id(run_name)
        memory_paths = _absolute_existing_paths(initial_memory_paths)
        condition_path = _absolute_existing_paths([condition_image_path or ""])
        first_condition_image = condition_path[0] if condition_path else None
        version_ids: list[str] = []
        video_paths: list[str] = []
        shot_results: list[dict[str, Any]] = []
        for shot_index, prompt in enumerate(prompts, start=1):
            result = self.generate_one(
                work_id=work_id,
                shot_id=f"shot_{shot_index:03d}",
                prompt=prompt,
                memory_paths=memory_paths,
                condition_image_path=first_condition_image if shot_index == 1 else None,
                reference_shot_ids=[],
                seed=int(seed) + shot_index - 1,
                width=width,
                height=height,
                num_frames=num_frames,
                output_dir=output_dir / f"shot_{shot_index:03d}",
                job_id=f"{work_id}_shot_{shot_index:03d}_{uuid.uuid4().hex[:8]}",
            )
            shot_results.append(result)
            if result.get("version_id"):
                version_ids.append(str(result["version_id"]))
            video_path = str(result.get("output_path") or result.get("video") or "")
            if video_path:
                video_paths.append(video_path)
        combined = _concat_videos(output_dir / "combined.mp4", video_paths)
        return {
            "ok": bool(video_paths),
            "request_id": uuid.uuid4().hex,
            "run_name": run_name,
            "combined_video": combined or (video_paths[-1] if video_paths else ""),
            "video": video_paths[-1] if video_paths else "",
            "shot_results": shot_results,
            "version_ids": version_ids,
        }

    def generate_one(
        self,
        *,
        work_id: str,
        shot_id: str,
        prompt: str,
        memory_paths: list[str],
        condition_image_path: str | None = None,
        reference_shot_ids: list[str] | None = None,
        seed: int,
        output_dir: Path,
        width: int | None = None,
        height: int | None = None,
        num_frames: int | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        memory_slots = self._memory_slots(memory_paths, reference_shot_ids or [])
        payload: dict[str, Any] = {
            "work_id": work_id,
            "shot_id": shot_id,
            "job_id": job_id or uuid.uuid4().hex,
            "prompt": prompt,
            "memory_slots": memory_slots,
            "seed": seed,
        }
        condition_paths = _absolute_existing_paths([condition_image_path or ""])
        if condition_paths:
            payload["condition_img"] = condition_paths[0]
        if width is not None:
            payload["width"] = int(width)
        if height is not None:
            payload["height"] = int(height)
        if num_frames is not None:
            payload["num_frames"] = int(num_frames)
        response = requests.post(f"{self.base_url}/r2v", json=payload, timeout=min(self.timeout, 120))
        _raise_for_status_with_body(response)
        accepted = response.json()
        final = self._wait_for_completion(accepted)
        status = str(final.get("status") or "").lower()
        if status not in _SUCCEEDED:
            raise RuntimeError(final.get("error") or final)
        result = final.get("result") if isinstance(final.get("result"), dict) else {}
        output_path = result.get("output_path") or final.get("output_path")
        video_url = result.get("video_url") or final.get("video_url") or result.get("artifact_url") or final.get("artifact_url")
        if isinstance(video_url, str) and video_url:
            video_url = self._rewrite_loopback_url(video_url)
        local_video = self._local_or_downloaded_video(output_path, video_url, output_dir)
        return {
            "ok": True,
            "version_id": final.get("version_id") or accepted.get("version_id") or accepted.get("task_id"),
            "task_id": final.get("task_id") or accepted.get("task_id"),
            "video": local_video,
            "combined_video": local_video,
            "video_url": video_url,
            "output_path": local_video,
            "condition_img": payload.get("condition_img"),
            "memory_slots": memory_slots,
            "metadata": result.get("metadata") or {},
            "raw": final,
        }

    def _memory_slots(self, memory_paths: list[str], reference_shot_ids: list[str]) -> list[dict[str, Any]]:
        slots: list[dict[str, Any]] = []
        for path in _absolute_existing_paths(memory_paths):
            slots.append({"image_url": path, "audio_mode": "empty", "metadata": {"source": "storymem_image"}})
        for shot_id in reference_shot_ids:
            slots.append({"shot_id": str(shot_id), "metadata": {"source": "previous_r2v_shot"}})
        # Callers order slots from most important to least important.  Preserve
        # that priority if a generic caller exceeds the server limit.
        return slots[: self.max_memory_slots]

    def _wait_for_completion(self, accepted: dict[str, Any]) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        status_url = str(accepted.get("status_url") or "")
        version_id = str(accepted.get("version_id") or accepted.get("task_id") or "")
        if status_url:
            status_url = self._rewrite_loopback_url(status_url)
        if not status_url:
            if not version_id:
                raise RuntimeError(f"Echo 1.5 response has no status_url/version_id: {accepted}")
            status_url = f"{self.base_url}/version/{quote(version_id, safe='')}"
        last: dict[str, Any] = accepted
        while True:
            status = str(last.get("status") or "").lower()
            if status in _SUCCEEDED | _FAILED:
                return last
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Echo 1.5 R2V job timed out: {version_id or status_url}")
            time.sleep(max(self.poll_seconds, 0.1))
            response = requests.get(status_url, timeout=min(self.timeout, 120))
            _raise_for_status_with_body(response)
            last = response.json()

    def _rewrite_loopback_url(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "0.0.0.0"}:
            return url
        base = urlparse(self.base_url)
        if not base.scheme or not base.netloc:
            return url
        return parsed._replace(scheme=base.scheme, netloc=base.netloc).geturl()

    @staticmethod
    def _safe_id(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))
        return cleaned.strip("_")[:96] or f"work_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _local_or_downloaded_video(output_path: Any, video_url: Any, output_dir: Path) -> str:
        if output_path:
            path = Path(str(output_path)).expanduser()
            if path.is_file():
                return str(path.resolve())
        if isinstance(video_url, str) and video_url:
            parsed = urlparse(video_url)
            if parsed.scheme in {"http", "https"}:
                output_dir.mkdir(parents=True, exist_ok=True)
                target = output_dir / "result.mp4"
                urlretrieve(video_url, target)
                return str(target.resolve())
            path = Path(video_url).expanduser()
            if path.is_file():
                return str(path.resolve())
        return ""
