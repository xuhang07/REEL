#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip()).strip("_")
    return value or "default"


def _env_int(names: list[str], default: int) -> int:
    for name in names:
        value = os.getenv(name)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _process_exists(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except Exception:
        return False
    if pid_int <= 0:
        return False
    return Path(f"/proc/{pid_int}").exists()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _is_stale(path: Path, *, now: float, stale_seconds: float) -> bool:
    if stale_seconds <= 0:
        return False
    try:
        return now - path.stat().st_mtime > stale_seconds
    except FileNotFoundError:
        return False


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Publish JSON without ever exposing a partially written queue entry."""
    temp_path = path.parent / f".request-{os.getpid()}-{uuid.uuid4().hex}.tmp"
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    fd: int | None = None
    try:
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(fd, "wb") as stream:
            fd = None
            stream.write(payload)
            stream.flush()
        os.replace(temp_path, path)
    finally:
        if fd is not None:
            os.close(fd)
        temp_path.unlink(missing_ok=True)


def _cleanup_stale_queue_temps(queue_dir: Path, *, now: float, stale_seconds: float) -> None:
    """Remove abandoned atomic-write files while leaving active writers alone."""
    for path in queue_dir.glob(".request-*.tmp"):
        match = re.fullmatch(r"\.request-(\d+)-[0-9a-f]+\.tmp", path.name)
        if match:
            should_remove = not _process_exists(match.group(1))
        else:
            should_remove = _is_stale(path, now=now, stale_seconds=stale_seconds)
        if should_remove:
            path.unlink(missing_ok=True)


def _queue_rank(queue_dir: Path, request_path: Path, *, now: float, stale_seconds: float) -> int | None:
    requests: list[tuple[float, float, str, Path]] = []
    for path in queue_dir.glob("*.json"):
        data = _read_json(path)
        if not data:
            # A malformed entry may be a legacy writer that is still publishing.
            # Ignore it for ranking, but only delete it once it is clearly stale.
            if _is_stale(path, now=now, stale_seconds=stale_seconds):
                path.unlink(missing_ok=True)
            continue
        try:
            created_at = float(data.get("created_at") or path.stat().st_mtime)
            priority = float(data.get("priority") or 0.0)
        except (FileNotFoundError, TypeError, ValueError):
            if _is_stale(path, now=now, stale_seconds=stale_seconds):
                path.unlink(missing_ok=True)
            continue
        if stale_seconds > 0 and now - created_at > stale_seconds:
            path.unlink(missing_ok=True)
            continue
        if not _process_exists(data.get("pid")):
            path.unlink(missing_ok=True)
            continue
        requests.append((-priority, created_at, path.name, path))
    requests.sort()
    for rank, (_, _, _, path) in enumerate(requests):
        if path == request_path:
            return rank
    return None


@contextlib.contextmanager
def resource_slot(
    name: str,
    *,
    size: int | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Acquire one cross-process slot for a named external resource.

    The lock is advisory and process-scoped. It is released automatically if the
    process exits, which makes it suitable for long-running subprocess workers.
    Set STORYMEM_<NAME>_POOL_SIZE or STORYMEM_<NAME>_POOL to enable a pool.
    A non-positive size disables limiting.
    """
    key = _safe_name(name).upper()
    pool_size = int(size) if size is not None else _env_int(
        [f"STORYMEM_{key}_POOL_SIZE", f"STORYMEM_{key}_POOL"],
        0,
    )
    if pool_size <= 0:
        if cancel_check is not None:
            cancel_check()
        yield {"enabled": False, "resource": name, "pool_size": pool_size, "wait_sec": 0.0}
        return

    pool_root = Path(os.getenv("STORYMEM_RESOURCE_POOL_DIR", "/tmp/storymem_resource_pools"))
    pool_dir = pool_root / _safe_name(name)
    pool_dir.mkdir(parents=True, exist_ok=True)
    queue_dir = pool_dir / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    priority = _env_float("STORYMEM_RESOURCE_PRIORITY", 0.0)
    priority_label = os.getenv("STORYMEM_RESOURCE_PRIORITY_LABEL", "")
    poll = max(0.02, _env_float("STORYMEM_POOL_POLL_SECONDS", 0.2))
    timeout = _env_float("STORYMEM_POOL_TIMEOUT_SECONDS", 0.0)
    stale_seconds = _env_float("STORYMEM_POOL_STALE_SECONDS", 86400.0)
    log_wait = os.getenv("STORYMEM_POOL_LOG", "").lower() in {"1", "true", "yes"}
    token_paths = [pool_dir / f"slot_{idx:04d}.lock" for idx in range(pool_size)]
    start = time.time()
    last_log = start
    fd: int | None = None
    slot_path: Path | None = None
    request_path = queue_dir / f"{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex}.json"
    request_data = {
        "resource": name,
        "pid": os.getpid(),
        "priority": priority,
        "priority_label": priority_label,
        "created_at": start,
        "created_at_text": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        _cleanup_stale_queue_temps(queue_dir, now=start, stale_seconds=stale_seconds)
        _atomic_write_json(request_path, request_data)
        while True:
            if cancel_check is not None:
                cancel_check()
            now = time.time()
            # Another scanner (including a legacy process) may have removed an
            # entry while it was being published. Restore the same request and
            # ordering identity instead of silently waiting forever.
            if not request_path.exists():
                _atomic_write_json(request_path, request_data)
            rank = _queue_rank(queue_dir, request_path, now=now, stale_seconds=stale_seconds)
            if rank is not None and rank < pool_size:
                random.shuffle(token_paths)
                for path in token_paths:
                    candidate_fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
                    try:
                        fcntl.flock(candidate_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        os.close(candidate_fd)
                        continue
                    fd = candidate_fd
                    slot_path = path
                    wait_sec = time.time() - start
                    request_path.unlink(missing_ok=True)
                    os.ftruncate(fd, 0)
                    os.write(
                        fd,
                        json.dumps(
                            {
                                "resource": name,
                                "pid": os.getpid(),
                                "priority": priority,
                                "priority_label": priority_label,
                                "acquired_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "wait_sec": round(wait_sec, 3),
                            },
                            ensure_ascii=False,
                        ).encode("utf-8"),
                    )
                    if cancel_check is not None:
                        cancel_check()
                    if log_wait and wait_sec >= 1.0:
                        print(
                            json.dumps(
                                {
                                    "event": "resource_pool_acquired",
                                    "resource": name,
                                    "pool_size": pool_size,
                                    "slot": str(path),
                                    "priority": priority,
                                    "priority_label": priority_label,
                                    "wait_sec": round(wait_sec, 3),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    yield {"enabled": True, "resource": name, "pool_size": pool_size, "slot": str(path), "priority": priority, "priority_label": priority_label, "wait_sec": wait_sec}
                    return

            elapsed = time.time() - start
            if timeout > 0 and elapsed >= timeout:
                raise TimeoutError(f"timed out waiting for {name} resource pool slot after {elapsed:.1f}s")
            now = time.time()
            if log_wait and now - last_log >= 30.0:
                print(
                    json.dumps(
                            {
                                "event": "resource_pool_wait",
                                "resource": name,
                                "pool_size": pool_size,
                                "priority": priority,
                                "priority_label": priority_label,
                                "queue_rank": rank,
                                "wait_sec": round(elapsed, 1),
                            },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                last_log = now
            time.sleep(poll + random.uniform(0.0, poll))
    finally:
        request_path.unlink(missing_ok=True)
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
