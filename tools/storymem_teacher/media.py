"""Deterministic media transport helpers for the canonical StoryMem teacher.

This module never asks a model to caption, rewrite, or inspect media.  It only
identifies bytes, transports those exact bytes to an API, and derives explicit
ffmpeg representations whose lineage can be recorded by the runtime.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import fcntl
import functools
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import time
from math import ceil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlparse

import requests

from .util import sha256_bytes, sha256_file


_MIME_BY_SUFFIX = {
    ".avif": "image/avif",
    ".flac": "audio/flac",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".m4a": "audio/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".oga": "audio/ogg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
    ".wav": "audio/wav",
    ".webm": "video/webm",
    ".webp": "image/webp",
}


_DERIVATION_CACHE_VERSION = "storymem.media-derivation-cache.v1"
_VISUAL_ONLY_OPERATION_VERSION = "storymem.visual-only-remux.v1"
_BOUNDARY_FRAME_OPERATION_VERSION = "storymem.boundary-frame-extract.v1"
_DERIVATION_CACHE_DIRNAME = ".storymem_media_cache_v1"


@dataclass(frozen=True)
class MediaDescriptor:
    path: str
    mime_type: str
    size_bytes: int
    sha256: str
    probe: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MediaInput:
    """Provider-neutral user-content item accepted by ``RemoteTeacherClient``."""

    path: str | Path
    mime_type: str | None = None
    detail: str | None = None


def require_file(path: str | Path, *, label: str = "media") -> Path:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {source}")
    if source.stat().st_size <= 0:
        raise ValueError(f"{label} file is empty: {source}")
    return source.resolve()


def _sniff_mime(header: bytes) -> str | None:
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return "audio/wav"
    if header.startswith(b"fLaC"):
        return "audio/flac"
    if header.startswith(b"OggS"):
        return "audio/ogg"
    if header.startswith(b"ID3") or (
        len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0
    ):
        return "audio/mpeg"
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return "video/webm"
    if header.startswith(b"%PDF-"):
        return "application/pdf"
    if len(header) >= 12 and header[4:8] == b"ftyp":
        brand = header[8:12]
        if brand in {b"M4A ", b"M4B ", b"M4P "}:
            return "audio/mp4"
        return "video/mp4"
    return None


def detect_mime(path: str | Path) -> str:
    """Return a stable MIME type using magic bytes before the filename suffix."""

    source = require_file(path)
    with source.open("rb") as stream:
        header = stream.read(64)
    sniffed = _sniff_mime(header)
    if sniffed:
        return sniffed
    suffix_type = _MIME_BY_SUFFIX.get(source.suffix.lower())
    if suffix_type:
        return suffix_type
    guessed, _ = mimetypes.guess_type(str(source), strict=False)
    return guessed or "application/octet-stream"


def ffprobe(path: str | Path, *, ffprobe_binary: str = "ffprobe") -> dict[str, Any]:
    source = require_file(path)
    executable = shutil.which(ffprobe_binary)
    if not executable:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(f"ffprobe binary is unavailable: {ffprobe_binary}")
        completed = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(source)],
            check=False,
            capture_output=True,
            text=True,
        )
        match = re.search(
            r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", completed.stderr or ""
        )
        if not match:
            raise RuntimeError(f"ffprobe unavailable and ffmpeg found no duration: {source}")
        hours, minutes, seconds = match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        stream_types = [
            value.casefold()
            for value in re.findall(
                r"^\s*Stream #\S+(?:\([^)]+\))?:\s*(Video|Audio|Subtitle|Data):",
                completed.stderr or "",
                flags=re.IGNORECASE | re.MULTILINE,
            )
        ]
        if not stream_types:
            stream_types = [detect_mime(source).split("/", 1)[0]]
        return {
            "format": {
                "duration": str(duration),
                "probe_backend": "ffmpeg_stderr_fallback",
            },
            "streams": [
                {"codec_type": media_type, "duration": str(duration)}
                for media_type in stream_types
            ],
        }
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise RuntimeError(f"ffprobe failed for {source}: {detail}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for {source}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"ffprobe returned a non-object for {source}")
    return value


def describe_media(
    path: str | Path,
    *,
    include_probe: bool = False,
    ffprobe_binary: str = "ffprobe",
) -> MediaDescriptor:
    source = require_file(path)
    probe = ffprobe(source, ffprobe_binary=ffprobe_binary) if include_probe else None
    return MediaDescriptor(
        path=str(source),
        mime_type=detect_mime(source),
        size_bytes=source.stat().st_size,
        sha256=sha256_file(source),
        probe=probe,
    )


def media_manifest(
    path: str | Path,
    *,
    representation_id: str | None = None,
    include_probe: bool = False,
    ffprobe_binary: str = "ffprobe",
) -> dict[str, Any]:
    """Return the runtime's JSON-serializable representation manifest."""

    descriptor = describe_media(
        path, include_probe=include_probe, ffprobe_binary=ffprobe_binary
    )
    media_type = descriptor.mime_type.split("/", 1)[0]
    if descriptor.mime_type == "application/pdf":
        media_type = "document"
    return {
        "representation_id": representation_id,
        "path": descriptor.path,
        "media_type": media_type,
        "mime_type": descriptor.mime_type,
        "size_bytes": descriptor.size_bytes,
        "sha256": descriptor.sha256,
        "probe": descriptor.probe,
    }


def data_uri(path: str | Path, *, mime_type: str | None = None) -> str:
    source = require_file(path)
    resolved_mime = mime_type or detect_mime(source)
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:{resolved_mime};base64,{encoded}"


def gemini_inline_part(path: str | Path, *, mime_type: str | None = None) -> dict[str, Any]:
    source = require_file(path)
    resolved_mime = mime_type or detect_mime(source)
    return {
        "inlineData": {
            "mimeType": resolved_mime,
            "data": base64.b64encode(source.read_bytes()).decode("ascii"),
        }
    }


def openai_content_block(
    path: str | Path,
    *,
    mime_type: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    """Create a Chat-Completions-compatible multimodal content block.

    The deployed OpenAI-compatible teacher supports ``video_url`` in addition
    to standard image and audio blocks.  No transcoding occurs here.
    """

    source = require_file(path)
    resolved_mime = mime_type or detect_mime(source)
    if resolved_mime.startswith("text/") or resolved_mime in {
        "application/json",
        "application/xml",
    }:
        return {"type": "text", "text": decode_text_document(source)[0]}
    uri = data_uri(source, mime_type=resolved_mime)
    if resolved_mime.startswith("image/"):
        image_url: dict[str, Any] = {"url": uri}
        if detail:
            image_url["detail"] = detail
        return {"type": "image_url", "image_url": image_url}
    if resolved_mime.startswith("video/"):
        return {"type": "video_url", "video_url": {"url": uri}}
    if resolved_mime.startswith("audio/"):
        audio_format = {
            "audio/wav": "wav",
            "audio/mpeg": "mp3",
            "audio/flac": "flac",
            "audio/ogg": "ogg",
            "audio/mp4": "m4a",
        }.get(resolved_mime, source.suffix.lower().lstrip(".") or "wav")
        return {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(source.read_bytes()).decode("ascii"),
                "format": audio_format,
            },
        }
    return {"type": "file_url", "file_url": {"url": uri}}


def content_block(
    item: MediaInput | str | Path,
    *,
    api_style: str,
) -> dict[str, Any]:
    media = item if isinstance(item, MediaInput) else MediaInput(item)
    resolved = require_file(media.path)
    resolved_mime = media.mime_type or detect_mime(resolved)
    if resolved_mime.startswith("text/") or resolved_mime in {
        "application/json",
        "application/xml",
    }:
        return {"type": "text", "text": decode_text_document(resolved)[0]} if api_style in {
            "openai",
            "openai_chat",
            "chat",
            "chat_completions",
        } else {"text": decode_text_document(resolved)[0]}
    if api_style in {"openai", "openai_chat", "chat", "chat_completions"}:
        return openai_content_block(
            resolved, mime_type=resolved_mime, detail=media.detail
        )
    if api_style in {"gemini", "gemini_style", "generate_content"}:
        return gemini_inline_part(resolved, mime_type=resolved_mime)
    raise ValueError(f"unsupported API style: {api_style!r}")


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve()


def decode_text_document(path: str | Path) -> tuple[str, str]:
    """Decode a text upload deterministically without content inference."""

    source = require_file(path, label="text document")
    payload = source.read_bytes()
    if payload.startswith(b"\xef\xbb\xbf"):
        return payload.decode("utf-8-sig"), "utf-8-sig"
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        return payload.decode("utf-16"), "utf-16"
    try:
        return payload.decode("utf-8"), "utf-8"
    except UnicodeDecodeError as exc:
        raise ValueError(f"text document is not UTF-8/UTF-16: {source}") from exc


def _suffix_for_mime(mime_type: str) -> str:
    return {
        "application/json": ".json",
        "application/pdf": ".pdf",
        "audio/flac": ".flac",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "image/gif": ".gif",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "text/html": ".html",
        "text/markdown": ".md",
        "text/plain": ".txt",
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
    }.get(mime_type.split(";", 1)[0].strip().lower(), ".bin")


def download_url(
    url: str,
    cache_dir: str | Path,
    *,
    expected_sha256: str | None = None,
    max_bytes: int = 512 * 1024 * 1024,
    timeout: float = 120.0,
    max_attempts: int = 3,
    session: Any | None = None,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Download an HTTP(S) upload once into a content-addressed local cache."""

    parsed = urlparse(str(url))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL upload must use an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("URL upload must not contain inline credentials")
    if max_bytes < 1 or max_attempts < 1 or timeout <= 0:
        raise ValueError("download limits must be positive")
    destination = Path(cache_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    url_digest = hashlib.sha256(str(url).encode("utf-8")).hexdigest()
    metadata_path = destination / f"url_{url_digest}.json"
    if metadata_path.is_file():
        try:
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached_path = Path(str(cached.get("path") or ""))
            if cached_path.is_file() and cached.get("sha256") == sha256_file(cached_path):
                if expected_sha256 and _normalize_sha256(expected_sha256) != cached["sha256"]:
                    raise ValueError("cached URL upload does not match expected_sha256")
                return {"path": str(cached_path.resolve()), "provenance": cached}
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    transport = session or requests.Session()
    last_error: BaseException | None = None
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        started = time.monotonic()
        try:
            response = transport.request(
                "GET", str(url), timeout=timeout, allow_redirects=True
            )
            status = int(getattr(response, "status_code", 0))
            payload = getattr(response, "content", b"")
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            payload = bytes(payload or b"")
            record = {
                "attempt": attempt,
                "http_status": status,
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
                "size_bytes": len(payload),
            }
            attempts.append(record)
            if not 200 <= status < 300:
                last_error = RuntimeError(f"HTTP {status}")
                if status not in {408, 425, 429} and not 500 <= status <= 599:
                    break
                raise last_error
            if not payload:
                raise ValueError("URL upload returned empty bytes")
            if len(payload) > max_bytes:
                raise ValueError(f"URL upload exceeds max_bytes={max_bytes}")
            content_type = str((getattr(response, "headers", {}) or {}).get("content-type") or "")
            mime_type = content_type.split(";", 1)[0].strip().lower()
            suffix = Path(parsed.path).suffix.lower()
            if suffix not in _MIME_BY_SUFFIX:
                suffix = _suffix_for_mime(mime_type)
            target = destination / f"url_{url_digest}{suffix}"
            atomic_write_bytes(target, payload)
            actual_sha = sha256_file(target)
            if expected_sha256 and _normalize_sha256(expected_sha256) != actual_sha:
                target.unlink(missing_ok=True)
                raise ValueError("URL upload does not match expected_sha256")
            final_url = str(getattr(response, "url", url) or url)
            provenance = {
                "transport": "https_download" if parsed.scheme == "https" else "http_download",
                "source_url_sha256": "sha256:" + url_digest,
                "source_origin": f"{parsed.scheme}://{parsed.netloc}",
                "final_url_sha256": sha256_bytes(final_url.encode("utf-8")),
                "path": str(target.resolve()),
                "sha256": actual_sha,
                "size_bytes": len(payload),
                "declared_content_type": content_type,
                "detected_mime_type": detect_mime(target),
                "attempts": attempts,
            }
            atomic_write_bytes(
                metadata_path,
                (json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
            return {"path": str(target.resolve()), "provenance": provenance}
        except (requests.RequestException, OSError, RuntimeError) as exc:
            last_error = exc
            if not attempts or attempts[-1].get("attempt") != attempt:
                attempts.append(
                    {
                        "attempt": attempt,
                        "http_status": 0,
                        "latency_ms": round((time.monotonic() - started) * 1000, 3),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            if attempt >= max_attempts:
                break
            sleep(min(30.0, 2 ** (attempt - 1)))
    raise RuntimeError(f"URL upload failed after {len(attempts)} attempt(s): {last_error}") from last_error


def _normalize_sha256(value: str) -> str:
    text = str(value).strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError("expected_sha256 must contain 64 hexadecimal characters")
    return "sha256:" + text


def _pdf_page_count(path: Path, *, pdfinfo_binary: str = "pdfinfo") -> int:
    executable = shutil.which(pdfinfo_binary)
    if executable:
        completed = subprocess.run(
            [executable, str(path)], check=False, capture_output=True, text=True
        )
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                if line.lower().startswith("pages:"):
                    try:
                        count = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        break
                    if count > 0:
                        return count
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"cannot determine PDF page count: {path}") from exc


def _duration_ms(path: Path) -> int:
    return max(1, int(round(_media_duration_seconds(path) * 1000)))


def selector_catalog(path: str | Path, *, text_page_chars: int = 16000) -> dict[str, Any]:
    """Describe deterministic selectors available for an uploaded file."""

    source = require_file(path)
    if text_page_chars < 1:
        raise ValueError("text_page_chars must be positive")
    mime_type = detect_mime(source)
    selectors: list[dict[str, Any]] = [{"mode": "default_preview"}, {"mode": "original"}]
    limits: dict[str, Any] = {}
    if mime_type == "application/pdf":
        page_count = _pdf_page_count(source)
        selectors.extend({"mode": "page", "page_index": index} for index in range(page_count))
        limits["page_count"] = page_count
    elif mime_type.startswith("text/") or mime_type in {"application/json", "application/xml"}:
        text, encoding = decode_text_document(source)
        page_count = max(1, ceil(len(text) / max(1, text_page_chars)))
        selectors.extend({"mode": "page", "page_index": index} for index in range(page_count))
        limits.update(page_count=page_count, text_page_chars=text_page_chars, encoding=encoding)
    elif mime_type.startswith("video/"):
        duration_ms = _duration_ms(source)
        selectors.append({"mode": "frame", "time_ms": 0})
        selectors.append({"mode": "frame", "time_ms": duration_ms // 2})
        selectors.append({"mode": "time_range", "start_ms": 0, "end_ms": duration_ms})
        limits["duration_ms"] = duration_ms
    elif mime_type.startswith("audio/"):
        duration_ms = _duration_ms(source)
        selectors.append({"mode": "time_range", "start_ms": 0, "end_ms": duration_ms})
        limits["duration_ms"] = duration_ms
    return {"selectors": selectors, "limits": limits}


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@functools.lru_cache(maxsize=4096)
def _sha256_for_file_identity(
    path: str,
    device: int,
    inode: int,
    size: int,
    mtime_ns: int,
    ctime_ns: int,
) -> str:
    """Hash immutable media once per process for one exact filesystem identity."""

    del device, inode, size, mtime_ns, ctime_ns
    return sha256_file(path)


def content_sha256(path: str | Path) -> str:
    """Return a streaming content hash, memoized only while file identity is stable.

    StoryMem artifacts are immutable after publication.  Including device,
    inode, size, mtime and ctime in the memoization key still invalidates the
    entry if a producer atomically replaces or edits a file.  This avoids
    repeatedly reading a generated video merely because several deterministic
    media views are requested in one process.
    """

    source = require_file(path)
    stat = source.stat()
    return _sha256_for_file_identity(
        str(source),
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


@functools.lru_cache(maxsize=32)
def _executable_sha256(path: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    del size, mtime_ns, ctime_ns
    return sha256_file(path)


def _tool_identity(binary: str) -> dict[str, str]:
    executable = shutil.which(binary)
    if not executable:
        raise RuntimeError(f"media executable is unavailable: {binary}")
    resolved = Path(executable).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": _executable_sha256(
            str(resolved),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        ),
    }


def _derivation_cache_key(specification: Mapping[str, Any]) -> str:
    payload = json.dumps(
        specification,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _cache_artifact_is_valid(
    path: Path,
    record: Any,
) -> bool:
    if not isinstance(record, Mapping):
        return False
    try:
        expected_size = int(record.get("size_bytes", -1))
    except (TypeError, ValueError):
        return False
    expected_sha256 = str(record.get("sha256") or "")
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and expected_size > 0
        and content_sha256(path) == expected_sha256
    )


def _cache_manifest_is_valid(
    manifest: Any,
    *,
    specification: Mapping[str, Any],
    required_artifacts: Mapping[str, Path],
) -> bool:
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("cache_version") != _DERIVATION_CACHE_VERSION
        or manifest.get("specification") != dict(specification)
        or not isinstance(manifest.get("artifacts"), Mapping)
    ):
        return False
    records = manifest["artifacts"]
    return all(
        _cache_artifact_is_valid(path, records.get(name))
        for name, path in required_artifacts.items()
    )


def _artifact_record(path: Path) -> dict[str, Any]:
    source = require_file(path, label="cached media artifact")
    return {
        "size_bytes": source.stat().st_size,
        "sha256": content_sha256(source),
    }


def _atomic_materialize_cached_file(
    cached: Path,
    destination: str | Path,
    *,
    expected_sha256: str,
) -> Path:
    """Atomically publish cached immutable bytes at a caller-compatible path."""

    source = require_file(cached, label="cached media artifact")
    target = Path(destination).expanduser().resolve()
    if target == source:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if (
        target.is_file()
        and target.stat().st_size == source.stat().st_size
        and content_sha256(target) == expected_sha256
    ):
        return target
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink(missing_ok=True)
        try:
            # Outputs are immutable StoryMem artifacts.  A hard link avoids a
            # second large copy on the common single-filesystem dataset path.
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _cache_root_for_output(output: Path, *, namespace: str) -> Path:
    """Choose a hidden cache shared by related outputs, never by source location."""

    parent = output if output.suffix == "" else output.parent
    if namespace == "boundary_frames" and parent.name.startswith("."):
        # The production finalizer first writes into ``.opening`` and then
        # requests ``tail.png`` from its parent.  Normalize both calls to the
        # same cache without changing either public artifact path.
        parent = parent.parent
    if namespace == "visual_only":
        for candidate in (parent, *parent.parents):
            if candidate.name == "perception_proxies":
                parent = candidate
                break
    root = parent / _DERIVATION_CACHE_DIRNAME / namespace
    root.mkdir(parents=True, exist_ok=True)
    return root


def _atomic_write_cache_manifest(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path,
        (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _asset_original(asset: Mapping[str, Any]) -> dict[str, Any]:
    representation_id = str(asset.get("default_representation_id") or "")
    representations = asset.get("representations")
    original = representations.get(representation_id) if isinstance(representations, Mapping) else None
    if not isinstance(original, Mapping):
        raise ValueError("asset has no default original representation")
    require_file(str(original.get("path") or ""), label="asset representation")
    return copy.deepcopy(dict(original))


def _derived_manifest(
    path: Path,
    *,
    representation_id: str,
    parent_sha256: str,
    selector: Mapping[str, Any],
    derivation: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = media_manifest(path, representation_id=representation_id)
    manifest.update(
        {
            "parent_sha256": parent_sha256,
            "selector": copy.deepcopy(dict(selector)),
            "derivation": copy.deepcopy(dict(derivation)),
        }
    )
    return manifest


def _run_checked(command: Sequence[str], *, label: str) -> None:
    completed = subprocess.run(list(command), check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise RuntimeError(f"{label} failed: {detail}")


def materialize_asset_selector(
    asset: Mapping[str, Any],
    selector: Mapping[str, Any],
    cache_dir: str | Path,
    *,
    text_page_chars: int = 16000,
    ffmpeg_binary: str = "ffmpeg",
    pdftoppm_binary: str = "pdftoppm",
    pdftotext_binary: str = "pdftotext",
) -> dict[str, Any]:
    """Resolve one upload selector to immutable, cached representations.

    PDF pages return a rendered page as the primary representation and, when
    available, exact parser-extracted page text as a supplemental one.  No VLM,
    captioner, OCR model, or semantic rewriter is used.
    """

    if not isinstance(selector, Mapping) or not isinstance(selector.get("mode"), str):
        raise ValueError("selector must be an object with a mode")
    original = _asset_original(asset)
    source = require_file(str(original["path"]), label="asset representation")
    source_sha = str(original.get("sha256") or sha256_file(source))
    if source_sha != sha256_file(source):
        raise ValueError("asset representation hash no longer matches its bytes")
    mode = str(selector["mode"])
    allowed_fields = {
        "default_preview": {"mode"},
        "original": {"mode"},
        "page": {"mode", "page_index"},
        "frame": {"mode", "time_ms"},
        "time_range": {"mode", "start_ms", "end_ms"},
    }
    if mode not in allowed_fields or set(selector) != allowed_fields[mode]:
        raise ValueError(f"invalid closed selector: {dict(selector)!r}")
    mime_type = str(original.get("mime_type") or detect_mime(source))
    if mode == "original" or (
        mode == "default_preview"
        and mime_type != "application/pdf"
        and not (mime_type.startswith("text/") or mime_type in {"application/json", "application/xml"})
    ):
        representation = copy.deepcopy(original)
        representation["selector"] = copy.deepcopy(dict(selector))
        representation["parent_sha256"] = source_sha
        representation["derivation"] = {"version": "storymem.selector.v1", "operation": "identity"}
        return {
            "selector": copy.deepcopy(dict(selector)),
            "representation": representation,
            "supplemental_representations": [],
        }

    effective = dict(selector)
    if mode == "default_preview":
        effective = {"mode": "page", "page_index": 0}
    selector_digest = hashlib.sha256(
        canonical_selector_bytes(source_sha, selector, effective)
    ).hexdigest()[:20]
    asset_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(asset.get("asset_id") or "asset")
    )[:64]
    representation_id = f"repr_{asset_id}_{selector_digest}"
    destination = Path(cache_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / f".{representation_id}.lock"
    supplemental: list[dict[str, Any]] = []
    with _file_lock(lock_path):
        if effective["mode"] == "page":
            page_index = int(effective.get("page_index", -1))
            if page_index < 0:
                raise ValueError("page_index must be non-negative")
            if mime_type == "application/pdf":
                page_count = _pdf_page_count(source)
                if page_index >= page_count:
                    raise ValueError(f"PDF page_index {page_index} is outside page_count={page_count}")
                target = destination / f"{representation_id}.png"
                render = shutil.which(pdftoppm_binary)
                if not render:
                    raise RuntimeError(f"pdftoppm binary is unavailable: {pdftoppm_binary}")
                if not target.is_file() or target.stat().st_size <= 0:
                    with tempfile.TemporaryDirectory(prefix="pdf-page-", dir=destination) as temp_dir:
                        prefix = Path(temp_dir) / "page"
                        _run_checked(
                            [
                                render,
                                "-f",
                                str(page_index + 1),
                                "-l",
                                str(page_index + 1),
                                "-singlefile",
                                "-png",
                                str(source),
                                str(prefix),
                            ],
                            label="PDF page rendering",
                        )
                        rendered = prefix.with_suffix(".png")
                        if not rendered.is_file():
                            raise RuntimeError("PDF page renderer produced no image")
                        os.replace(rendered, target)
                derivation = {
                    "version": "storymem.selector.v1",
                    "operation": "pdf_page_render",
                    "effective_selector": effective,
                    "tool": str(render),
                    "tool_sha256": sha256_file(render),
                }
                primary = _derived_manifest(
                    target,
                    representation_id=representation_id,
                    parent_sha256=source_sha,
                    selector=selector,
                    derivation=derivation,
                )
                text_target = destination / f"{representation_id}.txt"
                extractor = shutil.which(pdftotext_binary)
                extracted_text = ""
                extraction: dict[str, Any]
                if extractor:
                    completed = subprocess.run(
                        [
                            extractor,
                            "-f",
                            str(page_index + 1),
                            "-l",
                            str(page_index + 1),
                            "-layout",
                            "-enc",
                            "UTF-8",
                            str(source),
                            "-",
                        ],
                        check=False,
                        capture_output=True,
                    )
                    if completed.returncode == 0:
                        extracted_text = completed.stdout.decode("utf-8", errors="strict")
                    extraction = {
                        "operation": "pdf_page_text",
                        "tool": str(extractor),
                        "tool_sha256": sha256_file(extractor),
                    }
                else:
                    try:
                        from pypdf import PdfReader

                        extracted_text = PdfReader(str(source)).pages[page_index].extract_text() or ""
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError("no deterministic PDF text extractor is available") from exc
                    extraction = {"operation": "pdf_page_text", "tool": "pypdf"}
                if extracted_text:
                    atomic_write_bytes(text_target, extracted_text.encode("utf-8"))
                    supplemental.append(
                        _derived_manifest(
                            text_target,
                            representation_id=f"{representation_id}_text",
                            parent_sha256=source_sha,
                            selector=selector,
                            derivation={
                                "version": "storymem.selector.v1",
                                "effective_selector": effective,
                                **extraction,
                            },
                        )
                    )
            elif mime_type.startswith("text/") or mime_type in {"application/json", "application/xml"}:
                text, encoding = decode_text_document(source)
                page_count = max(1, ceil(len(text) / max(1, text_page_chars)))
                if page_index >= page_count:
                    raise ValueError(f"text page_index {page_index} is outside page_count={page_count}")
                start = page_index * text_page_chars
                end = min(len(text), start + text_page_chars)
                target = destination / f"{representation_id}.txt"
                atomic_write_bytes(target, text[start:end].encode("utf-8"))
                primary = _derived_manifest(
                    target,
                    representation_id=representation_id,
                    parent_sha256=source_sha,
                    selector=selector,
                    derivation={
                        "version": "storymem.selector.v1",
                        "operation": "text_codepoint_page",
                        "effective_selector": effective,
                        "source_encoding": encoding,
                        "source_span": {"unit": "unicode_codepoint", "start": start, "end": end},
                    },
                )
            else:
                raise ValueError(f"page selector is unsupported for {mime_type}")
        elif effective["mode"] == "frame":
            if not mime_type.startswith("video/"):
                raise ValueError(f"frame selector is unsupported for {mime_type}")
            time_ms = int(effective.get("time_ms", -1))
            duration_ms = _duration_ms(source)
            if time_ms < 0 or time_ms > duration_ms:
                raise ValueError(f"time_ms must be within [0, {duration_ms}]")
            target = destination / f"{representation_id}.png"
            executable = shutil.which(ffmpeg_binary)
            if not executable:
                raise RuntimeError(f"ffmpeg binary is unavailable: {ffmpeg_binary}")
            if not target.is_file() or target.stat().st_size <= 0:
                with tempfile.TemporaryDirectory(prefix="video-frame-", dir=destination) as temp_dir:
                    rendered = Path(temp_dir) / "frame.png"
                    _run_checked(
                        [
                            executable,
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-ss",
                            f"{time_ms / 1000.0:.6f}",
                            "-i",
                            str(source),
                            "-frames:v",
                            "1",
                            str(rendered),
                        ],
                        label="video frame extraction",
                    )
                    os.replace(rendered, target)
            primary = _derived_manifest(
                target,
                representation_id=representation_id,
                parent_sha256=source_sha,
                selector=selector,
                derivation={
                    "version": "storymem.selector.v1",
                    "operation": "video_frame",
                    "effective_selector": effective,
                    "tool": str(executable),
                    "tool_sha256": sha256_file(executable),
                },
            )
        elif effective["mode"] == "time_range":
            if not (mime_type.startswith("video/") or mime_type.startswith("audio/")):
                raise ValueError(f"time_range selector is unsupported for {mime_type}")
            start_ms = int(effective.get("start_ms", -1))
            end_ms = int(effective.get("end_ms", -1))
            duration_ms = _duration_ms(source)
            if start_ms < 0 or end_ms <= start_ms or end_ms > duration_ms:
                raise ValueError(f"time range must satisfy 0 <= start < end <= {duration_ms}")
            executable = shutil.which(ffmpeg_binary)
            if not executable:
                raise RuntimeError(f"ffmpeg binary is unavailable: {ffmpeg_binary}")
            is_video = mime_type.startswith("video/")
            target = destination / f"{representation_id}{'.mp4' if is_video else '.wav'}"
            if not target.is_file() or target.stat().st_size <= 0:
                with tempfile.TemporaryDirectory(prefix="media-range-", dir=destination) as temp_dir:
                    rendered = Path(temp_dir) / target.name
                    command = [
                        executable,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                    ]
                    if is_video:
                        command.extend(
                            [
                                "-ss",
                                f"{start_ms / 1000.0:.6f}",
                                "-i",
                                str(source),
                                "-t",
                                f"{(end_ms - start_ms) / 1000.0:.6f}",
                            ]
                        )
                        command.extend(
                            ["-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart"]
                        )
                    else:
                        command.extend(
                            [
                                "-i",
                                str(source),
                                "-map",
                                "0:a:0",
                                "-vn",
                                "-af",
                                (
                                    f"atrim=start={start_ms / 1000.0:.6f}:"
                                    f"end={end_ms / 1000.0:.6f},asetpts=PTS-STARTPTS"
                                ),
                                "-acodec",
                                "pcm_s16le",
                                "-ar",
                                "48000",
                                "-ac",
                                "1",
                            ]
                        )
                    command.append(str(rendered))
                    _run_checked(command, label="media time-range extraction")
                    os.replace(rendered, target)
            primary = _derived_manifest(
                target,
                representation_id=representation_id,
                parent_sha256=source_sha,
                selector=selector,
                derivation={
                    "version": "storymem.selector.v1",
                    "operation": "video_time_range" if is_video else "audio_time_range",
                    "effective_selector": effective,
                    "tool": str(executable),
                    "tool_sha256": sha256_file(executable),
                },
            )
            # A video range is useful both as visual context and, when an audio
            # stream exists, as an independently typed voice/audio source.
            # Never rely on an audio track embedded inside the video transport:
            # memory-slot authorization is representation- and modality-exact.
            if is_video:
                probe = ffprobe(source)
                has_audio_stream = any(
                    isinstance(stream, Mapping)
                    and stream.get("codec_type") == "audio"
                    for stream in probe.get("streams") or []
                )
                if has_audio_stream:
                    audio_target = destination / f"{representation_id}_audio.wav"
                    if not audio_target.is_file() or audio_target.stat().st_size <= 0:
                        with tempfile.TemporaryDirectory(
                            prefix="video-range-audio-", dir=destination
                        ) as temp_dir:
                            rendered_audio = Path(temp_dir) / audio_target.name
                            _run_checked(
                                [
                                    executable,
                                    "-hide_banner",
                                    "-loglevel",
                                    "error",
                                    "-y",
                                    "-ss",
                                    f"{start_ms / 1000.0:.6f}",
                                    "-i",
                                    str(source),
                                    "-t",
                                    f"{(end_ms - start_ms) / 1000.0:.6f}",
                                    "-map",
                                    "0:a:0",
                                    "-vn",
                                    "-acodec",
                                    "pcm_s16le",
                                    "-ar",
                                    "48000",
                                    "-ac",
                                    "1",
                                    str(rendered_audio),
                                ],
                                label="video time-range audio extraction",
                            )
                            os.replace(rendered_audio, audio_target)
                    supplemental.append(
                        _derived_manifest(
                            audio_target,
                            representation_id=f"{representation_id}_audio",
                            parent_sha256=source_sha,
                            selector=selector,
                            derivation={
                                "version": "storymem.selector.v1",
                                "operation": "video_time_range_audio",
                                "effective_selector": effective,
                                "tool": str(executable),
                                "tool_sha256": sha256_file(executable),
                            },
                        )
                    )
        else:
            raise AssertionError(f"unhandled selector mode: {effective['mode']}")
    return {
        "selector": copy.deepcopy(dict(selector)),
        "representation": primary,
        "supplemental_representations": supplemental,
    }


def canonical_selector_bytes(
    source_sha256: str,
    selector: Mapping[str, Any],
    effective_selector: Mapping[str, Any],
) -> bytes:
    return json.dumps(
        {
            "version": "storymem.selector.v1",
            "source_sha256": source_sha256,
            "selector": dict(selector),
            "effective_selector": dict(effective_selector),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _duration_seconds(probe: dict[str, Any]) -> float:
    values: list[Any] = []
    if isinstance(probe.get("format"), dict):
        values.append(probe["format"].get("duration"))
    for stream in probe.get("streams") or []:
        if isinstance(stream, dict) and stream.get("codec_type") == "video":
            values.append(stream.get("duration"))
    for value in values:
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            return duration
    raise ValueError("media duration is unavailable")


def _media_duration_seconds(
    path: str | Path,
    *,
    ffprobe_binary: str = "ffprobe",
    ffmpeg_binary: str = "ffmpeg",
) -> float:
    try:
        return _duration_seconds(ffprobe(path, ffprobe_binary=ffprobe_binary))
    except RuntimeError as probe_error:
        executable = shutil.which(ffmpeg_binary)
        if not executable:
            raise probe_error
        completed = subprocess.run(
            [executable, "-hide_banner", "-i", str(require_file(path))],
            check=False,
            capture_output=True,
            text=True,
        )
        match = re.search(
            r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", completed.stderr or ""
        )
        if not match:
            raise RuntimeError(f"cannot determine media duration: {path}") from probe_error
        hours, minutes, seconds = match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if duration <= 0:
            raise RuntimeError(f"media duration is not positive: {path}")
        return duration


def _boundary_decoder_identity(executable: str) -> dict[str, Any]:
    identity: dict[str, Any] = {"fallback": {"name": "ffmpeg_cli", **_tool_identity(executable)}}
    try:
        import av
    except ImportError:
        identity["preferred"] = identity["fallback"]
        return identity
    identity["preferred"] = {
        "name": "pyav",
        "version": str(av.__version__),
        "library_versions": {
            str(name): list(version)
            for name, version in sorted(av.library_versions.items())
        },
    }
    return identity


def _encode_video_frame_png(
    frame: Any,
    destination: Path,
    *,
    sample_aspect_ratio: Any,
) -> None:
    import av

    output = av.open(str(destination), mode="w")
    try:
        stream = output.add_stream("png")
        stream.width = int(frame.width)
        stream.height = int(frame.height)
        stream.pix_fmt = "rgb24"
        if sample_aspect_ratio is not None:
            stream.sample_aspect_ratio = sample_aspect_ratio
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)
    finally:
        output.close()


def _decode_video_boundaries(
    source: Path,
    *,
    opening_output: Path,
    tail_output: Path,
    executable: str,
) -> None:
    """Decode a video once while retaining only its first and final frames."""

    try:
        import av

        container = av.open(str(source))
        try:
            if not container.streams.video:
                raise RuntimeError(f"video has no decodable video stream: {source}")
            video_stream = container.streams.video[0]
            sample_aspect_ratio = video_stream.sample_aspect_ratio
            first_frame = None
            tail_frame = None
            for frame in container.decode(video_stream):
                if first_frame is None:
                    first_frame = frame
                tail_frame = frame
        finally:
            container.close()
        if first_frame is None or tail_frame is None:
            raise RuntimeError(f"video has no decodable frames: {source}")
        _encode_video_frame_png(
            first_frame,
            opening_output,
            sample_aspect_ratio=sample_aspect_ratio,
        )
        _encode_video_frame_png(
            tail_frame,
            tail_output,
            sample_aspect_ratio=sample_aspect_ratio,
        )
        return
    except ImportError:
        pass
    except Exception:
        # PyAV and the configured ffmpeg binary can differ in codec support.
        # Retrying via the existing CLI path preserves compatibility without
        # publishing either partially encoded file.
        opening_output.unlink(missing_ok=True)
        tail_output.unlink(missing_ok=True)

    _run_checked(
        [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            (
                "[0:v:0]split=2[opening_all][tail_all];"
                "[opening_all]select=eq(n\\,0)[opening]"
            ),
            "-map",
            "[opening]",
            "-frames:v:0",
            "1",
            "-an",
            str(opening_output),
            "-map",
            "[tail_all]",
            "-an",
            "-fps_mode",
            "passthrough",
            "-update",
            "1",
            str(tail_output),
        ],
        label="opening/tail frame extraction",
    )


def _ensure_boundary_frame_cache(
    source: Path,
    *,
    output_hint: Path,
    executable: str,
) -> tuple[Path, Path, dict[str, Any]]:
    """Decode one video once and cache its exact opening and final frames."""

    specification = {
        "cache_version": _DERIVATION_CACHE_VERSION,
        "operation": _BOUNDARY_FRAME_OPERATION_VERSION,
        "source_sha256": content_sha256(source),
        "image_format": "png",
        "decoder": _boundary_decoder_identity(executable),
    }
    cache_key = _derivation_cache_key(specification)
    cache_root = _cache_root_for_output(output_hint, namespace="boundary_frames")
    opening_cache = cache_root / f"{cache_key}.opening.blob"
    tail_cache = cache_root / f"{cache_key}.tail.blob"
    manifest_path = cache_root / f"{cache_key}.json"
    lock_path = cache_root / f"{cache_key}.lock"
    artifacts = {"opening": opening_cache, "tail": tail_cache}
    with _file_lock(lock_path):
        manifest = _read_json_object(manifest_path)
        if not _cache_manifest_is_valid(
            manifest,
            specification=specification,
            required_artifacts=artifacts,
        ):
            with tempfile.TemporaryDirectory(
                prefix=f".{cache_key[:16]}.", dir=cache_root
            ) as temporary_dir:
                opening_temporary = Path(temporary_dir) / "opening.png"
                tail_temporary = Path(temporary_dir) / "tail.png"
                _decode_video_boundaries(
                    source,
                    opening_output=opening_temporary,
                    tail_output=tail_temporary,
                    executable=executable,
                )
                require_file(opening_temporary, label="opening-frame extraction")
                require_file(tail_temporary, label="tail-frame extraction")
                os.replace(opening_temporary, opening_cache)
                os.replace(tail_temporary, tail_cache)
            manifest = {
                "cache_version": _DERIVATION_CACHE_VERSION,
                "cache_key": cache_key,
                "specification": specification,
                "artifacts": {
                    name: _artifact_record(path) for name, path in artifacts.items()
                },
            }
            _atomic_write_cache_manifest(manifest_path, manifest)
        assert isinstance(manifest, Mapping)
        return opening_cache, tail_cache, copy.deepcopy(dict(manifest))


def _ensure_selected_frame_cache(
    source: Path,
    *,
    output_hint: Path,
    executable: str,
    timestamp: float,
    extension: str,
) -> tuple[Path, dict[str, Any]]:
    """Cache one timestamp-selected frame without changing its legacy encoding."""

    specification = {
        "cache_version": _DERIVATION_CACHE_VERSION,
        "operation": "storymem.timestamp-frame-extract.v1",
        "source_sha256": content_sha256(source),
        "timestamp_seconds": f"{timestamp:.6f}",
        "image_format": extension,
        "tool": _tool_identity(executable),
    }
    cache_key = _derivation_cache_key(specification)
    cache_root = _cache_root_for_output(output_hint, namespace="selected_frames")
    artifact = cache_root / f"{cache_key}.blob"
    manifest_path = cache_root / f"{cache_key}.json"
    lock_path = cache_root / f"{cache_key}.lock"
    with _file_lock(lock_path):
        manifest = _read_json_object(manifest_path)
        if not _cache_manifest_is_valid(
            manifest,
            specification=specification,
            required_artifacts={"frame": artifact},
        ):
            with tempfile.TemporaryDirectory(
                prefix=f".{cache_key[:16]}.", dir=cache_root
            ) as temporary_dir:
                rendered = Path(temporary_dir) / f"frame.{extension}"
                command = [
                    executable,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{timestamp:.6f}",
                    "-i",
                    str(source),
                    "-frames:v",
                    "1",
                ]
                if extension in {"jpg", "jpeg"}:
                    command.extend(["-q:v", "2"])
                command.append(str(rendered))
                _run_checked(command, label=f"ffmpeg frame extraction at {timestamp:.6f}s")
                require_file(rendered, label="frame extraction")
                os.replace(rendered, artifact)
            manifest = {
                "cache_version": _DERIVATION_CACHE_VERSION,
                "cache_key": cache_key,
                "specification": specification,
                "artifacts": {"frame": _artifact_record(artifact)},
            }
            _atomic_write_cache_manifest(manifest_path, manifest)
        assert isinstance(manifest, Mapping)
        return artifact, copy.deepcopy(dict(manifest))


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    count: int = 8,
    timestamps: Sequence[float] | None = None,
    image_format: str = "png",
    ffmpeg_binary: str = "ffmpeg",
    ffprobe_binary: str = "ffprobe",
) -> list[Path]:
    """Extract deterministic full-resolution frames and return their manifests."""

    source = require_file(video_path, label="video")
    if not detect_mime(source).startswith("video/"):
        raise ValueError(f"not a recognized video: {source}")
    executable = shutil.which(ffmpeg_binary)
    if not executable:
        raise RuntimeError(f"ffmpeg binary is unavailable: {ffmpeg_binary}")
    if timestamps is None:
        if count < 1:
            raise ValueError("count must be positive")
        duration = _media_duration_seconds(
            source,
            ffprobe_binary=ffprobe_binary,
            ffmpeg_binary=ffmpeg_binary,
        )
        # Midpoints avoid fragile exact first/last timestamps.
        chosen = [duration * (index + 0.5) / count for index in range(count)]
    else:
        chosen = [float(value) for value in timestamps]
        if not chosen or any(value < 0 for value in chosen):
            raise ValueError("timestamps must be a non-empty sequence of non-negative values")
    extension = image_format.lower().lstrip(".")
    if extension not in {"png", "jpg", "jpeg", "webp"}:
        raise ValueError("image_format must be png, jpg, jpeg, or webp")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []
    if extension == "png" and len(chosen) == 1 and chosen[0] == 0.0:
        opening_cache, _, manifest = _ensure_boundary_frame_cache(
            source,
            output_hint=destination,
            executable=executable,
        )
        target = destination / "frame_000.png"
        return [
            _atomic_materialize_cached_file(
                opening_cache,
                target,
                expected_sha256=str(manifest["artifacts"]["opening"]["sha256"]),
            )
        ]
    for index, timestamp in enumerate(chosen):
        target = destination / f"frame_{index:03d}.{extension}"
        cached, manifest = _ensure_selected_frame_cache(
            source,
            output_hint=destination,
            executable=executable,
            timestamp=timestamp,
            extension=extension,
        )
        results.append(
            _atomic_materialize_cached_file(
                cached,
                target,
                expected_sha256=str(manifest["artifacts"]["frame"]["sha256"]),
            )
        )
    return results


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int = 48000,
    channels: int = 1,
    ffmpeg_binary: str = "ffmpeg",
    required: bool = True,
) -> Path | None:
    """Extract the first audio stream as deterministic PCM WAV."""

    source = require_file(video_path, label="video")
    executable = shutil.which(ffmpeg_binary)
    if not executable:
        raise RuntimeError(f"ffmpeg binary is unavailable: {ffmpeg_binary}")
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("sample_rate and channels must be positive")
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".wav":
        raise ValueError("audio output_path must end in .wav")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.wav")
    completed = subprocess.run(
        [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(int(sample_rate)),
            "-ac",
            str(int(channels)),
            str(temporary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size <= 0:
        temporary.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        if not required:
            return None
        raise RuntimeError(f"ffmpeg audio extraction failed: {detail}")
    os.replace(temporary, destination)
    return destination


def strip_video_audio(
    video_path: str | Path,
    output_path: str | Path,
    *,
    ffmpeg_binary: str = "ffmpeg",
    ffprobe_binary: str = "ffprobe",
) -> tuple[Path, bool]:
    """Return a visual-only video, remuxing only when an audio stream exists.

    The original provider or uploaded artifact is never modified.  Videos that
    already have no audio are returned directly so the common Echo path does
    not duplicate large files.  When audio is present, the encoded video stream
    is copied bit-for-bit into a separate container and every non-video stream
    is dropped before the media can be shown to a visual-only Teacher.
    """

    source = require_file(video_path, label="video")
    if not detect_mime(source).startswith("video/"):
        raise ValueError(f"not a recognized video: {source}")
    executable = shutil.which(ffmpeg_binary)
    if not executable:
        raise RuntimeError(f"ffmpeg binary is unavailable: {ffmpeg_binary}")
    destination = Path(output_path).expanduser().resolve()
    if destination == source:
        raise ValueError("visual-only output must not overwrite the source video")
    if not destination.suffix:
        raise ValueError("visual-only output_path must have a container suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffprobe_identity: dict[str, str]
    resolved_ffprobe = shutil.which(ffprobe_binary)
    if resolved_ffprobe:
        ffprobe_identity = _tool_identity(resolved_ffprobe)
    else:
        # ``ffprobe`` itself has a documented ffmpeg fallback.  Bind the cache
        # to that executable too, so a tool replacement cannot silently reuse
        # bytes derived under a different media stack.
        ffprobe_identity = {
            "fallback": "ffmpeg_stderr",
            **_tool_identity(executable),
        }
    specification = {
        "cache_version": _DERIVATION_CACHE_VERSION,
        "operation": _VISUAL_ONLY_OPERATION_VERSION,
        "source_sha256": content_sha256(source),
        "container_suffix": destination.suffix.lower(),
        "ffmpeg": _tool_identity(executable),
        "ffprobe": ffprobe_identity,
    }
    cache_key = _derivation_cache_key(specification)
    cache_root = _cache_root_for_output(destination, namespace="visual_only")
    cached_video = cache_root / f"{cache_key}.blob"
    manifest_path = cache_root / f"{cache_key}.json"
    lock_path = cache_root / f"{cache_key}.lock"
    with _file_lock(lock_path):
        manifest = _read_json_object(manifest_path)
        manifest_matches = (
            isinstance(manifest, Mapping)
            and manifest.get("cache_version") == _DERIVATION_CACHE_VERSION
            and manifest.get("specification") == specification
            and type(manifest.get("audio_removed")) is bool
        )
        if manifest_matches and manifest.get("audio_removed") is False:
            return source, False
        if manifest_matches and manifest.get("audio_removed") is True:
            if _cache_manifest_is_valid(
                manifest,
                specification=specification,
                required_artifacts={"visual_only": cached_video},
            ):
                record = manifest["artifacts"]["visual_only"]
                return (
                    _atomic_materialize_cached_file(
                        cached_video,
                        destination,
                        expected_sha256=str(record["sha256"]),
                    ),
                    True,
                )

        probe = ffprobe(source, ffprobe_binary=ffprobe_binary)
        streams = [
            row for row in probe.get("streams") or [] if isinstance(row, Mapping)
        ]
        if not any(row.get("codec_type") == "video" for row in streams):
            raise RuntimeError(f"video has no decodable video stream: {source}")
        if not any(row.get("codec_type") == "audio" for row in streams):
            _atomic_write_cache_manifest(
                manifest_path,
                {
                    "cache_version": _DERIVATION_CACHE_VERSION,
                    "cache_key": cache_key,
                    "specification": specification,
                    "audio_removed": False,
                    "artifacts": {},
                },
            )
            return source, False

        with tempfile.TemporaryDirectory(
            prefix=f".{cache_key[:16]}.", dir=cache_root
        ) as temporary_dir:
            temporary = Path(temporary_dir) / f"visual_only{destination.suffix.lower()}"
            command = [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                "-an",
                "-sn",
                "-dn",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                str(temporary),
            ]
            _run_checked(command, label="visual-only video remux")
            require_file(temporary, label="visual-only video remux")
            sanitized_probe = ffprobe(temporary, ffprobe_binary=ffprobe_binary)
            sanitized_streams = [
                row
                for row in sanitized_probe.get("streams") or []
                if isinstance(row, Mapping)
            ]
            if not any(row.get("codec_type") == "video" for row in sanitized_streams):
                raise RuntimeError("visual-only video remux lost the video stream")
            if any(row.get("codec_type") == "audio" for row in sanitized_streams):
                raise RuntimeError("visual-only video remux retained an audio stream")
            os.replace(temporary, cached_video)
        manifest = {
            "cache_version": _DERIVATION_CACHE_VERSION,
            "cache_key": cache_key,
            "specification": specification,
            "audio_removed": True,
            "artifacts": {"visual_only": _artifact_record(cached_video)},
        }
        _atomic_write_cache_manifest(manifest_path, manifest)
        record = manifest["artifacts"]["visual_only"]
        return (
            _atomic_materialize_cached_file(
                cached_video,
                destination,
                expected_sha256=str(record["sha256"]),
            ),
            True,
        )


def extract_video_tail_frame(
    video_path: str | Path,
    output_path: str | Path,
    *,
    ffmpeg_binary: str = "ffmpeg",
    ffprobe_binary: str = "ffprobe",
) -> Path:
    """Decode through the final short window and atomically retain its last frame."""

    source = require_file(video_path, label="video")
    if not detect_mime(source).startswith("video/"):
        raise ValueError(f"not a recognized video: {source}")
    executable = shutil.which(ffmpeg_binary)
    if not executable:
        raise RuntimeError(f"ffmpeg binary is unavailable: {ffmpeg_binary}")
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".png":
        raise ValueError("tail-frame output_path must end in .png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _, tail_cache, manifest = _ensure_boundary_frame_cache(
        source,
        output_hint=destination,
        executable=executable,
    )
    return _atomic_materialize_cached_file(
        tail_cache,
        destination,
        expected_sha256=str(manifest["artifacts"]["tail"]["sha256"]),
    )


def media_manifests(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    return [describe_media(path).to_dict() for path in paths]


__all__ = [
    "MediaDescriptor",
    "MediaInput",
    "atomic_write_bytes",
    "content_sha256",
    "content_block",
    "decode_text_document",
    "data_uri",
    "describe_media",
    "detect_mime",
    "extract_audio",
    "extract_video_frames",
    "extract_video_tail_frame",
    "download_url",
    "ffprobe",
    "gemini_inline_part",
    "media_manifests",
    "media_manifest",
    "materialize_asset_selector",
    "openai_content_block",
    "require_file",
    "selector_catalog",
    "sha256_file",
    "strip_video_audio",
]
