from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from build_reel_bench import sha256_file, write_json


BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "reel_bench"
SPLITS = {"subset30": "subset30_manifest.json", "full": "full_manifest.json"}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def locked_json(path: Path, value: dict[str, Any]) -> None:
    normalized = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    if path.exists():
        if read_json(path) != normalized:
            raise ValueError(f"immutable configuration differs: {path}; use a new output directory")
    else:
        write_json(path, normalized)


def local_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"benchmark path escapes root: {relative}")
    return path


def load_suite(root: Path, split: str, *, verify_assets: bool = True) -> dict[str, Any]:
    root = root.resolve()
    release = read_json(root / "release.json")
    if release["release_id"] != digest({key: value for key, value in release.items() if key != "release_id"}):
        raise ValueError("release fingerprint mismatch")
    for name, expected in release["manifests"].items():
        if sha256_file(root / SPLITS[name]) != expected:
            raise ValueError(f"frozen {name} manifest changed")
    manifest = read_json(root / SPLITS[split])
    records = manifest["records"]
    if len(records) != manifest["num_stories"] or len({row["story_id"] for row in records}) != len(records):
        raise ValueError("invalid split membership")
    checked = set()
    for row in records:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", row["story_id"]):
            raise ValueError("unsafe story ID")
        path = local_path(root, row["relative_path"])
        if sha256_file(path) != row["case_sha256"]:
            raise ValueError(f"frozen input changed: {row['story_id']}")
        case = read_json(path)
        if case["story_id"] != row["story_id"] or len(case["prompts"]) != row["num_shots"]:
            raise ValueError("case identity or shot count differs from release")
        for asset in row["assets"]:
            asset_path = local_path(root, asset["path"])
            if verify_assets and asset_path not in checked:
                if sha256_file(asset_path) != asset["sha256"]:
                    raise ValueError(f"frozen asset changed: {asset_path}")
                checked.add(asset_path)
    return {"root": root, "release": release, "manifest": manifest, "split": split,
            "split_sha256": release["manifests"][split], "records": records}


def case_for(suite: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return read_json(local_path(suite["root"], row["relative_path"]))


def shot_seed(base_seed: int, story_id: str, shot_index: int) -> int:
    return int(digest([base_seed, story_id, shot_index])[:8], 16) % (2**31 - 1)


def uploads_for(suite: dict[str, Any], row: dict[str, Any], shot_index: int, *, only_new: bool = False) -> list[dict[str, Any]]:
    case = case_for(suite, row)
    uploads = []
    for key, items in case["uploaded_assets_by_shot"].items():
        arrival = int(key)
        if arrival > shot_index or (only_new and arrival != shot_index):
            continue
        for item in items:
            upload = dict(item)
            locator = next(field for field in ("image_path", "audio_path", "video_path", "path") if upload.get(field))
            path = (local_path(suite["root"], row["relative_path"]).parent / upload[locator]).resolve()
            if not path.is_relative_to(suite["root"]):
                raise ValueError("upload escapes benchmark root")
            upload[locator] = str(path)
            upload["available_from_shot"] = arrival
            if upload.get("declared_asset_type") == "voice_reference":
                upload["character_declarations"] = case.get("metadata", {})
            uploads.append(upload)
    return uploads


def probe_video(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"video missing: {path}")
    process = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,avg_frame_rate,duration",
         "-of", "json", str(path)], capture_output=True, text=True, timeout=60, check=True,
    )
    value = json.loads(process.stdout)
    streams = value.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("no video stream")
    duration = float(video.get("duration") or value.get("format", {}).get("duration") or 0)
    if not 0 < duration < 3600 or min(video.get("width", 0), video.get("height", 0)) < 16:
        raise ValueError("invalid video duration or dimensions")
    return {"duration": duration, "width": video["width"], "height": video["height"],
            "frame_rate": video.get("avg_frame_rate"), "has_audio_stream": any(stream.get("codec_type") == "audio" for stream in streams)}


def result_header(suite: dict[str, Any], model: dict[str, Any], base_seed: int, settings: dict[str, Any]) -> dict[str, Any]:
    if not model.get("name") or not model.get("revision"):
        raise ValueError("model.name and model.revision (checkpoint/revision identity) are required")
    return {"schema_version": "reel_bench.results.v1", "release_id": suite["release"]["release_id"],
            "split": suite["split"], "split_sha256": suite["split_sha256"], "model": model,
            "base_seed": base_seed, "generation_settings": settings}


def load_results(path: Path, suite: dict[str, Any]) -> dict[str, Any]:
    result = read_json(path)
    if result.get("schema_version") != "reel_bench.results.v1":
        raise ValueError("unsupported results schema")
    if result.get("release_id") != suite["release"]["release_id"] or result.get("split_sha256") != suite["split_sha256"]:
        raise ValueError("results belong to a different benchmark/split")
    expected_ids = {row["story_id"] for row in suite["records"]}
    if set(result["stories"]) != expected_ids:
        raise ValueError("results must explicitly account for every story in the split")
    for row in suite["records"]:
        shots = result["stories"][row["story_id"]]
        if len(shots) != row["num_shots"] or [shot["shot_index"] for shot in shots] != list(range(row["num_shots"])):
            raise ValueError("wrong shot count/order")
        for shot in shots:
            if shot.get("seed") != shot_seed(result["base_seed"], row["story_id"], shot["shot_index"]):
                raise ValueError("result seed does not match the declared seed rule")
            if shot["status"] not in {"complete", "failed", "blocked", "pending", "indeterminate"}:
                raise ValueError("invalid shot status")
            if shot["status"] == "complete":
                video = Path(shot["video_path"])
                if not video.is_absolute() or sha256_file(video) != shot["video_sha256"]:
                    raise ValueError("completed video missing or changed")
    return result
