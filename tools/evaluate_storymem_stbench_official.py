#!/usr/bin/env python3
"""Evaluate StoryMem ST-Bench videos with paper-aligned non-VLM metrics.

The five primary metrics follow StoryMem Sec. 4.3:

* LAION/VBench aesthetic quality (OpenAI CLIP ViT-L/14 + linear head, /10)
* ViCLIP global prompt following (whole story vs. story overview)
* ViCLIP single-shot prompt following
* ViCLIP all-pairs cross-shot consistency
* ViCLIP Top-10 cross-shot consistency, with pairs selected by prompt similarity

The input is a normalized ST-Bench manifest plus a JSON mapping from story IDs
to ordered per-shot MP4 paths.  A direct Echo result JSON containing a top-level
``shot_videos`` list is also accepted when exactly one ``--story-id`` is given.

StoryMem did not publish an evaluator, the exact ViCLIP checkpoint, its temporal
sampling implementation, or its cross-story aggregation rule.  This file is
therefore a paper-aligned, auditable reimplementation rather than an official
evaluator.  It intentionally does not produce a composite score and never
replaces missing shots with zero.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA_VERSION = "storymem_stbench_paper_aligned_metrics_v1"
STORYMEM_PAPER = "https://arxiv.org/abs/2512.19539"
STORYMEM_REPOSITORY = "https://github.com/Kevin-thu/StoryMem"
STORYMEM_SOURCE_COMMIT = "052c68d2627a22a95f79dbb4d3376cc30c96f1a3"
CANONICAL_MANIFEST_SHA256 = "fe838eea3d02a9a10eb0d1b4341c19ee09dd4bb51ae561798fa63bb327c6a424"
VBENCH_REPOSITORY = "https://github.com/Vchitect/VBench"
VBENCH_SOURCE_COMMIT = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
CANONICAL_STORY_COUNT = 30
CANONICAL_SHOT_COUNT = 300
PRIMARY_METRICS = (
    "aesthetic_quality",
    "prompt_following_global",
    "prompt_following_single_shot",
    "cross_shot_consistency_overall",
    "cross_shot_consistency_top10",
)


@dataclass(frozen=True)
class StorySpec:
    story_id: str
    title: str
    overview: str
    prompts: tuple[str, ...]
    cut_flags: tuple[bool | None, ...]
    prompt_path: Path


def _json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _finite_mean(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else None


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denom)


def middle_indices(length: int, count: int) -> list[int]:
    """Match VBench's deterministic ``sample='middle'`` frame selection."""
    if length <= 0:
        raise ValueError("length must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    actual = min(count, length)
    intervals = np.linspace(0, length, num=actual + 1).astype(int)
    indices = [(int(start) + int(intervals[i + 1]) - 1) // 2 for i, start in enumerate(intervals[:-1])]
    if len(indices) < count:
        indices.extend([indices[-1]] * (count - len(indices)))
    return indices


def flatten_cut_flags(prompt_data: dict[str, Any], num_prompts: int) -> tuple[bool | None, ...]:
    """Flatten StoryMem scene-local cut flags to prompt order.

    ``cut[i]`` states whether shot ``i`` starts with a hard cut.  The first flag
    has no preceding boundary and is retained only for auditability.
    """
    result: list[bool | None] = []
    scenes = ((prompt_data.get("benchmark_metadata") or {}).get("raw_scenes") or [])
    for scene in scenes:
        scene_prompts = scene.get("video_prompts") or []
        cuts = scene.get("cut") or []
        for index in range(len(scene_prompts)):
            result.append(bool(cuts[index]) if index < len(cuts) else None)
    if len(result) < num_prompts:
        result.extend([None] * (num_prompts - len(result)))
    return tuple(result[:num_prompts])


def rank_top_prompt_pairs(text_features: Sequence[np.ndarray], top_k: int = 10) -> list[dict[str, Any]]:
    """Select unordered shot pairs by descending prompt-feature cosine."""
    pairs: list[dict[str, Any]] = []
    for left in range(len(text_features)):
        for right in range(left + 1, len(text_features)):
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "prompt_similarity": _cosine(text_features[left], text_features[right]),
                }
            )
    pairs.sort(key=lambda item: (-item["prompt_similarity"], item["left"], item["right"]))
    return pairs[: min(max(0, top_k), len(pairs))]


def bootstrap_mean_ci(values: Iterable[float | None], *, seed: int, samples: int) -> dict[str, Any]:
    """Return a story-clustered percentile CI; one story cannot define a CI."""
    array = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    result: dict[str, Any] = {
        "mean": float(array.mean()) if array.size else None,
        "n_stories": int(array.size),
        "ci95": None,
        "ci_method": "story-level percentile bootstrap",
    }
    if array.size < 2:
        result["ci_status"] = "insufficient_stories"
        return result
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, array.size, size=(samples, array.size))
    draws = array[draw_indices].mean(axis=1)
    result["ci95"] = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
    result["bootstrap_samples"] = int(samples)
    result["ci_status"] = "ok"
    return result


def _resolve_reference(reference: str | Path, anchors: Sequence[Path]) -> Path:
    path = Path(reference).expanduser()
    if path.is_absolute():
        return path.resolve()
    for anchor in anchors:
        candidate = (anchor / path).resolve()
        if candidate.exists():
            return candidate
    return (anchors[0] / path).resolve()


def _manifest_anchors(manifest: Path) -> list[Path]:
    anchors = [manifest.parent]
    anchors.extend(manifest.parents)
    anchors.extend([Path.cwd(), Path(__file__).resolve().parents[1]])
    unique: list[Path] = []
    for anchor in anchors:
        resolved = anchor.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def load_story_specs(manifest: Path, requested_story_ids: Sequence[str]) -> list[StorySpec]:
    data = _json_load(manifest)
    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, list):
        raise ValueError(f"Manifest must contain a records list: {manifest}")
    requested = set(requested_story_ids)
    anchors = _manifest_anchors(manifest)
    stories: list[StorySpec] = []
    for record in records:
        story_id = str(record.get("story_id") or Path(str(record.get("path", ""))).stem)
        if requested and story_id not in requested:
            continue
        prompt_path = _resolve_reference(str(record["path"]), anchors)
        prompt_data = _json_load(prompt_path)
        prompts = tuple(str(value).strip() for value in prompt_data.get("prompts") or [] if str(value).strip())
        if not prompts:
            raise ValueError(f"No prompts in {prompt_path}")
        stories.append(
            StorySpec(
                story_id=story_id,
                title=str(prompt_data.get("title") or record.get("title") or story_id),
                overview=str(prompt_data.get("story_overview") or "").strip(),
                prompts=prompts,
                cut_flags=flatten_cut_flags(prompt_data, len(prompts)),
                prompt_path=prompt_path,
            )
        )
    if requested:
        missing = sorted(requested - {story.story_id for story in stories})
        if missing:
            raise KeyError(f"Requested stories absent from manifest: {missing}")
    if not stories:
        raise ValueError("No stories selected")
    return stories


def _coerce_video_value(value: Any, *, map_path: Path) -> list[Path | None]:
    if isinstance(value, dict):
        for key in ("shot_videos", "videos", "shots"):
            if key in value:
                return _coerce_video_value(value[key], map_path=map_path)
        raise ValueError(f"Video-map object has no shot_videos/videos/shots field: {value.keys()}")
    if not isinstance(value, list):
        raise TypeError(f"Expected an ordered video list, got {type(value).__name__}")
    paths: list[Path | None] = []
    for item in value:
        if item is None or str(item).strip() == "":
            paths.append(None)
            continue
        candidate = Path(str(item)).expanduser()
        if not candidate.is_absolute():
            candidate = map_path.parent / candidate
        paths.append(candidate.resolve())
    return paths


def load_video_map(video_map_path: Path, selected_story_ids: Sequence[str]) -> dict[str, list[Path | None]]:
    """Accept common mapping layouts and a single Echo direct-result JSON."""
    data = _json_load(video_map_path)
    selected = list(selected_story_ids)
    if isinstance(data, dict) and "shot_videos" in data:
        if len(selected) != 1:
            raise ValueError("A top-level shot_videos result requires exactly one --story-id")
        return {selected[0]: _coerce_video_value(data["shot_videos"], map_path=video_map_path)}
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        result: dict[str, list[Path | None]] = {}
        for record in data["records"]:
            story_id = str(record.get("story_id") or "")
            if not story_id:
                raise ValueError("Every video-map record needs story_id")
            result[story_id] = _coerce_video_value(record, map_path=video_map_path)
        return result
    if isinstance(data, dict) and isinstance(data.get("stories"), dict):
        data = data["stories"]
    if not isinstance(data, dict):
        raise ValueError("Video map must be an object, records list, or direct Echo result")
    return {str(key): _coerce_video_value(value, map_path=video_map_path) for key, value in data.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_identity(path: Path, *, hash_large: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result = {"path": str(path.resolve()), "size_bytes": int(stat.st_size)}
    if hash_large or stat.st_size <= 16 * 1024 * 1024:
        result["sha256"] = _sha256(path)
    return result


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _to_json_number(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def inspect_video(path: Path) -> dict[str, Any]:
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    frame_count = len(reader)
    if frame_count <= 0:
        raise ValueError("video has no decodable frames")
    first = reader[0].asnumpy()
    return {
        "frame_count": int(frame_count),
        "fps": float(reader.get_avg_fps()),
        "width": int(first.shape[1]),
        "height": int(first.shape[0]),
    }


def read_video_indices(path: Path, indices: Sequence[int]) -> list[np.ndarray]:
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    if not indices:
        return []
    frames = reader.get_batch([int(index) for index in indices]).asnumpy()
    return [np.asarray(frame, dtype=np.uint8) for frame in frames]


class AestheticScorer:
    """VBench LAION aesthetic scorer with the exact OpenAI ViT-L/14 backbone."""

    def __init__(self, clip_path: Path, head_path: Path, device: str, batch_size: int) -> None:
        import clip
        import torch
        import torch.nn as nn
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.clip_model, _ = clip.load(str(clip_path), device=self.device, jit=False)
        self.clip_model.eval()
        state = torch.load(head_path, map_location="cpu", weights_only=True)
        if set(state) == {"weight", "bias"}:
            head: nn.Module = nn.Linear(768, 1)
            self.head_kind = "VBench sa_0_4 ViT-L/14 linear head"
        elif "layers.0.weight" in state:
            head = nn.Sequential(
                nn.Linear(768, 1024),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.Dropout(0.1),
                nn.Linear(64, 16),
                nn.Linear(16, 1),
            )
            # Preserve the checkpoint's published ``layers.*`` key names.
            head = nn.ModuleDict({"layers": head})
            self.head_kind = "LAION improved-aesthetic-predictor MLP"
        else:
            raise ValueError(f"Unsupported aesthetic head state dict: {sorted(state)[:10]}")
        head.load_state_dict(state)
        self.head = head.to(self.device).float().eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=False),
                transforms.CenterCrop(224),
                transforms.Lambda(lambda tensor: tensor.float().div(255.0)),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def _encode(self, frames: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        # VBench applies Resize/CenterCrop once to a TCHW tensor.  Keep that
        # exact batched path rather than transforming frames one by one.
        tensors = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2)
        tensors = self.transform(tensors).to(self.device)
        with torch.inference_mode():
            features = self.clip_model.encode_image(tensors).float()
            features = torch.nn.functional.normalize(features, dim=-1, p=2)
            raw_scores = self.head(features).reshape(-1)
        return features.cpu().numpy(), raw_scores.cpu().numpy()

    def score_video(self, path: Path, frame_count: int, frame_stride: int) -> dict[str, Any]:
        from decord import VideoReader, cpu

        if frame_stride <= 0:
            raise ValueError("aesthetic frame stride must be positive")
        indices = list(range(0, frame_count, frame_stride))
        if indices[-1] != frame_count - 1:
            indices.append(frame_count - 1)
        # VBench decodes and preprocesses the complete video before splitting
        # it into inference batches.  Reproduce that order exactly: applying a
        # resize to individual chunks can select a different CUDA kernel.
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        decoded = reader.get_batch(indices).asnumpy()
        # ``vbench.utils.load_video`` constructs ``torch.Tensor(buffer)``,
        # which converts uint8 to float32 *before* Resize.  The order changes
        # bicubic interpolation slightly, so preserve it for score parity.
        transformed = self.torch.from_numpy(decoded).permute(0, 3, 1, 2).float()
        transformed = self.transform(transformed)
        all_scores: list[float] = []
        first_feature: np.ndarray | None = None
        last_feature: np.ndarray | None = None
        for offset in range(0, len(indices), self.batch_size):
            tensors = transformed[offset : offset + self.batch_size].to(self.device)
            with self.torch.inference_mode():
                encoded = self.clip_model.encode_image(tensors).float()
                encoded = self.torch.nn.functional.normalize(encoded, dim=-1, p=2)
                scores_tensor = self.head(encoded).reshape(-1)
            features = encoded.cpu().numpy()
            scores = scores_tensor.cpu().numpy()
            if first_feature is None:
                first_feature = features[0]
            last_feature = features[-1]
            all_scores.extend(float(value) for value in scores)
        return {
            "raw_mean": float(np.mean(all_scores)),
            "normalized_mean": float(np.mean(all_scores) / 10.0),
            "evaluated_frames": len(indices),
            "frame_stride": frame_stride,
            "first_feature": first_feature,
            "last_feature": last_feature,
        }


class ViClipScorer:
    """Thin adapter over VBench's vendored ViCLIP implementation."""

    def __init__(self, vbench_root: Path, checkpoint: Path, tokenizer_path: Path, device: str) -> None:
        import torch
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        os.environ.setdefault("VBENCH_CACHE_DIR", str(tokenizer_path.parent.parent))
        sys.path.insert(0, str(vbench_root))
        try:
            from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
            from vbench.third_party.ViCLIP.viclip import ViCLIP
        except Exception as exc:  # pragma: no cover - dependency-specific error path
            raise RuntimeError(f"Cannot import ViCLIP from VBench root {vbench_root}: {exc}") from exc
        self.torch = torch
        self.device = torch.device(device)
        tokenizer = SimpleTokenizer(str(tokenizer_path))
        self.model = ViCLIP(tokenizer=tokenizer, pretrain=str(checkpoint), freeze_text=True).to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=False),
                transforms.CenterCrop(224),
                transforms.Lambda(lambda tensor: tensor.float().div(255.0)),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.text_cache: dict[str, np.ndarray] = {}

    def text_feature(self, text: str) -> np.ndarray:
        if text not in self.text_cache:
            torch = self.torch
            with torch.inference_mode():
                feature = self.model.encode_text(text).float()
                feature = torch.nn.functional.normalize(feature, dim=-1)
            self.text_cache[text] = feature[0].cpu().numpy()
        return self.text_cache[text]

    def video_feature(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        if len(frames) != 8:
            raise ValueError(f"ViCLIP requires exactly 8 sampled frames, got {len(frames)}")
        torch = self.torch
        tensor = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2)
        tensor = self.transform(tensor).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            feature = self.model.encode_vision(tensor, test=True).float()
            feature = torch.nn.functional.normalize(feature, dim=-1)
        return feature[0].cpu().numpy()


def _read_global_middle_frames(
    paths: Sequence[Path], frame_counts: Sequence[int], count: int = 8
) -> tuple[list[np.ndarray], list[dict[str, int]]]:
    total = sum(frame_counts)
    global_indices = middle_indices(total, count)
    cumulative = np.cumsum([0, *frame_counts])
    locations: list[tuple[int, int]] = []
    for global_index in global_indices:
        shot_index = int(np.searchsorted(cumulative[1:], global_index, side="right"))
        local_index = int(global_index - cumulative[shot_index])
        locations.append((shot_index, local_index))
    frames: list[np.ndarray | None] = [None] * len(locations)
    for shot_index in sorted({location[0] for location in locations}):
        positions = [position for position, location in enumerate(locations) if location[0] == shot_index]
        local_indices = [locations[position][1] for position in positions]
        decoded = read_video_indices(paths[shot_index], local_indices)
        for position, frame in zip(positions, decoded):
            frames[position] = frame
    return [frame for frame in frames if frame is not None], [
        {"global_frame_index": int(global_index), "shot_index": shot, "shot_frame_index": local}
        for global_index, (shot, local) in zip(global_indices, locations)
    ]


def _metric_pair_means(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def select(kind: str) -> list[float]:
        values: list[float] = []
        for pair in pairs:
            if kind == "cut" and pair.get("cut") is not True:
                continue
            if kind == "non_cut" and pair.get("cut") is not False:
                continue
            value = pair.get("video_similarity")
            if value is not None:
                values.append(float(value))
        return values

    return {
        kind: {"mean": _finite_mean(select(kind)), "n_pairs": len(select(kind))}
        for kind in ("all", "cut", "non_cut")
    }


def evaluate_story(
    story: StorySpec,
    mapped_paths: Sequence[Path | None],
    aesthetic: AestheticScorer,
    viclip: ViClipScorer,
    aesthetic_frame_stride: int,
    top_k: int,
) -> dict[str, Any]:
    expected = len(story.prompts)
    paths = list(mapped_paths[:expected]) + [None] * max(0, expected - len(mapped_paths))
    shot_rows: list[dict[str, Any]] = []
    valid_paths: list[Path] = []
    video_info: list[dict[str, Any]] = []
    for index, (prompt, cut, path) in enumerate(zip(story.prompts, story.cut_flags, paths)):
        row: dict[str, Any] = {
            "shot_index": index,
            "prompt": prompt,
            "starts_with_hard_cut": cut,
            "video": str(path) if path else None,
            "status": "missing",
        }
        if path is not None and path.is_file():
            try:
                metadata = inspect_video(path)
                row.update({"status": "ok", "video_metadata": metadata})
                valid_paths.append(path)
                video_info.append(metadata)
            except Exception as exc:  # noqa: BLE001 - persist exact decode failure per shot
                row.update({"status": "decode_error", "error": repr(exc)})
        shot_rows.append(row)

    completed = sum(row["status"] == "ok" for row in shot_rows)
    completion_rate = completed / expected
    complete = completed == expected and len(mapped_paths) == expected
    result: dict[str, Any] = {
        "story_id": story.story_id,
        "title": story.title,
        "prompt_path": str(story.prompt_path),
        "expected_shots": expected,
        "completed_shots": completed,
        "completion_rate": completion_rate,
        "paper_aligned_metric_eligible": complete,
        "extra_mapped_videos": max(0, len(mapped_paths) - expected),
        "metrics": {key: None for key in PRIMARY_METRICS},
        "diagnostics": {},
        "shots": shot_rows,
        "pairs": [],
    }
    if not complete:
        result["invalid_reason"] = (
            "paper-aligned StoryMem metrics require one decodable MP4 per expected shot"
        )
        return result

    aesthetic_frame_scores: list[float] = []
    boundary_features: list[tuple[np.ndarray, np.ndarray]] = []
    for row, path, metadata in zip(shot_rows, valid_paths, video_info):
        scored = aesthetic.score_video(path, metadata["frame_count"], aesthetic_frame_stride)
        row["aesthetic_quality"] = scored["normalized_mean"]
        row["aesthetic_raw_1_to_10"] = scored["raw_mean"]
        row["aesthetic_evaluated_frames"] = scored["evaluated_frames"]
        aesthetic_frame_scores.extend([scored["normalized_mean"]] * scored["evaluated_frames"])
        boundary_features.append((scored["first_feature"], scored["last_feature"]))

    # All shots have the same generated length in the official setup.  We retain
    # exact frame weighting so this also matches concatenated-video evaluation
    # when custom shot lengths differ.
    weighted_aesthetic_sum = sum(
        float(row["aesthetic_quality"]) * int(row["aesthetic_evaluated_frames"]) for row in shot_rows
    )
    aesthetic_frame_count = sum(int(row["aesthetic_evaluated_frames"]) for row in shot_rows)
    result["metrics"]["aesthetic_quality"] = weighted_aesthetic_sum / aesthetic_frame_count

    video_features: list[np.ndarray] = []
    text_features: list[np.ndarray] = []
    for row, path, metadata, prompt in zip(shot_rows, valid_paths, video_info, story.prompts):
        indices = middle_indices(metadata["frame_count"], 8)
        video_feature = viclip.video_feature(read_video_indices(path, indices))
        text_feature = viclip.text_feature(prompt)
        video_features.append(video_feature)
        text_features.append(text_feature)
        row["viclip_frame_indices"] = indices
        row["prompt_following"] = _cosine(video_feature, text_feature)

    global_frames, global_locations = _read_global_middle_frames(
        valid_paths, [int(metadata["frame_count"]) for metadata in video_info], count=8
    )
    global_video_feature = viclip.video_feature(global_frames)
    overview_feature = viclip.text_feature(story.overview)
    result["global_viclip_frame_locations"] = global_locations
    result["metrics"]["prompt_following_global"] = _cosine(global_video_feature, overview_feature)
    result["metrics"]["prompt_following_single_shot"] = _finite_mean(
        row["prompt_following"] for row in shot_rows
    )

    all_pairs: list[dict[str, Any]] = []
    pair_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for left in range(expected):
        for right in range(left + 1, expected):
            pair = {
                "left": left,
                "right": right,
                "video_similarity": _cosine(video_features[left], video_features[right]),
                "prompt_similarity": _cosine(text_features[left], text_features[right]),
                "adjacent": right == left + 1,
                "cut": story.cut_flags[right] if right == left + 1 else None,
                "selected_top10": False,
            }
            all_pairs.append(pair)
            pair_lookup[(left, right)] = pair
    top_pairs = rank_top_prompt_pairs(text_features, top_k=top_k)
    for selected in top_pairs:
        pair_lookup[(selected["left"], selected["right"])]["selected_top10"] = True
    result["pairs"] = all_pairs
    result["metrics"]["cross_shot_consistency_overall"] = _finite_mean(
        pair["video_similarity"] for pair in all_pairs
    )
    result["metrics"]["cross_shot_consistency_top10"] = _finite_mean(
        pair_lookup[(pair["left"], pair["right"])]["video_similarity"] for pair in top_pairs
    )

    adjacent_pairs = [pair for pair in all_pairs if pair["adjacent"]]
    result["diagnostics"]["adjacent_cross_shot_viclip_consistency"] = _metric_pair_means(adjacent_pairs)
    boundary_pairs: list[dict[str, Any]] = []
    for right in range(1, expected):
        boundary_pairs.append(
            {
                "left": right - 1,
                "right": right,
                "cut": story.cut_flags[right],
                "video_similarity": _cosine(boundary_features[right - 1][1], boundary_features[right][0]),
            }
        )
    result["diagnostics"]["boundary_continuity_clip_image_cosine"] = _metric_pair_means(boundary_pairs)
    result["diagnostics"]["boundary_pairs"] = boundary_pairs
    return result


def audit_common_media_protocol(
    stories: Sequence[dict[str, Any]],
    *,
    expected_width: int,
    expected_height: int,
    expected_frames: int,
    expected_fps: float,
    fps_tolerance: float,
) -> dict[str, Any]:
    """Verify every mapped shot uses the frozen cross-method media contract."""
    mismatches: list[dict[str, Any]] = []
    observed: set[tuple[int, int, int, float]] = set()
    checked = 0
    for story in stories:
        for shot in story.get("shots") or []:
            metadata = shot.get("video_metadata")
            if shot.get("status") != "ok" or not isinstance(metadata, dict):
                continue
            checked += 1
            actual = {
                "width": int(metadata["width"]),
                "height": int(metadata["height"]),
                "frame_count": int(metadata["frame_count"]),
                "fps": float(metadata["fps"]),
            }
            observed.add(
                (
                    actual["width"],
                    actual["height"],
                    actual["frame_count"],
                    round(actual["fps"], 6),
                )
            )
            if (
                actual["width"] != expected_width
                or actual["height"] != expected_height
                or actual["frame_count"] != expected_frames
                or abs(actual["fps"] - expected_fps) > fps_tolerance
            ):
                mismatches.append(
                    {
                        "story_id": story.get("story_id"),
                        "shot_index": shot.get("shot_index"),
                        "actual": actual,
                    }
                )
    expected_shots = sum(int(story.get("expected_shots", 0)) for story in stories)
    return {
        "valid": checked == expected_shots and not mismatches,
        "expected": {
            "width": expected_width,
            "height": expected_height,
            "frame_count": expected_frames,
            "fps": expected_fps,
            "fps_tolerance": fps_tolerance,
        },
        "shots_checked": checked,
        "observed_configs": [
            {"width": width, "height": height, "frame_count": frames, "fps": fps}
            for width, height, frames, fps in sorted(observed)
        ],
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def summarize(
    stories: Sequence[dict[str, Any]],
    *,
    seed: int,
    bootstrap_samples: int,
    top_k: int = 10,
    aesthetic_frame_stride: int = 1,
    selection_unclipped: bool = False,
    common_media_protocol_valid: bool = False,
    canonical_manifest_valid: bool = False,
) -> dict[str, Any]:
    complete = [story for story in stories if story["paper_aligned_metric_eligible"]]
    stories_requested = len(stories)
    shots_expected = sum(int(story["expected_shots"]) for story in stories)
    shots_completed = sum(int(story["completed_shots"]) for story in stories)
    validity_checks = {
        "canonical_30_stories": stories_requested == CANONICAL_STORY_COUNT,
        "canonical_300_shots": shots_expected == CANONICAL_SHOT_COUNT,
        "canonical_manifest_sha256": bool(canonical_manifest_valid),
        "all_shots_decodable": len(complete) == stories_requested
        and shots_completed == shots_expected,
        "top_k_is_10": top_k == 10,
        "aesthetic_frame_stride_is_1": aesthetic_frame_stride == 1,
        "selection_unclipped": bool(selection_unclipped),
        "common_media_protocol_valid": bool(common_media_protocol_valid),
    }
    summary: dict[str, Any] = {
        "stories_requested": stories_requested,
        "stories_complete": len(complete),
        "shots_expected": shots_expected,
        "shots_completed": shots_completed,
        "completion_rate": shots_completed / shots_expected,
        "paper_aligned_protocol_valid": all(validity_checks.values()),
        "paper_aligned_protocol_checks": validity_checks,
        "paper_aligned_metrics_story_macro": {},
        "paper_aligned_metrics_micro": {},
    }
    for metric_index, metric in enumerate(PRIMARY_METRICS):
        summary["paper_aligned_metrics_story_macro"][metric] = bootstrap_mean_ci(
            [story["metrics"].get(metric) for story in complete],
            seed=seed + metric_index,
            samples=bootstrap_samples,
        )

    # Micro views are included because the StoryMem paper does not state how
    # variable-length stories are weighted.  They are never silently conflated
    # with the story-macro estimates above.
    shot_rows = [shot for story in complete for shot in story["shots"]]
    all_pairs = [pair for story in complete for pair in story["pairs"]]
    top_pairs = [pair for pair in all_pairs if pair["selected_top10"]]
    aesthetic_weight = sum(int(shot["aesthetic_evaluated_frames"]) for shot in shot_rows)
    summary["paper_aligned_metrics_micro"] = {
        "aesthetic_quality": (
            sum(
                float(shot["aesthetic_quality"]) * int(shot["aesthetic_evaluated_frames"])
                for shot in shot_rows
            )
            / aesthetic_weight
            if aesthetic_weight
            else None
        ),
        "prompt_following_global": _finite_mean(
            story["metrics"]["prompt_following_global"] for story in complete
        ),
        "prompt_following_single_shot": _finite_mean(shot.get("prompt_following") for shot in shot_rows),
        "cross_shot_consistency_overall": _finite_mean(pair["video_similarity"] for pair in all_pairs),
        "cross_shot_consistency_top10": _finite_mean(pair["video_similarity"] for pair in top_pairs),
        "counts": {
            "aesthetic_frames": aesthetic_weight,
            "shots": len(shot_rows),
            "all_pairs": len(all_pairs),
            "top_pairs": len(top_pairs),
        },
    }
    return summary


def default_vbench_root() -> Path:
    candidates = [
        os.environ.get("STORYMEM_VBENCH_ROOT"),
        str(Path(__file__).resolve().parents[1] / "third_party" / "VBench"),
    ]
    for value in candidates:
        if value and (Path(value) / "vbench/third_party/ViCLIP/viclip.py").is_file():
            return Path(value)
    return Path("/path/to/VBench")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--story-id", action="append", default=[], help="Repeat to select stories.")
    parser.add_argument("--limit", type=int, default=0, help="Debug-only story limit after filtering.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vbench-root", type=Path, default=default_vbench_root())
    parser.add_argument(
        "--viclip-checkpoint",
        type=Path,
        default=Path("models/ViCLIP/ViClip-InternVid-10M-FLT.pth"),
    )
    parser.add_argument(
        "--viclip-tokenizer",
        type=Path,
        default=Path("models/ViCLIP/bpe_simple_vocab_16e6.txt.gz"),
    )
    parser.add_argument(
        "--aesthetic-clip",
        type=Path,
        default=Path("models/CLIP/ViT-L-14.pt"),
    )
    parser.add_argument(
        "--aesthetic-head",
        type=Path,
        default=Path("models/LAION_Aesthetic/sa_0_4_vit_l_14_linear.pth"),
    )
    parser.add_argument("--aesthetic-batch-size", type=int, default=32)
    parser.add_argument(
        "--aesthetic-frame-stride",
        type=int,
        default=1,
        help="1 is the official VBench all-frame protocol; values >1 are debug-only.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--expected-width", type=int, default=832)
    parser.add_argument("--expected-height", type=int, default=480)
    parser.add_argument("--expected-frames", type=int, default=121)
    parser.add_argument("--expected-fps", type=float, default=25.0)
    parser.add_argument("--fps-tolerance", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260918)
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.manifest = args.manifest.resolve()
    args.video_map = args.video_map.resolve()
    args.output = args.output.resolve()
    if args.top_k != 10:
        print("WARNING: --top-k != 10 is outside the paper-aligned protocol", file=sys.stderr)
    if args.aesthetic_frame_stride != 1:
        print(
            "WARNING: --aesthetic-frame-stride > 1 is debug-only and not VBench scoring",
            file=sys.stderr,
        )
    if min(args.expected_width, args.expected_height, args.expected_frames) < 1:
        raise ValueError("expected media dimensions/frame count must be positive")
    if args.expected_fps <= 0 or args.fps_tolerance < 0:
        raise ValueError("expected fps must be positive and tolerance non-negative")
    stories = load_story_specs(args.manifest, args.story_id)
    if args.limit > 0:
        stories = stories[: args.limit]
    video_map = load_video_map(args.video_map, [story.story_id for story in stories])

    required_files = [
        args.viclip_checkpoint,
        args.viclip_tokenizer,
        args.aesthetic_clip,
        args.aesthetic_head,
        args.vbench_root / "vbench/third_party/ViCLIP/viclip.py",
    ]
    missing_dependencies = [str(path) for path in required_files if not path.is_file()]
    if missing_dependencies:
        raise FileNotFoundError(f"Missing evaluation dependencies: {missing_dependencies}")
    print(json.dumps({"event": "load_aesthetic", "device": args.device}, ensure_ascii=False), flush=True)
    aesthetic = AestheticScorer(
        args.aesthetic_clip, args.aesthetic_head, args.device, args.aesthetic_batch_size
    )
    print(json.dumps({"event": "load_viclip", "device": args.device}, ensure_ascii=False), flush=True)
    viclip = ViClipScorer(args.vbench_root, args.viclip_checkpoint, args.viclip_tokenizer, args.device)

    results: list[dict[str, Any]] = []
    for story in stories:
        print(json.dumps({"event": "story_start", "story_id": story.story_id}, ensure_ascii=False), flush=True)
        result = evaluate_story(
            story,
            video_map.get(story.story_id, []),
            aesthetic,
            viclip,
            args.aesthetic_frame_stride,
            args.top_k,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "event": "story_done",
                    "story_id": story.story_id,
                    "completion_rate": result["completion_rate"],
                    "metrics": result["metrics"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    media_protocol = audit_common_media_protocol(
        results,
        expected_width=args.expected_width,
        expected_height=args.expected_height,
        expected_frames=args.expected_frames,
        expected_fps=args.expected_fps,
        fps_tolerance=args.fps_tolerance,
    )
    summary = summarize(
        results,
        seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        top_k=args.top_k,
        aesthetic_frame_stride=args.aesthetic_frame_stride,
        selection_unclipped=not args.story_id and args.limit == 0,
        common_media_protocol_valid=bool(media_protocol["valid"]),
        canonical_manifest_valid=_sha256(args.manifest) == CANONICAL_MANIFEST_SHA256,
    )
    script_path = Path(__file__).resolve()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "primary_source": STORYMEM_PAPER,
            "source_section": "Sec. 4.3 Quantitative Results",
            "status": "paper_aligned_reimplementation_not_official_evaluator",
            "storymem_repository": STORYMEM_REPOSITORY,
            "storymem_source_commit": STORYMEM_SOURCE_COMMIT,
            "vbench_source": VBENCH_REPOSITORY,
            "paper_aligned_primary_metrics": {
                "aesthetic_quality": "Mean LAION aesthetic score over every generated frame, divided by 10 as in VBench.",
                "prompt_following_global": "ViCLIP cosine: 8 middle-sampled frames from the concatenated story vs. story_overview.",
                "prompt_following_single_shot": "Mean ViCLIP cosine: each 8-frame shot feature vs. its shot prompt feature.",
                "cross_shot_consistency_overall": "Mean ViCLIP video-feature cosine over all unordered shot pairs within a story.",
                "cross_shot_consistency_top10": "Mean video-feature cosine for the 10 unordered pairs with highest ViCLIP prompt-feature cosine.",
            },
            "additional_diagnostics": {
                "adjacent_cross_shot_viclip_consistency": "Adjacent subset of the paper-aligned pair score, split by StoryMem cut flags.",
                "boundary_continuity_clip_image_cosine": "Cosine between previous last-frame and next first-frame OpenAI CLIP ViT-L/14 features, split by cut flags.",
            },
            "cut_semantics": "For boundary (i-1,i), cut[i]=true is a hard cut; false requests smooth continuation.",
            "aesthetic_all_frames": args.aesthetic_frame_stride == 1,
            "no_generative_vlm_judge": True,
            "no_composite_score": True,
            "unpublished_details": {
                "evaluator_code": "The StoryMem repository does not publish its quantitative evaluator.",
                "viclip_checkpoint": "The paper names ViCLIP but does not identify the exact checkpoint.",
                "temporal_sampling": "The paper does not specify the exact 8-frame sampling implementation; this evaluator uses deterministic middle-bin sampling.",
                "aggregation": "The paper does not specify story-macro versus shot/pair-micro aggregation; both are reported separately.",
            },
            "common_media_protocol": media_protocol,
            "generation_protocol_scope": {
                "this_controlled_echo_comparison": "832x480, 121 frames, 25 fps by default",
                "storymem_paper_reference": "832x480, 81 frames, 16 fps",
                "comparability": "StoryMem paper numbers are an external cross-generator reference, not a same-generation-protocol baseline.",
            },
        },
        "run": {
            "argv": sys.argv if argv is None else [str(script_path), *argv],
            "manifest": _artifact_identity(args.manifest, hash_large=True),
            "video_map": _artifact_identity(args.video_map, hash_large=True),
            "script": _artifact_identity(script_path, hash_large=True),
            "models": {
                "viclip": _artifact_identity(args.viclip_checkpoint, hash_large=True),
                "viclip_tokenizer": _artifact_identity(args.viclip_tokenizer, hash_large=True),
                "aesthetic_clip": _artifact_identity(args.aesthetic_clip, hash_large=True),
                "aesthetic_head": _artifact_identity(args.aesthetic_head, hash_large=True),
                "aesthetic_head_kind": aesthetic.head_kind,
            },
            "vbench_root": str(args.vbench_root.resolve()),
            "vbench_commit": _git_commit(args.vbench_root),
            "vbench_source_commit": VBENCH_SOURCE_COMMIT,
            "viclip_implementation": {
                name: _artifact_identity(
                    args.vbench_root / f"vbench/third_party/ViCLIP/{name}", hash_large=True
                )
                for name in ("viclip.py", "viclip_vision.py", "viclip_text.py", "simple_tokenizer.py")
            },
            "device": args.device,
            "package_versions": {
                name: _package_version(name)
                for name in ("torch", "torchvision", "decord", "numpy", "openai-clip", "timm")
            },
            "aesthetic_batch_size": args.aesthetic_batch_size,
            "aesthetic_frame_stride": args.aesthetic_frame_stride,
            "top_k": args.top_k,
            "story_filter_present": bool(args.story_id),
            "debug_limit": args.limit,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "summary": summary,
        "stories": results,
    }
    _atomic_write_json(args.output, payload)
    print(json.dumps({"event": "complete", "output": str(args.output), "summary": payload["summary"]}, ensure_ascii=False), flush=True)
    if args.fail_on_incomplete and not payload["summary"]["paper_aligned_protocol_valid"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
