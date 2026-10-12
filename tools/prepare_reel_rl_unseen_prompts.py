#!/usr/bin/env python3
"""Freeze completed Echo shots whose task prompts were absent from the SFT snapshot.

Read source assets only. Use a new output directory; family splits remain fixed.
All SFT selection splits are excluded, a stricter boundary than train-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

SCHEMA = "reel.rl-unseen-prompts.v1"
NORMALIZER = "unicode-nfkc-casefold-whitespace-v1"
DATA = Path(__file__).resolve().parents[1] / "data"
DEFAULT_SFT = DATA / "sft_snapshot" / "selection.jsonl"
LEGACY = (("storymem_echo1_style_v2_10k_20260922", "main"),
          ("reel_diverse_v3_1k_20260922", "diverse"),
          ("reel_variable_budgets_200_20260922", "budgeted"))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_bytes(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return "sha256:" + result.hexdigest()


def prompt_hash(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("empty task prompt")
    return digest_bytes(" ".join(unicodedata.normalize("NFKC", prompt).casefold().split()).encode())


def identity(row):
    return row["run_namespace"], row["story_id"], int(row["shot_index"])


def read(path):
    return json.loads(Path(path).read_text())


def read_prompt(row):
    path = row["receipt"]["trajectory_path"]
    trajectory = read(path)
    if digest_bytes(canonical(trajectory).encode()) != row["receipt"]["trajectory_sha256"]:
        raise ValueError(f"trajectory changed since receipt: {path}")
    return {**row, "rl_task_prompt_sha256": prompt_hash(trajectory["raw_user_request"])}


def story_rows(task):
    root, record, metadata, accepted, namespace = task
    story = record["story_id"]
    path = root / "trajectories/stories" / story / "story_checkpoint.json"
    if not path.exists() and accepted is None:
        return []
    checkpoint = read(path)
    if checkpoint["story_id"] != story or checkpoint["run_namespace"] != namespace:
        raise ValueError(f"checkpoint identity changed: {path}")
    family, split = metadata["family"], metadata["split"]
    if record.get("family", family) != family or record.get("split", split) != split:
        raise ValueError(f"source family/split changed: {path}")
    indices = accepted if accepted is not None else range(checkpoint["next_shot_index"])
    result = []
    for index in indices:
        receipt = checkpoint["shots"][str(index)]
        if receipt["status"] != "complete" or index >= checkpoint["next_shot_index"]:
            raise ValueError(f"incomplete accepted shot: {path}:{index}")
        recipes = metadata.get("recipes", [])
        result.append({"root": str(root), "backend": "echo15", "cohort": metadata["cohort"],
            "story_id": story, "shot_index": index, "family": family, "split": split,
            "run_namespace": namespace, "receipt": receipt,
            "diversity": recipes[index] if index < len(recipes) else {}})
    return result


def discover(workers):
    tasks, sources, metadata = [], [], {}
    for name, cohort in LEGACY:
        root = DATA / name
        source = read(root / "source_manifest.json")
        manifest = read(root / "pipeline_run.json")
        records = {r["story_id"]: r for r in source["records"]}
        metadata.update({k: {**v, "cohort": cohort} for k, v in records.items()})
        accepted = defaultdict(list)
        with sqlite3.connect(f"file:{root}/progress.sqlite3?mode=ro", uri=True, timeout=30) as connection:
            for story, index, split in connection.execute(
                "SELECT shots.story_id, shots.shot_index, stories.split FROM shots "
                "JOIN stories ON stories.story_id=shots.story_id WHERE shots.trajectory_status='complete'"
            ):
                if records[story]["split"] != split:
                    raise ValueError(f"database split changed: {root}:{story}")
                accepted[story].append(index)
        tasks.extend((root, records[story], metadata[story], sorted(indices), manifest["run_namespace"])
                     for story, indices in accepted.items())
        sources.append({"root": str(root), "manifest_sha256": file_hash(root / "pipeline_run.json"),
                        "source_manifest_sha256": file_hash(root / "source_manifest.json")})
    root = DATA / "reel_balanced_h3_echo15_128_20260923/echo15"
    manifest = read(root / "backend_run.json")
    tasks.extend((root, record, metadata[record["story_id"]], None, manifest["run_namespace"])
                 for record in manifest["records"])
    sources.append({"root": str(root), "manifest_sha256": file_hash(root / "backend_run.json")})
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(story_rows, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Read committed story receipts"):
            rows.extend(future.result())
    return rows, sources


def select_unseen(candidates, excluded):
    excluded_ids = {identity(row) for row in excluded}
    excluded_paths = {row["receipt"]["trajectory_path"] for row in excluded}
    excluded_prompts = {row["rl_task_prompt_sha256"] for row in excluded}
    heldout = {row["rl_task_prompt_sha256"] for row in candidates if row["split"] == "test"}
    families, selected, reasons, seen_prompts, seen_ids = {}, [], Counter(), set(), set()
    # Validation takes precedence when a prompt is duplicated across families.
    order = {"test": 0, "validation": 1, "train": 2}
    for row in sorted(candidates, key=lambda r: (order[r["split"]], r["root"], r["story_id"], r["shot_index"])):
        family, split = row["family"], row["split"]
        if not family or families.setdefault(family, split) != split:
            raise ValueError(f"family split leakage: {family}")
        key = row["rl_task_prompt_sha256"]
        if split == "test" or key in heldout:
            reason = "heldout_test"
        elif identity(row) in excluded_ids or row["receipt"]["trajectory_path"] in excluded_paths:
            reason = "sft_source_identity"
        elif key in excluded_prompts:
            reason = "sft_prompt_content"
        elif key in seen_prompts or identity(row) in seen_ids:
            reason = "duplicate_new_prompt"
        else:
            selected.append(row)
            seen_prompts.add(key)
            seen_ids.add(identity(row))
            continue
        reasons[reason] += 1
    return selected, dict(reasons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-selection", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    original_hash = file_hash(args.sft_selection)
    excluded = [json.loads(line) for line in args.sft_selection.read_text().splitlines() if line.strip()]
    candidates, sources = discover(args.workers)
    unique = {row["receipt"]["trajectory_path"]: row for row in [*excluded, *candidates]}
    todo = sorted(unique.values(), key=lambda row: row["receipt"]["trajectory_path"])
    random.Random(args.seed).shuffle(todo)
    sample_count = min(1000, len(todo))
    hashes = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for label, rows in (("Prompt sample", todo[:sample_count]), ("Remaining prompt audit", todo[sample_count:])):
            futures = [pool.submit(read_prompt, row) for row in rows]
            for future in tqdm(as_completed(futures), total=len(futures), desc=label):
                row = future.result()
                hashes[row["receipt"]["trajectory_path"]] = row["rl_task_prompt_sha256"]
    # Bind each frozen receipt too, including any duplicate source paths.
    for row in [*excluded, *candidates]:
        source = unique[row["receipt"]["trajectory_path"]]
        if source["receipt"]["trajectory_sha256"] != row["receipt"]["trajectory_sha256"]:
            raise ValueError("SFT and current receipt disagree for the same source path")
        row["rl_task_prompt_sha256"] = hashes[row["receipt"]["trajectory_path"]]
    selected, reasons = select_unseen(candidates, excluded)
    counts = Counter(row["split"] for row in selected)
    if not counts["train"] or not counts["validation"]:
        raise ValueError(f"both unseen train and validation prompts required: {counts}")
    random.Random(args.seed).shuffle(selected)
    path = args.output / "selection.jsonl"
    path.write_text("".join(canonical(row) + "\n" for row in selected))
    overlap = len({r["rl_task_prompt_sha256"] for r in selected} & {r["rl_task_prompt_sha256"] for r in excluded})
    if overlap or file_hash(args.sft_selection) != original_hash:
        raise ValueError("prompt overlap or changed SFT selection")
    report = {"schema_version": SCHEMA, "created_at": time.time(), "normalizer": NORMALIZER,
        "sft_selection_path": str(args.sft_selection.resolve()), "sft_selection_sha256": original_hash,
        "excluded_sft_splits": ["train", "validation", "test"], "excluded_sft_rows": len(excluded),
        "candidate_rows": len(candidates), "selected_rows": len(selected), "counts": dict(counts),
        "exclusions": reasons, "sft_prompt_overlap": overlap, "duplicate_prompts": 0,
        "split_policy": "preserve source family splits; exclude test and all SFT snapshot prompts",
        "selection_sha256": file_hash(path), "sources": sources,
        "workers": args.workers, "sample_size": sample_count, "sample_seed": args.seed}
    (args.output / "unseen_prompt_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
