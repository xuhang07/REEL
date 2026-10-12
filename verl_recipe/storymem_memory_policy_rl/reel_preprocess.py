"""Build single-shot REEL RL rows from a sealed selection, preserving splits."""

from __future__ import annotations

import copy
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verl_recipe.storymem_memory_policy_rl import preprocess as base
from verl_recipe.storymem_memory_policy_rl.policy_profile import PROMPT_AUTHORING

_PROMPT_VALIDATOR = None


def _init_prompt_validator(model_path, tool_config_path, max_prompt_length):
    global _PROMPT_VALIDATOR
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _PROMPT_VALIDATOR = base.PromptLengthValidator(model_path=model_path, tool_config_path=tool_config_path,
                                                max_prompt_length=max_prompt_length)


def _load_and_count(row):
    row, spec = load_selected_shot(row)
    spec["initial_prompt_token_count"] = _PROMPT_VALIDATOR.count(spec["shot_requirement"])
    return row, spec


def build_reel_row(row, spec, index, contract):
    split = "val" if row["split"] == "validation" else "train"
    result = base.build_row(story_id=row["story_id"], split=split, index=index,
        mode="single", start=row["shot_index"], shot_specs=[spec], prompt_contract=contract)
    # Different source cohorts reuse story/shot IDs. Keep their GRPO groups and
    # artifact lineage distinct even when the original story text is identical.
    prefix = base.sha256_bytes(str(Path(spec["source_trajectory_path"]).resolve()).encode())[7:23]
    episode_id = f"reel:{prefix}:{result['extra_info']['episode_id']}"
    result["extra_info"]["episode_id"] = episode_id
    result["reward_model"]["ground_truth"]["episode_id"] = episode_id
    return result


def load_selected_shot(row):
    receipt = row["receipt"]
    shot_dir = Path(receipt["trajectory_path"]).parent
    trajectory, _ = base.read_stable_json(shot_dir / "trajectory.json", label="sealed source trajectory")
    if base.sha256_json(trajectory) != receipt["trajectory_sha256"]:
        raise base.PreprocessError("source trajectory differs from frozen SFT selection")
    if (trajectory.get("final_phase") != "complete" or trajectory.get("tool_schema_sha256") != base.tool_schema_hash()
            or trajectory.get("runtime_version") != base.RUNTIME_VERSION):
        raise base.PreprocessError("source trajectory is not a completed REEL authoring shot")
    sft, _ = base.read_stable_json(Path(receipt["sft_path"]), label="sealed source SFT")
    if base.sha256_json(sft) != receipt["sft_sha256"]:
        raise base.PreprocessError("source SFT differs from frozen selection")
    state, _ = base.read_stable_json(shot_dir / "policy_state.json", label="source state")
    index = int(row["shot_index"])
    requirement = trajectory["raw_user_request"]
    if "rl_task_prompt_sha256" in row:
        from tools.prepare_reel_rl_unseen_prompts import prompt_hash
        if prompt_hash(requirement) != row["rl_task_prompt_sha256"]:
            raise base.PreprocessError("task prompt differs from unseen-prompt audit")
    base._validate_final_state(state, story_id=row["story_id"], shot_index=index,
        expected_prompt=requirement, expected_request_id=receipt["request_id"], label="source state")
    previous_path = shot_dir.parent / f"shot_{index - 1:04d}" / "policy_state.json"
    memory = {}
    if index:
        previous, _ = base.read_stable_json(previous_path, label="preceding committed state")
        if previous.get("phase") != "complete" or previous.get("source_story_id") != row["story_id"]:
            raise base.PreprocessError("preceding memory state is not complete or belongs to another story")
        memory = previous["memory_bank"]
    # The registry as seen at the start of this shot excludes later writeback and
    # any subsequently prepared subjects. It is fully paginated in the trajectory.
    pages = [event["observation"] for event in trajectory["events"]
             if event["tool_call"]["name"] == "view_material_registry" and event["observation"].get("ok")]
    if not pages or pages[-1].get("next_cursor") is not None:
        raise base.PreprocessError("source registry pagination is incomplete")
    registry = {item["subject_id"]: item for page in pages for item in page["materials"]}
    seed = base._state_input_seed(current_state=state, registry_items=registry, memory_bank=memory,
        uploaded_evidence=state["uploaded_evidence"], previous_evidence=state["previous_evidence"])
    encoded = base.canonical_json(seed)
    if base.sha256_file(Path(receipt["output_path"])) != receipt["output_sha256"]:
        raise base.PreprocessError("source reference video differs from sealed receipt")
    spec = {"schema_version": base.SHOT_SPEC_SCHEMA_VERSION, "episode_position": 0,
        "shot_index": index, "shot_requirement": requirement,
        "shot_requirement_sha256": state["shot_requirement_sha256"],
        "source_story_dir": str(shot_dir.parent.parent), "source_shot_dir": str(shot_dir),
        "source_policy_state_path": str(shot_dir / "policy_state.json"),
        "source_previous_policy_state_path": str(previous_path) if index else "",
        "source_trajectory_path": receipt["trajectory_path"], "source_trajectory_sha256": receipt["trajectory_sha256"],
        "source_sft_path": receipt["sft_path"], "source_sft_sha256": receipt["sft_sha256"],
        "reference_output_path": receipt["output_path"], "reference_output_sha256": receipt["output_sha256"],
        "source_request_id": receipt["request_id"], "registry_memory_source_shot_index": index - 1,
        "state_seed": encoded, "state_seed_sha256": base.sha256_bytes(encoded.encode()),
        "inherit_registry_from_previous_rollout_shot": False, "inherit_memory_bank_from_previous_rollout_shot": False,
        "inherit_previous_evidence_from_previous_rollout_shot": False, "fresh_conversation": True}
    return row, spec


def export_dataset(args):
    from tqdm import tqdm
    if not PROMPT_AUTHORING or args.mode not in {"single", "multi"}:
        raise base.PreprocessError("sealed REEL exporter requires reel_joint_v1 and single or multi mode")
    selection = args.data_root / "selection.jsonl"
    digest = base.sha256_file(selection)
    audit_path = args.data_root / "unseen_prompt_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    audit_digest = base.sha256_file(audit_path) if audit else None
    if audit and (audit["selection_sha256"] != digest or audit["sft_prompt_overlap"] != 0):
        raise base.PreprocessError("unseen-prompt audit does not match selection")
    with selection.open() as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(json.loads, (line for line in stream if line.strip()), chunksize=128))
    families = {}
    for row in rows:
        family, split = row["family"], row["split"]
        if family in families and families[family] != split:
            raise base.PreprocessError("family overlap in frozen selection")
        families[family] = split
    selected = []
    for split, limit in (("train", args.max_stories), ("validation", args.val_stories or 8)):
        candidates = [row for row in rows if row["split"] == split and row["backend"] == "echo15"]
        stories = base.ranked_story_ids({r["story_id"] for r in candidates}, seed=args.split_seed, purpose=split)
        chosen = set(stories[:limit] if limit else stories)
        selected.extend(row for row in candidates if row["story_id"] in chosen)
    if not selected:
        raise base.PreprocessError("no REEL shots selected")
    if len({r["receipt"]["trajectory_path"] for r in selected}) != len(selected):
        raise base.PreprocessError("duplicate source trajectories in frozen selection")
    selected.sort(key=lambda r: (r["split"], r["story_id"], r["shot_index"]))
    validator = base.PromptLengthValidator(model_path=args.model_path, tool_config_path=args.tool_config_path,
                                         max_prompt_length=args.max_prompt_length)
    contract = validator.manifest_record()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    writers = {split: base.BufferedParquetWriter(output / f"{split}.parquet", batch_size=args.write_batch_size,
                                                compression=args.compression) for split in ("train", "val")}
    audits, token_counts, grouped_specs = [], [], {}
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_prompt_validator,
                                 initargs=(args.model_path, args.tool_config_path, args.max_prompt_length)) as pool:
            for index, (row, spec) in enumerate(tqdm(pool.map(_load_and_count, selected), total=len(selected), desc="REEL source shots")):
                count = spec["initial_prompt_token_count"]
                if count > args.max_prompt_length:
                    raise base.PreprocessError("initial REEL prompt exceeds limit; no silent filtering")
                spec["initial_prompt_token_count"] = count
                token_counts.append(count)
                split = "val" if row["split"] == "validation" else "train"
                if args.mode == "single":
                    writers[split].write(build_reel_row(row, spec, index, contract))
                else:
                    key = (split, row["story_id"], spec["source_story_dir"])
                    grouped_specs.setdefault(key, []).append(spec)
                audits.append(row)
        if args.mode == "multi":
            episode_index = 0
            for (split, story_id, source_dir), specs in sorted(grouped_specs.items()):
                specs.sort(key=lambda spec: spec["shot_index"])
                for start, window in base.episode_windows(specs, mode="multi", window_size=args.window_size, window_stride=args.window_stride):
                    indices = [spec["shot_index"] for spec in window]
                    if indices != list(range(indices[0], indices[0] + len(indices))):
                        continue
                    for spec in window[1:]:
                        seed = json.loads(spec["state_seed"])
                        for field in ("registry_items", "memory_bank", "previous_evidence"):
                            if field in seed:
                                seed[field] = {}
                        spec["state_seed"] = base.canonical_json(seed)
                        spec["state_seed_sha256"] = base.sha256_bytes(spec["state_seed"].encode())
                    result = base.build_row(story_id=story_id, split=split, index=episode_index, mode="multi",
                                            start=indices[0], shot_specs=window, prompt_contract=contract)
                    prefix = base.sha256_bytes(source_dir.encode())[7:23]
                    episode_id = f"reel:{prefix}:{result['extra_info']['episode_id']}"
                    result["extra_info"]["episode_id"] = episode_id
                    result["reward_model"]["ground_truth"]["episode_id"] = episode_id
                    writers[split].write(result)
                    episode_index += 1
    finally:
        for writer in writers.values():
            writer.close()
    if any(writer.rows == 0 for writer in writers.values()):
        raise base.PreprocessError("both train and validation rows are required")
    if base.sha256_file(selection) != digest:
        raise base.PreprocessError("frozen source selection changed")
    manifest = {"schema_version": base.MANIFEST_SCHEMA_VERSION,
        "source": {"selection_path": str(selection.resolve()), "selection_sha256": digest,
                   "backend": "echo15", "reference_hashes_verified": True},
        "selection": {"story_overlap": 0, "family_overlap": 0, "split_seed": args.split_seed,
                      "split_policy": audit["split_policy"] if audit else "preserve frozen SFT family splits; test excluded",
                      "selected_stories": len({r['story_id'] for r in selected})},
        "construction": {"mode": args.mode, "window_size": 1 if args.mode == "single" else args.window_size, "fresh_conversation_per_shot": True,
            "system_prompt_version": base.TRAINING_SYSTEM_PROMPT_VERSION,
            "system_prompt_sha256": base.training_system_prompt_hash(), "tool_schema_sha256": base.tool_schema_hash(),
            "prompt_validation": {**contract, "checked_shots": len(selected),
                                  "minimum_prompt_tokens": min(token_counts), "maximum_prompt_tokens": max(token_counts)}},
        "datasets": {s: base._file_summary(output / f"{s}.parquet", w.rows) for s, w in writers.items()},
        "episodes": {"total": sum(w.rows for w in writers.values()), **{s: w.rows for s, w in writers.items()}}}
    if audit:
        if base.sha256_file(audit_path) != audit_digest:
            raise base.PreprocessError("unseen-prompt audit changed during export")
        manifest["source"]["unseen_prompt_audit"] = {"path": str(audit_path.resolve()), "sha256": audit_digest}
    (output / "selection.jsonl").write_text("".join(base.canonical_json(r) + "\n" for r in audits))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(export_dataset(base.parse_args()), indent=2))
