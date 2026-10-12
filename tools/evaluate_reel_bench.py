from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import json
import random
import statistics
import subprocess
from pathlib import Path
from typing import Any

from reel_bench import common, judging
from reel_bench.common import (
    BENCHMARK_ROOT, case_for, digest, load_results, load_suite, locked_json,
    read_json, sha256_file, uploads_for, write_json,
)
from reel_bench.judging import EvidenceCacheError, JudgeClient, image_part, judge_cached, sample_video


def image_references(suite: dict, row: dict, shot_index: int, protocol: dict) -> tuple[list[dict], bool]:
    content = []
    for upload in uploads_for(suite, row, shot_index):
        locator = upload.get("image_path")
        if locator is None and upload.get("declared_asset_type") != "voice_reference":
            locator = upload.get("path")
        if locator is None:
            continue
        description = {key: upload.get(key) for key in ("upload_id", "user_description", "declared_asset_type", "available_from_shot")}
        content.append({"type": "text", "text": "UPLOADED IMAGE REFERENCE: " + json.dumps(description, ensure_ascii=False)})
        content.append(image_part(Path(locator), protocol["sampling"]))
    return content, bool(content)


def evaluate_story(suite: dict, row: dict, result_shots: list[dict], output: Path, protocol: dict, client: Any, contract: str) -> dict:
    story_dir = output / "stories" / row["story_id"]
    case = case_for(suite, row)
    evidence: dict[int, dict] = {}
    shots = []
    for result in result_shots:
        index = result["shot_index"]
        reference_parts, has_references = image_references(suite, row, index, protocol)
        shot = {"shot_index": index, "generation_status": result["status"], "has_references": has_references}
        if result["status"] != "complete":
            shots.append(shot)
            continue
        if sha256_file(Path(result["video_path"])) != result["video_sha256"]:
            raise ValueError("generation video changed during evaluation")
        try:
            sampled = sample_video(Path(result["video_path"]), output / "frame_cache", protocol["sampling"])
            evidence[index] = sampled
        except EvidenceCacheError:
            raise
        except (ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            shot.update({"generation_status": "invalid_video", "error_type": type(error).__name__})
            shots.append(shot)
            continue
        content = [{"type": "text", "text": json.dumps({"shot_index": index, "request": case["prompts"][index],
                    "video_metadata": sampled["media"], "reference_rule": "Only these already-arrived image references are available."}, ensure_ascii=False)}]
        content.extend(reference_parts)
        for frame in sampled["frames"]:
            content.append({"type": "text", "text": f"GENERATED shot_index={index} frame_index={frame['frame_index']} time={frame['timestamp']:.3f}s"})
            content.append(image_part(Path(frame["path"]), protocol["sampling"]))
        allowed = {(index, frame["frame_index"]) for frame in sampled["frames"]}
        shot["judge"] = judge_cached(story_dir / f"shot_{index:03d}.json", client, protocol, "shot", content, allowed, has_references, contract)
        shot["media"] = sampled["media"]
        shots.append(shot)
    global_judge = {"status": "insufficient_generated_video"}
    if len(evidence) >= 2:
        references, has_references = image_references(suite, row, row["num_shots"] - 1, protocol)
        content = [{"type": "text", "text": json.dumps({"requests": [{"shot_index": index, "request": prompt,
                    "video_available": index in evidence} for index, prompt in enumerate(case["prompts"])],
                    "reference_rule": "Each reference can condition only shots at or after its available_from_shot. Each board is one shot; its four frames are chronological."}, ensure_ascii=False)}]
        content.extend(references)
        allowed = set()
        for index, sampled in evidence.items():
            content.append({"type": "text", "text": f"GENERATED shot_index={index}; board frame indices={protocol['sampling']['story_frame_indices']}"})
            content.append(image_part(Path(sampled["board"]), protocol["sampling"]))
            allowed.update((index, frame_index) for frame_index in protocol["sampling"]["story_frame_indices"])
        global_judge = judge_cached(story_dir / "story.json", client, protocol, "story", content, allowed, has_references, contract)
    evaluated = {"story_id": row["story_id"], "shots": shots, "story_judge": global_judge}
    write_json(story_dir / "evaluation.json", evaluated)
    return evaluated


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def bootstrap_interval(values: list[float], samples: int, seed: int) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    draws = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]]


def summarize_story(row: dict, evaluation: dict, protocol: dict) -> dict:
    shots = evaluation["shots"]
    completed = sum(shot["generation_status"] == "complete" for shot in shots)
    completion = completed / row["num_shots"]
    errors = sum(shot.get("judge", {}).get("status") == "judge_error" for shot in shots)
    errors += evaluation["story_judge"]["status"] == "judge_error"
    values: dict[str, float | None] = {}
    for metric in ("prompt_following", "visual_quality", "motion_temporal"):
        scores = []
        for shot in shots:
            if shot["generation_status"] != "complete":
                scores.append(0.0)
            elif shot["judge"]["status"] == "ok":
                scores.append(shot["judge"]["judgment"]["scores"][metric]["value"] * 25.0)
        values[metric] = mean(scores) if len(scores) == len(shots) else None
    global_judge = evaluation["story_judge"]
    for metric in ("cross_shot_consistency", "narrative_coherence"):
        if global_judge["status"] == "ok":
            values[metric] = global_judge["judgment"]["scores"][metric]["value"] * 25.0 * completion
        else:
            values[metric] = None if global_judge["status"] == "judge_error" else 0.0
    optional = {name: {"score": None, "applicable": 0, "not_applicable": 0, "missing": 0} for name in protocol["optional_metrics"]}
    reference_scores = []
    for shot in shots:
        reference = optional["reference_fidelity"]
        if not shot["has_references"]:
            reference["not_applicable"] += 1
        elif shot.get("judge", {}).get("status") != "ok":
            reference["missing"] += 1
        else:
            score = shot["judge"]["judgment"]["scores"]["reference_fidelity"]["value"]
            if score is None:
                reference["not_applicable"] += 1
            else:
                reference["applicable"] += 1
                reference_scores.append(score * 25.0)
    optional["reference_fidelity"]["score"] = mean(reference_scores)
    state = optional["state_compliance"]
    if global_judge["status"] != "ok":
        state["missing"] = 1
    else:
        score = global_judge["judgment"]["scores"]["state_compliance"]["value"]
        if score is None:
            state["not_applicable"] = 1
        else:
            state.update({"score": score * 25.0 * completion, "applicable": 1})
    core = sum(values[name] * weight for name, weight in protocol["core_weights"].items()) if all(value is not None for value in values.values()) else None
    served_models = {judge["served_model"] for judge in [global_judge, *(shot.get("judge", {}) for shot in shots)] if judge.get("status") == "ok"}
    return {"story_id": row["story_id"], "input_group": "no_upload" if not row["num_uploads"] else "image_audio" if row["has_audio"] else "image_only",
            "story_style": row["story_style"], "upload_combination": row["upload_combination"], "upload_timing": row["upload_timing"],
            "num_shots": row["num_shots"], "completed_shots": completed, "core_100": core, "metrics": values,
            "optional_metrics": optional, "judge_errors": errors, "served_models": sorted(served_models),
            "unfinalized_shots": sum(shot["generation_status"] in {"pending", "indeterminate"} for shot in shots),
            "audio_track_count": sum(shot.get("media", {}).get("has_audio_stream", False) for shot in shots)}


def aggregate(stories: list[dict], protocol: dict) -> dict:
    served = sorted({model for row in stories for model in row["served_models"]})
    rankable = all(row["core_100"] is not None and not row["judge_errors"] and not row["unfinalized_shots"] for row in stories) and len(served) <= 1 and "unreported" not in served
    cores = [row["core_100"] for row in stories if row["core_100"] is not None]
    groups = {}
    for field in ("input_group", "story_style", "upload_combination", "upload_timing"):
        groups[field] = {value: {"stories": len(selected), "core_100": mean([row["core_100"] for row in selected]) if all(row["core_100"] is not None for row in selected) else None}
                         for value in sorted({row[field] for row in stories})
                         for selected in [[row for row in stories if row[field] == value]]}
    optional = {}
    for metric in protocol["optional_metrics"]:
        rows = [row["optional_metrics"][metric] for row in stories]
        optional[metric] = {"score": mean([row["score"] for row in rows if row["score"] is not None]),
                            "scored_stories": sum(row["score"] is not None for row in rows),
                            **{name: sum(row[name] for row in rows) for name in ("applicable", "not_applicable", "missing")}}
    metrics = {name: mean([row["metrics"][name] for row in stories]) if all(row["metrics"][name] is not None for row in stories) else None for name in protocol["core_weights"]}
    expected = sum(row["num_shots"] for row in stories)
    completed = sum(row["completed_shots"] for row in stories)
    return {"rankable": rankable, "core_100": mean(cores) if rankable else None,
            "core_95ci": bootstrap_interval(cores, protocol["bootstrap"]["samples"], protocol["bootstrap"]["seed"]) if rankable else None,
            "metrics": metrics, "optional_metrics": optional, "groups": groups,
            "coverage": {"stories": len(stories), "complete_stories": sum(row["completed_shots"] == row["num_shots"] for row in stories),
                         "expected_shots": expected, "completed_shots": completed, "generation_success_rate": completed / expected,
                         "judge_errors": sum(row["judge_errors"] for row in stories), "unfinalized_shots": sum(row["unfinalized_shots"] for row in stories)},
            "audio": {"status": "not_evaluated_by_visual_judge", "generated_clips_with_audio_track": sum(row["audio_track_count"] for row in stories)},
            "served_judge_models": served, "stories": stories}


def evaluate(suite: dict, results: dict, output: Path, protocol: dict, client: Any, identity: dict, workers: int = 2) -> dict:
    if set(protocol["core_weights"]) != {"prompt_following", "visual_quality", "motion_temporal", "cross_shot_consistency", "narrative_coherence"} or abs(sum(protocol["core_weights"].values()) - 1) > 1e-9 or any(weight < 0 for weight in protocol["core_weights"].values()):
        raise ValueError("invalid core metrics/weights")
    sampling = protocol["sampling"]
    if sampling["shot_frames"] < 2 or len(sampling["story_frame_indices"]) != 4 or any(not 0 <= index < sampling["shot_frames"] for index in sampling["story_frame_indices"]):
        raise ValueError("invalid frame sampling protocol")
    implementation = digest([sha256_file(Path(__file__)), sha256_file(Path(common.__file__)), sha256_file(Path(judging.__file__))])
    ffmpeg_version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    judge_contract = {"protocol_sha256": digest(protocol), "judge": identity, "implementation_sha256": implementation, "ffmpeg": ffmpeg_version}
    config = {"release_id": suite["release"]["release_id"], "split_sha256": suite["split_sha256"],
              "results_sha256": digest(results), "judge_contract": judge_contract}
    locked_json(output / "eval_config.json", config)
    contract = digest(judge_contract)
    evaluations = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_story, suite, row, results["stories"][row["story_id"]], output, protocol, client, contract): row for row in suite["records"]}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            evaluations[row["story_id"]] = future.result()
            print(json.dumps({"story_id": row["story_id"], "evaluated_stories": len(evaluations)}), flush=True)
    stories = [summarize_story(row, evaluations[row["story_id"]], protocol) for row in suite["records"]]
    report = aggregate(stories, protocol)
    comparison = {"release_id": suite["release"]["release_id"], "split_sha256": suite["split_sha256"],
                  "judge_contract": judge_contract, "base_seed": results["base_seed"], "generation_settings": results["generation_settings"],
                  "served_judge_models": report["served_judge_models"]}
    report.update({"schema_version": "reel_bench.report.v1", "model": results["model"], "split": suite["split"],
                   "comparison_contract": comparison, "comparison_key": digest(comparison), "protocol_status": protocol["status"],
                   "generation_provenance": results["provenance"]})
    write_json(output / "summary.json", report)
    with (output / "per_story.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["story_id", "input_group", "story_style", "core_100", "completed_shots", "num_shots", *protocol["core_weights"], *protocol["optional_metrics"]]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in stories:
            writer.writerow({**{key: row[key] for key in fields if key in row}, **row["metrics"],
                             **{key: value["score"] for key, value in row["optional_metrics"].items()}})
    with (output / "per_shot.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["story_id", "shot_index", "generation_status", "judge_status", *protocol["shot_metrics"]])
        writer.writeheader()
        for row in suite["records"]:
            for shot in evaluations[row["story_id"]]["shots"]:
                judge = shot.get("judge", {})
                scores = judge.get("judgment", {}).get("scores", {})
                writer.writerow({"story_id": row["story_id"], "shot_index": shot["shot_index"],
                                 "generation_status": shot["generation_status"], "judge_status": judge.get("status", "not_called"),
                                 **{name: score["value"] * 25 if score["value"] is not None else None for name, score in scores.items()}})
    text = [f"# REEL-Bench · {results['model']['name']}", f"Split: {suite['split']}; rankable: {report['rankable']}; protocol: {protocol['version']} ({protocol['status']}).",
            f"Core-100: {report['core_100']}; story-bootstrap 95% CI: {report['core_95ci']}.",
            f"Generation coverage: {report['coverage']['completed_shots']}/{report['coverage']['expected_shots']} shots.",
            "", "| Metric | Score (0–100) |", "| --- | ---: |"]
    text.extend(f"| {name} | {score} |" for name, score in report["metrics"].items())
    text.extend(f"| {name} (conditional; inspect applicability/missing counts) | {value['score']} |" for name, value in report["optional_metrics"].items())
    text.extend(["", "Audio is not judged. Temporal quality is a sampled-frame proxy. Review summary.json denominators and raw evidence before making claims."])
    (output / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return report


def compare(first: dict, second: dict) -> dict:
    if not first["rankable"] or not second["rankable"] or first["comparison_key"] != second["comparison_key"]:
        raise ValueError("comparison requires rankable reports with identical benchmark, split, judge, seed and generation settings")
    left = {row["story_id"]: row["core_100"] for row in first["stories"]}
    right = {row["story_id"]: row["core_100"] for row in second["stories"]}
    if set(left) != set(right):
        raise ValueError("paired comparison requires identical story IDs")
    differences = [right[key] - left[key] for key in sorted(left)]
    return {"direction": "second_minus_first", "first_model": first["model"], "second_model": second["model"],
            "paired_stories": len(differences), "core_delta": mean(differences),
            "paired_story_bootstrap_95ci": bootstrap_interval(differences, 2000, 20260922),
            "story_wins": sum(value > 0 for value in differences), "story_ties": differences.count(0)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Blinded, evidence-grounded VLM evaluation for frozen REEL-Bench.")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--benchmark", type=Path, default=BENCHMARK_ROOT)
    evaluation.add_argument("--split", choices=["subset30", "full"], default="subset30")
    evaluation.add_argument("--results", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--protocol", type=Path)
    evaluation.add_argument("--judge-model", required=True)
    evaluation.add_argument("--judge-revision", required=True, help="Pinned provider/model revision; do not reuse this label for a changed judge.")
    evaluation.add_argument("--api-base")
    evaluation.add_argument("--workers", type=int, default=2)
    evaluation.add_argument("--timeout", type=float, default=180)
    evaluation.add_argument("--dry-run", action="store_true")
    comparison = commands.add_parser("compare")
    comparison.add_argument("first", type=Path)
    comparison.add_argument("second", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        result = compare(read_json(args.first), read_json(args.second))
        write_json(args.output, result)
    else:
        if args.workers < 1 or args.timeout <= 0:
            parser.error("workers and timeout must be positive")
        suite = load_suite(args.benchmark, args.split)
        results = load_results(args.results, suite)
        protocol = read_json(args.protocol or args.benchmark / "judge_protocol.json")
        if args.dry_run:
            result = {"stories": len(suite["records"]), "max_initial_judge_calls": suite["manifest"]["num_shots"] + len(suite["records"]),
                      "judge_model": args.judge_model, "protocol": protocol["version"], "network_calls_made": 0}
        else:
            client = JudgeClient(args.judge_model, args.api_base, args.timeout)
            identity = {"model": args.judge_model, "revision": args.judge_revision, "endpoint": client.endpoint}
            args.output.mkdir(parents=True, exist_ok=True)
            with (args.output / ".eval.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                report = evaluate(suite, results, args.output, protocol, client, identity, args.workers)
            result = {"summary": str((args.output / "summary.json").resolve()), "rankable": report["rankable"],
                      "core_100": report["core_100"], "coverage": report["coverage"]}
            if not report["rankable"]:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
