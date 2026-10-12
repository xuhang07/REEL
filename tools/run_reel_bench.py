from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from reel_bench import common
from reel_bench.common import (
    BENCHMARK_ROOT, case_for, digest, load_suite, locked_json, probe_video, read_json,
    result_header, sha256_file, shot_seed, uploads_for, write_json,
)


def complete_shot(video: Path, shot_dir: Path, shot_index: int, seed: int) -> dict[str, Any]:
    media = probe_video(video)
    destination = shot_dir / ("video" + video.suffix.lower())
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.resolve() != video.resolve():
        shutil.copyfile(video, destination)
    return {"shot_index": shot_index, "status": "complete", "seed": seed,
            "video_path": str(destination.resolve()), "video_sha256": sha256_file(destination), "media": media}


def adapter_request(suite: dict, row: dict, shot_index: int, base_seed: int, settings: dict, history: list, state: dict) -> dict:
    case = case_for(suite, row)
    return {"schema_version": "reel_bench.shot_request.v1", "story_id": row["story_id"],
            "shot_index": shot_index, "prompt": case["prompts"][shot_index],
            "seed": shot_seed(base_seed, row["story_id"], shot_index), "generation_settings": settings,
            "new_uploads": uploads_for(suite, row, shot_index, only_new=True),
            "available_uploads": uploads_for(suite, row, shot_index), "previous_outputs": history,
            "state": state}


def run_adapter(suite: dict, config: dict, output: Path, base_seed: int, timeout: float, retry_failed: bool) -> dict:
    command = config.get("adapter_command")
    if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
        raise ValueError("config.adapter_command must be an argv array; shell execution is not supported")
    if not any("{request}" in part for part in command) or not any("{response}" in part for part in command):
        raise ValueError("adapter_command must contain {request} and {response} placeholders")
    settings = config.get("generation", {})
    header = result_header(suite, config["model"], base_seed, settings)
    run_config = {**header, "adapter_command": command, "timeout": timeout, "transport": "command.v1",
                  "runner_sha256": sha256_file(Path(__file__)), "common_sha256": sha256_file(Path(common.__file__))}
    locked_json(output / "run_config.json", run_config)
    result = {**header, "provenance": "sequential_adapter", "stories": {}}
    for row in suite["records"]:
        state: dict = {}
        history: list = []
        shots = []
        stopped = False
        for shot_index in range(row["num_shots"]):
            seed = shot_seed(base_seed, row["story_id"], shot_index)
            shot_dir = output / "stories" / row["story_id"] / f"shot_{shot_index:03d}"
            shot_dir.mkdir(parents=True, exist_ok=True)
            receipt_path = shot_dir / "result.json"
            if stopped:
                shots.append({"shot_index": shot_index, "status": "blocked", "seed": seed})
                continue
            existing = read_json(receipt_path) if receipt_path.exists() else None
            previous_attempt = int(existing.get("attempt", 0)) if existing else 0
            retry = bool(existing and existing["status"] == "failed" and retry_failed)
            attempt = previous_attempt + int(retry)
            request = adapter_request(suite, row, shot_index, base_seed, settings, history, state)
            request["output_dir"] = str(shot_dir.resolve())
            request["attempt"] = attempt
            request["request_id"] = digest([digest(run_config), row["story_id"], shot_index, attempt])
            request_path, response_path = shot_dir / "request.json", shot_dir / "response.json"
            dispatch = shot_dir / "dispatched.json"
            if retry:
                locked_json(shot_dir / f"failed_attempt_{previous_attempt:03d}.json", existing)
                response_path.unlink(missing_ok=True)
                dispatch.unlink(missing_ok=True)
                write_json(request_path, request)
            else:
                locked_json(request_path, request)
            if existing and existing["status"] == "complete":
                if sha256_file(Path(existing["video_path"])) != existing["video_sha256"]:
                    raise ValueError("cached video changed; refusing to resume")
                shot = existing
            else:
                try:
                    if not response_path.exists():
                        if dispatch.exists():
                            raise RuntimeError("indeterminate prior dispatch: reconcile adapter/backend before retrying")
                        write_json(receipt_path, {"shot_index": shot_index, "status": "indeterminate", "seed": seed, "attempt": attempt})
                        write_json(dispatch, {"request_id": request["request_id"], "request_sha256": digest(request)})
                        argv = [part.replace("{request}", str(request_path.resolve())).replace("{response}", str(response_path.resolve())).replace("{python}", sys.executable) for part in command]
                        environment = os.environ.copy()
                        environment["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + environment.get("PYTHONPATH", "")
                        with (shot_dir / "adapter.log").open("ab") as log:
                            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True, env=environment)
                    response = read_json(response_path)
                    if response.get("request_id") != request["request_id"]:
                        raise ValueError("adapter response request_id mismatch")
                    if response.get("status") != "complete":
                        if response.get("status") != "failed":
                            raise ValueError("adapter status must be complete or terminal failed")
                        shot = {"shot_index": shot_index, "status": "failed", "seed": seed,
                                "error": str(response.get("error", "adapter reported terminal failure"))[:500]}
                    else:
                        video = Path(response["video_path"])
                        if not video.is_absolute():
                            video = shot_dir / video
                        if not isinstance(response.get("state", {}), dict):
                            raise ValueError("adapter state must be a JSON object")
                        shot = complete_shot(video, shot_dir, shot_index, seed)
                        shot["state"] = response.get("state", {})
                except Exception as error:
                    shot = {"shot_index": shot_index, "status": "indeterminate", "seed": seed,
                            "error": f"{type(error).__name__}: {error}"[:500]}
                shot["attempt"] = attempt
                write_json(receipt_path, shot)
            shots.append(shot)
            if shot["status"] != "complete":
                stopped = True
            else:
                state = shot.get("state", {})
                history.append({key: shot[key] for key in ("shot_index", "video_path", "video_sha256")})
        result["stories"][row["story_id"]] = [{key: value for key, value in shot.items() if key != "state"} for shot in shots]
        write_json(output / "results.partial.json", result)
    write_json(output / "results.json", result)
    return result


def import_videos(suite: dict, config: dict, video_map: Path, output: Path, base_seed: int) -> dict:
    supplied = read_json(video_map)["stories"]
    if set(supplied) != {row["story_id"] for row in suite["records"]}:
        raise ValueError("video map must list every selected story; represent failed shots explicitly with null")
    header = result_header(suite, config["model"], base_seed, config.get("generation", {}))
    locked_json(output / "run_config.json", {**header, "transport": "external_import.v1", "map_sha256": sha256_file(video_map)})
    result = {**header, "provenance": "external_import_generation_conditions_user_declared", "stories": {}}
    for row in suite["records"]:
        videos = supplied[row["story_id"]]
        if not isinstance(videos, list) or len(videos) != row["num_shots"]:
            raise ValueError(f"wrong number of videos for {row['story_id']}")
        shots = []
        for shot_index, value in enumerate(videos):
            seed = shot_seed(base_seed, row["story_id"], shot_index)
            if value is None:
                shots.append({"shot_index": shot_index, "status": "failed", "seed": seed, "error": "explicitly missing imported output"})
            else:
                video = Path(value)
                if not video.is_absolute():
                    video = video_map.parent / video
                shots.append(complete_shot(video, output / "stories" / row["story_id"] / f"shot_{shot_index:03d}", shot_index, seed))
        result["stories"][row["story_id"]] = shots
    write_json(output / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sequential model adapter or import existing outputs for frozen REEL-Bench.")
    parser.add_argument("command", choices=["run", "import", "plan"])
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--split", choices=["subset30", "full"], default="subset30")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos-map", type=Path)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.base_seed < 0 or args.timeout <= 0:
        parser.error("base seed must be nonnegative and timeout positive")
    suite = load_suite(args.benchmark, args.split)
    config = read_json(args.config)
    header = result_header(suite, config["model"], args.base_seed, config.get("generation", {}))
    if args.command == "plan":
        print(json.dumps({**header, "stories": len(suite["records"]), "shots": suite["manifest"]["num_shots"],
                          "seed_rule": "sha256(base_seed, story_id, shot_index); identical in subset and full"}, ensure_ascii=False, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        if args.command == "import":
            if args.videos_map is None:
                parser.error("import requires --videos-map")
            result = import_videos(suite, config, args.videos_map, args.output, args.base_seed)
        else:
            result = run_adapter(suite, config, args.output, args.base_seed, args.timeout, args.retry_failed)
    counts = {status: sum(shot["status"] == status for shots in result["stories"].values() for shot in shots)
              for status in ("complete", "failed", "blocked", "indeterminate")}
    print(json.dumps({"results": str((args.output / "results.json").resolve()), "elapsed_seconds": time.monotonic() - started, "shots": counts}, indent=2))
    if any(counts[status] for status in ("failed", "blocked", "indeterminate")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
