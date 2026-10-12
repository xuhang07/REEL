"""Run PE-only canaries against immutable completed REEL states."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from reel_h3_adapter import H3R2AVPromptAdapter, VERSION
from storymem_adapter.models import sha256_bytes, sha256_text
from storymem_shot_assets import Services
from storymem_teacher.util import atomic_write_json


REPO = Path(__file__).resolve().parents[1]


def run_case(path: Path, root: Path, adapter: H3R2AVPromptAdapter, services: Services) -> dict:
    started = time.monotonic()
    source_bytes = path.read_bytes()
    state = json.loads(source_bytes)
    case_id = path.parent.name + "-" + sha256_text(str(path)).removeprefix("sha256:")[:16]
    result = {
        "case_id": case_id,
        "state_path": str(path),
        "state_sha256": sha256_bytes(source_bytes),
        "h3_submitted": False,
    }
    try:
        if state.get("phase") != "complete":
            raise ValueError("PE canaries read completed states only; do not race an active turn")
        compilation = adapter.compile(state=state, raw_prompt=state["shot_requirement"])
        document = compilation.document
        references = document["prepared"]["input"]["references"]
        result.update({
            "status": "pe_passed",
            "reference_count": len(references),
            "reference_roles": [reference["role"] for reference in references],
            "definitions": [reference["definition"] for reference in references],
            "artifact_path": str(compilation.artifact_path),
            "models": [request["model"] for request in document["pe_requests"]],
            "prompt_sha256": document["prompt_sha256"],
        })
    except Exception as error:
        result.update({
            "status": "pe_failed",
            "error_type": type(error).__name__,
            "error_code": getattr(error, "code", None),
            "error": services.safe_error(error),
        })
    result["wall_seconds"] = round(time.monotonic() - started, 3)
    result["source_state_unchanged"] = path.read_bytes() == source_bytes
    if not result["source_state_unchanged"]:
        result["status"] = "source_changed"
    atomic_write_json(root / "cases" / f"{case_id}.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-config", type=Path, default=REPO / "Director_Agent/.config.local.json")
    parser.add_argument("--api-key-script", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-references", type=int, default=12)
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 4:
        parser.error("PE canaries are bounded to one through four workers, not H3 production workers")
    root = arguments.output_root.expanduser().resolve()
    paths = [path.expanduser().resolve() for path in arguments.states]
    if len(set(paths)) != len(paths) or not all(path.is_file() for path in paths):
        parser.error("states must be unique existing files")
    inside_production = any((ancestor / "pipeline_run.json").exists() for ancestor in (root, *root.parents))
    if inside_production or any(path.is_relative_to(root) or root.is_relative_to(path.parent) for path in paths):
        parser.error("use a separate output root, never a production dataset or source-state directory")
    root.mkdir(parents=True, exist_ok=True)
    services = Services(
        root, arguments.api_config,
        concurrency=arguments.workers, qps=1.0,
        native_text_concurrency=1, native_text_qps=0.5,
        image_concurrency=2, offline_image_concurrency=1,
        api_key_script=arguments.api_key_script,
    )
    adapter = H3R2AVPromptAdapter(
        artifact_root=root / "adapter",
        gemini_client=services,
        max_references=arguments.max_references,
    )
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        results = list(executor.map(lambda path: run_case(path, root, adapter, services), paths))
    summary = {
        "schema_version": VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "pe_only",
        "h3_submitted": False,
        "h3_outputs": 0,
        "scope_note": "PE-only validation; real H3 transport and video validation are separate canaries",
        "planner_changed": False,
        "workers": arguments.workers,
        "wall_seconds": round(time.monotonic() - started, 3),
        "passed": sum(result["status"] == "pe_passed" for result in results),
        "failed": sum(result["status"] != "pe_passed" for result in results),
        "results": results,
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    atomic_write_json(root / "canary_runs" / f"{run_id}.json", summary)
    atomic_write_json(root / "pe_canary_summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
