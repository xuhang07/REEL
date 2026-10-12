from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import html
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "source" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "reel_bench"
LOCATORS = ("image_path", "audio_path", "video_path", "path")
UPLOAD_FIELDS = {
    "upload_id", "user_description", "declared_asset_type", "condition_eligible", "tags",
    "shot_index", "speaker_name", "speaker_binding", "paired_image_upload_id", "sha256",
}
DIVERSITY_WEIGHTS = {
    "story_style": 12.0, "task_type": 5.0, "upload_timing": 3.0,
    "primary_character": 3.0, "secondary_character": 2.0, "prop": 3.0,
    "scene_pair": 3.0, "num_shots": 2.0, "num_uploads": 2.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_digest(prompts: list[str]) -> str:
    return hashlib.sha256(json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def asset_path(upload: dict[str, Any], source: Path) -> tuple[str, Path]:
    locators = [(key, upload[key]) for key in LOCATORS if upload.get(key)]
    if len(locators) != 1:
        raise ValueError(f"expected one local asset locator in {source}")
    key, locator = locators[0]
    path = Path(locator).expanduser()
    return key, (source.parent / path).resolve() if not path.is_absolute() else path.resolve()


def inspect_source(record: dict[str, Any]) -> dict[str, Any]:
    source = Path(record["path"])
    data = json.loads(source.read_text(encoding="utf-8"))
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
        raise ValueError(f"invalid prompts: {source}")
    schedule = data.get("uploaded_assets_by_shot") or {}
    if not isinstance(schedule, dict) or any(not isinstance(items, list) for items in schedule.values()):
        raise ValueError(f"invalid upload schedule: {source}")
    uploads = [upload for items in schedule.values() for upload in items]
    missing = [str(path) for upload in uploads if not (path := asset_path(upload, source)[1]).is_file()]
    metadata = data.get("metadata") or {}
    roles = sorted({upload.get("declared_asset_type", "reference") for upload in uploads})
    return {
        "story_id": data["story_id"], "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source), "prompts_sha256": prompt_digest(prompts),
        "num_shots": len(prompts), "num_uploads": len(uploads), "upload_types": roles,
        "upload_combination": "+".join(roles) or "no_upload", "has_audio": "voice_reference" in roles,
        "story_style": metadata.get("style", {}).get("description", "unknown"),
        "task_type": data.get("task_type", "unknown"),
        "upload_timing": "none" if not uploads else "initial_and_mid" if "0" in schedule and len(schedule) > 1 else "initial_only" if "0" in schedule else "mid_only",
        "primary_character": metadata.get("primary_character", {}).get("role", "unknown"),
        "secondary_character": metadata.get("secondary_character", {}).get("role", "unknown"),
        "prop": metadata.get("prop", {}).get("name", "unknown"),
        "scene_pair": " -> ".join(scene["name"] for scene in metadata.get("scenes", [])),
        "missing_assets": missing,
    }


def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["upload_combination", *DIVERSITY_WEIGHTS]
    return {field: dict(sorted(Counter(row[field] for row in rows).items(), key=lambda item: str(item[0]))) for field in fields}


def quotas_for(values: list[str], count: int) -> dict[str, int]:
    if count and not values:
        raise ValueError("no candidates for requested quota")
    return dict(Counter(values[index % len(values)] for index in range(count)))


def select_candidates(rows: list[dict[str, Any]], count: int, audio_count: int, seed: int, *, no_upload_count: int = 0) -> list[dict[str, Any]]:
    if min(audio_count, no_upload_count) < 0 or audio_count + no_upload_count > count or count < 1:
        raise ValueError("require nonnegative audio/no-upload quotas summing to at most count, and count > 0")
    candidates = [row for row in rows if (row["num_uploads"] or no_upload_count) and not row["missing_assets"]]
    combinations = sorted({row["upload_combination"] for row in candidates})
    image_combinations = [value for value in combinations if value != "no_upload" and "voice_reference" not in value]
    audio_combinations = [value for value in combinations if "voice_reference" in value]
    quotas = quotas_for(image_combinations, count - audio_count - no_upload_count) | quotas_for(audio_combinations, audio_count)
    if no_upload_count:
        quotas["no_upload"] = no_upload_count
    rng = random.Random(seed)
    rng.shuffle(candidates)
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        pools[row["upload_combination"]].append(row)
    for combination, quota in quotas.items():
        if len(pools[combination]) < quota:
            raise ValueError(f"insufficient candidates for {combination}: {len(pools[combination])} < {quota}")
    style_quotas = quotas_for(sorted({row["story_style"] for row in candidates}), count)
    counts: dict[str, Counter] = {field: Counter() for field in DIVERSITY_WEIGHTS}
    selected_combinations: Counter = Counter()
    pair_counts: Counter = Counter()
    seen_prompts: set[str] = set()
    selected = []
    for _ in range(count):
        eligible = {
            combination: [row for row in pools[combination]
                          if row["prompts_sha256"] not in seen_prompts
                          and counts["story_style"][row["story_style"]] < style_quotas[row["story_style"]]]
            for combination, quota in quotas.items() if selected_combinations[combination] < quota
        }
        if any(not pool for pool in eligible.values()):
            raise ValueError("cannot satisfy style/combination quotas without duplicates; adjust quotas or seed")
        combination = min(eligible, key=lambda value: len(eligible[value]) / (quotas[value] - selected_combinations[value]))

        def score(row: dict[str, Any]) -> float:
            diversity = sum(weight / (1 + counts[field][row[field]]) for field, weight in DIVERSITY_WEIGHTS.items())
            return diversity + 5.0 / (1 + pair_counts[(combination, row["story_style"])])

        chosen = max(eligible[combination], key=score)
        selected.append(chosen)
        selected_combinations[combination] += 1
        pair_counts[(combination, chosen["story_style"])] += 1
        seen_prompts.add(chosen["prompts_sha256"])
        for field in DIVERSITY_WEIGHTS:
            counts[field][chosen[field]] += 1
    return sorted(selected, key=lambda row: row["story_id"])


def export_case(row: dict[str, Any], output: Path) -> dict[str, Any]:
    source = Path(row["source_path"])
    if sha256_file(source) != row["source_sha256"]:
        raise ValueError(f"source changed since inventory: {source}")
    original = json.loads(source.read_text(encoding="utf-8"))
    schedule = {}
    assets = []
    for shot, uploads in (original.get("uploaded_assets_by_shot") or {}).items():
        schedule[shot] = []
        for upload in uploads:
            locator, original_path = asset_path(upload, source)
            digest = sha256_file(original_path)
            declared = upload.get("sha256")
            if declared and str(declared).removeprefix("sha256:") != digest:
                raise ValueError(f"declared asset hash mismatch: {original_path}")
            extension = original_path.suffix.lower()
            if locator != "audio_path" and extension in {".jpg", ".jpeg", ".png", ".webp"}:
                with Image.open(original_path) as image:
                    image.verify()
            relative_path = f"assets/{digest}{extension}"
            target = output / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(original_path, target)
            if sha256_file(target) != digest:
                raise ValueError(f"copied asset hash mismatch: {target}")
            cleaned = {key: copy.deepcopy(value) for key, value in upload.items() if key in UPLOAD_FIELDS}
            if upload.get("declared_asset_type") == "voice_reference":
                cleaned["memory_role"] = upload["memory_role"]
            cleaned[locator] = "../" + relative_path
            schedule[shot].append(cleaned)
            assets.append({"upload_id": upload["upload_id"], "upload_shot_key": shot, "locator": locator,
                           "source_path": str(original_path), "path": relative_path, "sha256": digest})
    case = {"story_id": original["story_id"], "title": original.get("title", original["story_id"]),
            "uploaded_assets_by_shot": schedule, "prompts": original["prompts"]}
    if row["has_audio"]:
        case["metadata"] = {
            key: copy.deepcopy(original["metadata"][key])
            for key in ("primary_character", "secondary_character") if key in original.get("metadata", {})
        }
    path = output / "cases" / f"{row['story_id']}.json"
    write_json(path, case)
    return {key: value for key, value in row.items() if key != "missing_assets"} | {
        "path": str(path.resolve()), "relative_path": f"cases/{path.name}",
        "case_sha256": sha256_file(path), "assets": assets,
    }


def input_groups(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "no_upload": [row for row in records if not row["num_uploads"]],
        "image_only": [row for row in records if row["num_uploads"] and not row["has_audio"]],
        "image_audio": [row for row in records if row["has_audio"]],
    }


def write_subset_manifests(output: Path, records: list[dict[str, Any]]) -> None:
    for name, subset in input_groups(records).items():
        write_json(output / f"{name}_manifest.json", {"records": [
            {"story_id": row["story_id"], "path": row["path"], "relative_path": row["relative_path"], "num_shots": row["num_shots"]}
            for row in subset
        ]})


def validate(output: Path) -> dict[str, Any]:
    from storymem_teacher.source import load_story_source, validate_source_asset_locators

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    if len(records) != manifest["num_stories"] or len({row["story_id"] for row in records}) != len(records):
        raise ValueError("wrong or duplicate story count")
    actual_files = {path.name for path in (output / "cases").glob("*.json")}
    if actual_files != {f"{row['story_id']}.json" for row in records}:
        raise ValueError("unexpected or missing case files")
    prompt_hashes = set()
    verified_assets = set()
    shot_count = 0
    for row in records:
        case_path = output / row["relative_path"]
        if sha256_file(case_path) != row["case_sha256"]:
            raise ValueError(f"case hash mismatch: {case_path}")
        case = json.loads(case_path.read_text(encoding="utf-8"))
        if validate_source_asset_locators(load_story_source(case_path))["asset_count"] != row["num_uploads"]:
            raise ValueError("canonical source loader upload count mismatch")
        expected_keys = {"story_id", "title", "uploaded_assets_by_shot", "prompts"}
        if row["has_audio"]:
            expected_keys.add("metadata")
            if not case.get("metadata") or set(case["metadata"]) - {"primary_character", "secondary_character"}:
                raise ValueError("voice cases require only source character declarations as metadata")
        if set(case) != expected_keys or case["story_id"] != row["story_id"]:
            raise ValueError("case must contain only ID, title, uploads and prompts")
        prompt_hash = prompt_digest(case["prompts"])
        if prompt_hash != row["prompts_sha256"] or prompt_hash in prompt_hashes:
            raise ValueError("changed or duplicate prompt sequence")
        prompt_hashes.add(prompt_hash)
        shot_count += len(case["prompts"])
        if len(case["prompts"]) != row["num_shots"]:
            raise ValueError("wrong shot count")
        uploads = {(shot, upload["upload_id"]): upload for shot, items in case["uploaded_assets_by_shot"].items() for upload in items}
        if len(uploads) != row["num_uploads"] or len(row["assets"]) != len(uploads):
            raise ValueError("wrong upload count")
        has_audio = any(upload.get("declared_asset_type") == "voice_reference" for upload in uploads.values())
        if has_audio != row["has_audio"] or (not uploads) != (row["upload_combination"] == "no_upload"):
            raise ValueError("input group does not match actual uploads")
        for asset in row["assets"]:
            upload = uploads[(asset["upload_shot_key"], asset["upload_id"])]
            actual_path = (case_path.parent / upload[asset["locator"]]).resolve()
            if actual_path != (output / asset["path"]).resolve():
                raise ValueError("upload path mismatch")
            if str(actual_path) not in verified_assets:
                if sha256_file(actual_path) != asset["sha256"]:
                    raise ValueError("reference hash mismatch")
                verified_assets.add(str(actual_path))
    if shot_count != manifest["num_shots"]:
        raise ValueError("manifest shot count mismatch")
    groups = input_groups(records)
    group_fields = {"no_upload": "no_upload_stories", "image_only": "image_only_stories", "image_audio": "audio_stories"}
    for group, field in group_fields.items():
        if field not in manifest:
            continue
        if len(groups[group]) != manifest[field]:
            raise ValueError(f"wrong {group} story count")
        subset = json.loads((output / f"{group}_manifest.json").read_text(encoding="utf-8"))["records"]
        expected = {row["story_id"]: row for row in groups[group]}
        if len(subset) != len(expected) or {row["story_id"] for row in subset} != set(expected):
            raise ValueError(f"wrong {group} subset membership")
        for row in subset:
            if any(row[key] != expected[row["story_id"]][key] for key in ("path", "relative_path", "num_shots")):
                raise ValueError(f"wrong {group} subset record")
    return {"status": "passed", "num_stories": len(records), "num_shots": shot_count,
            "unique_asset_files": len(verified_assets), "original_prompts_unchanged": True,
            "canonical_source_loader_validated": True,
            **{field: len(groups[group]) for group, field in group_fields.items()}}


def gallery(output: Path, records: list[dict[str, Any]]) -> None:
    cards = []
    for row in records:
        case = json.loads((output / row["relative_path"]).read_text(encoding="utf-8"))
        media = []
        for shot, uploads in case["uploaded_assets_by_shot"].items():
            for upload in uploads:
                locator = next(key for key in LOCATORS if key in upload)
                path = html.escape(upload[locator].removeprefix("../"), quote=True)
                description = html.escape(upload.get("user_description", ""))
                element = f'<audio controls preload="none" src="{path}"></audio>' if locator == "audio_path" else f'<img loading="lazy" src="{path}" alt="{description}">'
                media.append(f'<figure>{element}<figcaption>shot {int(shot) + 1} 上传 · {html.escape(upload["declared_asset_type"])}<br>{description}</figcaption></figure>')
        prompts = "".join(f"<li>{html.escape(prompt)}</li>" for prompt in case["prompts"])
        if not media:
            media.append("<p>无上传素材 · 仅使用原始逐镜头文本请求，不注入离线素材库。</p>")
        cards.append(f'<article data-upload="{row["upload_combination"]}" data-style="{html.escape(row["story_style"], quote=True)}"><h2>{row["story_id"]} — {html.escape(case["title"])}</h2><p>{html.escape(row["story_style"])} · {row["num_shots"]} shots · {row["upload_timing"]}</p><div class="media">{"".join(media)}</div><details><summary>展开全部 shot prompts</summary><ol>{prompts}</ol></details></article>')
    upload_options = "".join(f'<option>{html.escape(value)}</option>' for value in sorted({row["upload_combination"] for row in records}))
    style_options = "".join(f'<option>{html.escape(value)}</option>' for value in sorted({row["story_style"] for row in records}))
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>REEL-Bench 300</title>
<style>body{{font:16px/1.6 system-ui;background:#f3f5f7;color:#16232e;max-width:1250px;margin:32px auto;padding:0 24px}}article{{background:white;border-radius:12px;padding:20px;margin:18px 0}}h2{{font-size:20px}}.media{{display:flex;gap:12px;flex-wrap:wrap}}figure{{margin:0;width:260px}}img{{height:200px;width:260px;object-fit:contain;background:#eef1f4;border-radius:8px}}audio{{width:260px}}figcaption{{font-size:12px;color:#63717c}}select,input{{padding:10px;margin:6px;max-width:480px}}summary{{cursor:pointer;margin-top:16px}}li{{margin:12px 0}}</style>
<h1>REEL-Bench · 原始上传素材 + 每个 shot 的 prompt</h1><p>{len(records)} stories · {sum(row['num_shots'] for row in records)} shots。来自原有 12,000-story 输入池；不含工具轨迹、生成视频或评测标注。上传时机与 prompts 保持原样。</p>
<select id="upload"><option value="">全部上传组合</option>{upload_options}</select><select id="style"><option value="">全部故事风格</option>{style_options}</select><input id="query" placeholder="搜索"><span id="count"></span>
{''.join(cards)}<script>
const controls=['upload','style','query'].map(name=>document.getElementById(name));
function filter(){{let shown=0;for(const card of document.querySelectorAll('article')){{const match=(!controls[0].value||card.dataset.upload===controls[0].value)&&(!controls[1].value||card.dataset.style===controls[1].value)&&card.textContent.toLowerCase().includes(controls[2].value.toLowerCase());card.hidden=!match;if(match)shown++;}}document.getElementById('count').textContent=shown+' stories';}}
controls.forEach(control=>control.addEventListener('input',filter));filter();</script></html>"""
    (output / "preview.html").write_text(page, encoding="utf-8")


def build(source_manifest: Path, output: Path, count: int, audio_count: int, seed: int, workers: int, *, no_upload_count: int = 60) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(inspect_source, source["records"]))
    print(json.dumps({"source_stories": len(rows), "with_uploads": sum(bool(row["num_uploads"]) for row in rows),
                      "without_uploads": sum(not row["num_uploads"] for row in rows),
                      "missing_asset_stories": sum(bool(row["missing_assets"]) for row in rows)}), flush=True)
    selected = select_candidates(rows, count, audio_count, seed, no_upload_count=no_upload_count)
    records = [export_case(row, output) for row in selected]
    manifest = {
        "schema_version": "reel_bench.uploads_prompts.v1", "status": "curated_source_inputs",
        "source_manifest": str(source_manifest.resolve()), "source_manifest_sha256": sha256_file(source_manifest),
        "seed": seed, "num_stories": len(records), "num_shots": sum(row["num_shots"] for row in records),
        "num_uploads": sum(row["num_uploads"] for row in records), "audio_stories": audio_count,
        "no_upload_stories": no_upload_count, "image_only_stories": count - audio_count - no_upload_count,
        "coverage": coverage(records),
        "selection_policy": "balanced upload combinations and visual styles; greedy coverage of tasks, subjects, props, scenes, upload timing and shot lengths; distinct exact prompt sequences",
        "records": records,
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "source_inventory.json", {
        "source_manifest": str(source_manifest.resolve()), "num_stories": len(rows),
        "num_shots": sum(row["num_shots"] for row in rows),
        "stories_with_uploads": sum(bool(row["num_uploads"]) for row in rows),
        "stories_without_uploads": sum(not row["num_uploads"] for row in rows),
        "missing_asset_stories": [row["story_id"] for row in rows if row["missing_assets"]],
        "coverage": coverage(rows),
    })
    write_subset_manifests(output, records)
    gallery(output, records)
    result = validate(output)
    write_json(output / "validation.json", result)
    return {**result, "audio_stories": audio_count, "coverage": manifest["coverage"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Select diverse source inputs, never teacher trajectories, for REEL-Bench.")
    parser.add_argument("command", choices=["build", "validate"])
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--audio-count", type=int, default=30)
    parser.add_argument("--no-upload-count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    result = validate(args.output) if args.command == "validate" else build(args.source_manifest, args.output, args.count, args.audio_count, args.seed, args.workers, no_upload_count=args.no_upload_count)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
