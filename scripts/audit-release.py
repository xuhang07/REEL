import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re


PACKAGE_ROOTS = (
    "tools/storymem_policy", "tools/storymem_adapter", "tools/storymem_teacher",
    "tools/reel_bench", "verl_recipe/storymem_memory_policy_rl",
)
ENTRY_POINTS = (
    "tools/storymem_policy_collector.py",
    "tools/run_storymem_stbench_qwen35_policy.py",
    "tools/run_reel_bench.py",
    "tools/evaluate_reel_bench.py",
    "tools/evaluate_storymem_stbench_official.py",
    "tools/prepare_storymem_memory_policy_qwen35_sft.py",
    "tools/reel_h3_collector.py",
    "tools/test_reel_h3_adapter.py",
    "tools/test_reel_h3_transport.py",
    "tools/test_storymem_policy_contract.py",
    "tools/test_storymem_policy_state.py",
    "tools/test_storymem_adapter_pe.py",
)
PATTERNS = {
    "credential_token": re.compile(r"(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|olp_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})"),
    "assigned_secret": re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|password|secret[_-]?key)\s*(?:=|:)\s*[\"']([^\"'\n]{12,})[\"']"),
    "private_network": re.compile(r"(?:https?://)?(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)(?::\d+)?"),
    "private_path": re.compile(r"/(?:pfs\d*|root|home/[A-Za-z0-9_-]+)/[^\s\"'`]*"),
    "credential_file": re.compile(r"(?i)(?:\.env|provider.*\.json|launch_echo.*\.sh|\.git-credentials)"),
}


def inspect_file(item):
    path, root = item
    relative = path.relative_to(root).as_posix()
    content = path.read_text()
    dependencies = set()
    if path.suffix == ".py":
        parsed = ast.parse(content, filename=relative)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                dependencies.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                dependencies.add(node.module)
    matches = []
    for line_number, line in enumerate(content.splitlines(), 1):
        categories = [name for name, pattern in PATTERNS.items() if pattern.search(line)]
        if categories:
            matches.append({"line": line_number, "categories": categories})
    return {"path": relative, "bytes": path.stat().st_size, "dependencies": sorted(dependencies), "matches": matches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/release-audit.json"))
    args = parser.parse_args()
    root = args.source.resolve()
    selected = set()
    for package in PACKAGE_ROOTS:
        selected.update(path for path in (root / package).rglob("*.py") if "__pycache__" not in path.parts)
    selected.update(root / name for name in ENTRY_POINTS if (root / name).is_file())
    inspected = {}
    external = set()
    with ThreadPoolExecutor(max_workers=32) as workers:
        while selected - inspected.keys():
            pending = sorted(selected - inspected.keys())
            for path, result in zip(pending, workers.map(inspect_file, ((path, root) for path in pending))):
                inspected[path] = result
                for module in result["dependencies"]:
                    module_path = Path(*module.split("."))
                    candidates = [root / "tools" / module_path.with_suffix(".py"), root / module_path.with_suffix(".py")]
                    candidates += [root / "tools" / module_path / "__init__.py", root / module_path / "__init__.py"]
                    found = [candidate for candidate in candidates if candidate.is_file()]
                    if found:
                        for candidate in found:
                            selected.add(candidate)
                            if candidate.name == "__init__.py":
                                selected.update(candidate.parent.glob("*.py"))
                    else:
                        external.add(module.split(".")[0])
    report = {"files": sorted(inspected.values(), key=lambda record: record["path"]), "external_modules": sorted(external)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"files": len(inspected), "bytes": sum(record["bytes"] for record in inspected.values()), "flagged_files": [{"path": record["path"], "matches": record["matches"]} for record in report["files"] if record["matches"]], "external_modules": report["external_modules"]}, indent=2))


if __name__ == "__main__":
    main()
