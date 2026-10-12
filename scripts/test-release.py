import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


class ReleaseTests(unittest.TestCase):
    def test_python_sources_compile(self):
        files = [path for directory in ("tools", "verl_recipe") for path in (ROOT / directory).rglob("*.py")]
        with ThreadPoolExecutor(max_workers=32) as executor:
            list(executor.map(lambda path: ast.parse(path.read_text(), filename=str(path)), files))
        self.assertGreater(len(files), 80)

    def test_no_private_deployment_defaults_or_credentials(self):
        pattern = re.compile(r"/pfs\d*/(?:xuhang|daiwenxun|shared_models)|(?:https?://)?10\.(?:2\.104\.108|3\.34\.120|3\.44\.30|3\.43\.160)|(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|olp_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})")
        files = [path for directory in ("tools", "verl_recipe", "prompts", "Director_Agent", "patches", "docs") for path in (ROOT / directory).rglob("*") if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".md", ".html", ".js", ".css", ".patch"}]
        with ThreadPoolExecutor(max_workers=32) as executor:
            findings = list(executor.map(lambda path: str(path.relative_to(ROOT)) if pattern.search(path.read_text()) else None, files))
        self.assertEqual([finding for finding in findings if finding], [])

    def test_prompt_resources_match_versioned_manifest(self):
        resource_root = ROOT / "tools/storymem_teacher/resources"
        manifest = json.loads((resource_root / "manifest.json").read_text())
        for resource in manifest["resources"].values():
            root = ROOT / "Director_Agent/pe" if resource.get("source_root") == "director_pe" else resource_root
            digest = "sha256:" + hashlib.sha256((root / resource["path"]).read_bytes()).hexdigest()
            self.assertEqual(digest, resource["sha256"], resource["path"])

    def test_public_resource_revision_retains_provenance(self):
        manifest = json.loads((ROOT / "tools/storymem_teacher/resources/manifest.json").read_text())
        resource = manifest["resources"]["director.video.v7.shot_prompt_writer"]
        self.assertEqual(manifest["revision"], "reel-public-20261012")
        self.assertEqual(resource["previously_declared_sha256"], "sha256:6eb1ec73a339590ec45d0c7b14c9b0988a90022388a91c7f06dac69067a826f6")
        self.assertIn("not an assertion", resource["release_note"])

    def test_nine_tools_and_schema_validity(self):
        from jsonschema import Draft202012Validator
        from storymem_policy.contract import CANONICAL_TOOL_NAMES, get_tool_schemas

        schemas = get_tool_schemas()
        self.assertEqual(len(schemas), 9)
        self.assertEqual(tuple(schema["function"]["name"] for schema in schemas), CANONICAL_TOOL_NAMES)
        for schema in schemas:
            Draft202012Validator.check_schema(schema["function"]["parameters"])

    def test_backend_clients_import_without_credentials(self):
        from reel_h3_collector import H3BackendAdapter
        from storymem_adapter.echo15 import Echo15BackendAdapter

        self.assertTrue(callable(H3BackendAdapter))
        self.assertTrue(callable(Echo15BackendAdapter))

    def test_no_credential_or_weight_files(self):
        disallowed = {".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".onnx", ".npz", ".npy"}
        files = [path for directory in ("tools", "verl_recipe", "prompts", "Director_Agent", "patches", "docs") for path in (ROOT / directory).rglob("*") if path.is_file()]
        self.assertEqual([str(path.relative_to(ROOT)) for path in files if path.suffix in disallowed or path.name.endswith(".local.json") or path.name.startswith(".env")], [])


if __name__ == "__main__":
    unittest.main()
