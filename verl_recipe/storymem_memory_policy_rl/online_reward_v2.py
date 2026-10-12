"""Explicit, acceptance-gated online adapter for the single-shot v2 evaluator."""

from __future__ import annotations

import argparse
import fcntl
import math
import os
from pathlib import Path
from typing import Any

from . import gemini_reward as legacy
from . import single_shot_reward_v2 as engine


ONLINE_REVIEW_SCHEMA = "storymem.reward-review.single-v2.v1"
ACCEPTANCE_CHECKS = {
    "all_cases_completed", "infrastructure", "complete_groups", "quality_repeatability",
    "full_repeatability", "unconfirmed_fraction", "visual_audit", "visual_preference_pairs",
}


def validate_acceptance(config: legacy.RewardConfig | None = None, *,
                        acceptance_path: str | Path | None = None,
                        acceptance_sha256: str | None = None) -> dict[str, Any]:
    raw_path = str(acceptance_path) if acceptance_path is not None else os.getenv("STORYMEM_RL_REWARD_V2_ACCEPTANCE_PATH", "")
    expected_hash = acceptance_sha256 if acceptance_sha256 is not None else os.getenv("STORYMEM_RL_REWARD_V2_ACCEPTANCE_SHA256", "")
    if not raw_path or not expected_hash:
        raise legacy.RewardInfrastructureError("single_v2 requires a pinned acceptance path and SHA-256")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file() or legacy._file_sha256(path) != expected_hash:
        raise legacy.RewardInfrastructureError("reward-v2 acceptance report is missing or changed")
    receipt = engine._load(path)
    if (receipt.get("schema_version") != "storymem.reward-v2-acceptance.v1"
            or receipt.get("passed") is not True or set(receipt.get("checks", {})) != ACCEPTANCE_CHECKS
            or not all(value is True for value in receipt["checks"].values())):
        raise legacy.RewardInfrastructureError("reward-v2 acceptance has not passed every required check")
    if receipt.get("reward_schema") != engine.REVIEW_VERSION or receipt.get("engine_sha256") != engine.ENGINE_SOURCE_SHA256:
        raise legacy.RewardInfrastructureError("reward-v2 implementation differs from the calibrated version")
    if legacy._file_sha256(Path(engine.__file__)) != engine.ENGINE_SOURCE_SHA256:
        raise legacy.RewardInfrastructureError("reward-v2 source changed after import")
    if config is not None:
        if config.review_repeats < 2 or receipt.get("judge_config_fingerprint") != engine.judge_config_fingerprint(config):
            raise legacy.RewardInfrastructureError("online judge configuration differs from reward-v2 acceptance")
    manifest_path = Path(receipt.get("manifest_path", ""))
    if not manifest_path.is_file() or legacy._file_sha256(manifest_path) != receipt.get("manifest_sha256"):
        raise legacy.RewardInfrastructureError("reward-v2 acceptance manifest changed")
    reports = receipt.get("reports")
    if not isinstance(reports, dict) or len(reports) < 14:
        raise legacy.RewardInfrastructureError("reward-v2 acceptance needs twelve cases and two repeat runs")
    for report in reports.values():
        report_path = Path(report.get("path", ""))
        if not report_path.is_file() or legacy._file_sha256(report_path) != report.get("sha256"):
            raise legacy.RewardInfrastructureError("reward-v2 acceptance evidence changed")
    return {"path": str(path), "sha256": expected_hash, "engine_sha256": receipt["engine_sha256"],
            "judge_config_fingerprint": receipt["judge_config_fingerprint"]}


def evaluate_episode(trajectory_path: str | Path, *, mode: str) -> dict[str, Any]:
    if mode != "single":
        raise legacy.RewardInfrastructureError("single_v2 is not a multi-shot reward; no silent fallback is allowed")
    path = Path(trajectory_path).expanduser().resolve()
    evidence = engine.build_evidence(path)
    config = legacy._load_config() if evidence["protocol"]["valid"] else None
    acceptance = validate_acceptance(config)
    alias_pool = None
    if config is not None and getattr(config, "model_aliases", ()):
        from .gemini_alias_pool import validate_pool_approval

        alias_pool = validate_pool_approval(config)
    review_path = path.with_name("reward_review.json")
    with review_path.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if review_path.exists() and engine._load(review_path).get("schema_version") != ONLINE_REVIEW_SCHEMA:
            raise legacy.RewardInfrastructureError("refusing to overwrite an existing reward from another version")
        evaluation_path = path.with_name("reward_v2_evaluation.json")
        detailed = engine._evaluate_locked(path, evaluation_path)
        reward = float(detailed["reward"])
        if not math.isfinite(reward) or not -1e-12 <= reward <= 1.0 + 1e-12:
            raise legacy.RewardInfrastructureError("single_v2 returned an invalid reward")
        reward = min(1.0, max(0.0, reward))
        result = {
            "schema_version": ONLINE_REVIEW_SCHEMA, "reward_version": "single_v2", "mode": "single",
            "reward": reward, "local_shot_rewards": [reward],
            "policy_failure": detailed["policy_failure"], "judge_called": detailed["judge_called"],
            "trajectory_path": str(path), "trajectory_input_sha256": evidence["trajectory_input_sha256"],
            "video_paths": [item["path"] for item in evidence["media"] if item["media_id"] == "TARGET_VIDEO"],
            "acceptance": acceptance, "evaluation": detailed, "online_applied": True,
            "evaluation_path": str(evaluation_path), "evaluation_sha256": legacy._file_sha256(evaluation_path),
        }
        if alias_pool is not None:
            result["alias_pool"] = alias_pool
        legacy._atomic_write_json(review_path, result)
        return {**result, "review_path": str(review_path)}


def validate_online_review(review: dict[str, Any], *, trajectory_path: Path, expected_model: str) -> float:
    if review.get("schema_version") != ONLINE_REVIEW_SCHEMA or review.get("reward_version") != "single_v2":
        raise legacy.RewardInfrastructureError("not a single_v2 online receipt")
    if review.get("mode") != "single" or review.get("online_applied") is not True:
        raise legacy.RewardInfrastructureError("single_v2 receipt is not an applied single-shot reward")
    evidence = engine.build_evidence(trajectory_path)
    if not evidence["protocol"]["valid"] or review.get("policy_failure") is not False or review.get("judge_called") is not True:
        raise legacy.RewardInfrastructureError("acceptance verification needs a completed judged trajectory")
    if review.get("trajectory_path") != str(trajectory_path.resolve()):
        raise legacy.RewardInfrastructureError("single_v2 receipt belongs to another trajectory")
    if review.get("trajectory_input_sha256") != evidence["trajectory_input_sha256"]:
        raise legacy.RewardInfrastructureError("single_v2 trajectory input changed")
    expected_videos = [item["path"] for item in evidence["media"] if item["media_id"] == "TARGET_VIDEO"]
    if review.get("video_paths") != expected_videos:
        raise legacy.RewardInfrastructureError("single_v2 target video differs from its receipt")
    detailed = engine._object(review.get("evaluation"), "single_v2 evaluation")
    if detailed.get("schema_version") != engine.REVIEW_VERSION or detailed.get("engine_sha256") != engine.ENGINE_SOURCE_SHA256:
        raise legacy.RewardInfrastructureError("single_v2 evaluation version changed")
    evaluation_path = Path(review.get("evaluation_path", ""))
    if (not evaluation_path.is_file() or legacy._file_sha256(evaluation_path) != review.get("evaluation_sha256")
            or engine._load(evaluation_path) != detailed):
        raise legacy.RewardInfrastructureError("single_v2 detailed evaluation changed")
    acceptance = validate_acceptance()
    if review.get("acceptance") != acceptance or detailed.get("judge_config_fingerprint") != acceptance["judge_config_fingerprint"]:
        raise legacy.RewardInfrastructureError("single_v2 receipt is not bound to the accepted judge configuration")
    for key, value in evidence.items():
        if key != "trajectory_sha256" and detailed.get("evidence", {}).get(key) != value:
            raise legacy.RewardInfrastructureError(f"single_v2 evidence changed: {key}")
    quality = detailed.get("quality_reviews", [])
    if len(quality) < 2:
        raise legacy.RewardInfrastructureError("single_v2 requires at least two blind quality reviews")
    for item in quality:
        engine.validate_quality(item, evidence)
    engine.validate_decision(detailed.get("decision_review"), evidence)
    expected = engine.aggregate(quality, detailed["decision_review"])
    for key, value in expected.items():
        if detailed.get(key) != value:
            raise legacy.RewardInfrastructureError(f"single_v2 aggregation mismatch: {key}")
    reward = review.get("reward")
    if (isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward)
            or not 0 < reward <= 1 or not math.isclose(reward, expected["reward"], rel_tol=0, abs_tol=1e-12)
            or review.get("local_shot_rewards") != [reward]):
        raise legacy.RewardInfrastructureError("single_v2 online/local rewards do not match the evaluation")
    judge = detailed.get("judge", {})
    receipts = judge.get("receipts", [])
    allowed_models = {expected_model}
    if review.get("alias_pool") is not None:
        from .gemini_alias_pool import validate_pool_approval

        approved_pool = validate_pool_approval(primary_model=expected_model, pool_receipt=review["alias_pool"])
        allowed_models = set(approved_pool["models"])
    if judge.get("model") != expected_model or len(receipts) != len(quality) + 1:
        raise legacy.RewardInfrastructureError("single_v2 judge model or receipt count is invalid")
    stages = [item.get("stage") for item in receipts]
    valid_stages = [["quality"] * len(quality) + ["decision"],
                    ["quality"] * (len(quality) - 1) + ["critical_recheck", "decision"]]
    if stages not in valid_stages:
        raise legacy.RewardInfrastructureError("single_v2 quality/decision receipts are not aligned")
    for receipt in receipts:
        if isinstance(receipt.get("attempts"), bool) or not isinstance(receipt.get("attempts"), int) or receipt["attempts"] < 1:
            raise legacy.RewardInfrastructureError("single_v2 receipt has no valid attempt count")
        provenance = receipt.get("provenance", {})
        if (provenance.get("model") not in allowed_models or provenance.get("api_style") != "gemini"
                or provenance.get("assistant_source") != "remote_closed_model_api"):
            raise legacy.RewardInfrastructureError("single_v2 receipt lacks genuine provider provenance")
        for field in ("endpoint_sha256", "request_payload_sha256", "request_sha256", "response_sha256", "assistant_text_sha256"):
            legacy._declared_sha256(provenance.get(field), label=field)
        if provenance["request_payload_sha256"] != provenance["request_sha256"]:
            raise legacy.RewardInfrastructureError("single_v2 request hashes disagree")
    return float(reward)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-acceptance", action="store_true", required=True)
    parser.add_argument("--without-provider-config", action="store_true")
    args = parser.parse_args()
    config = None if args.without_provider_config else legacy._load_config()
    validate_acceptance(config)
    from .gemini_alias_pool import load_model_aliases, validate_pool_approval

    primary_model = os.getenv("STORYMEM_RL_GEMINI_MODEL", legacy.DEFAULT_GEMINI_MODEL)
    if load_model_aliases(primary_model):
        validate_pool_approval(config, primary_model=primary_model)
    print("[reward-v2] pinned acceptance and implementation validated", flush=True)


if __name__ == "__main__":
    main()
