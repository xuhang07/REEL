from __future__ import annotations

import copy
from pathlib import Path

from . import gemini_reward as transport
from . import reel_joint_reward as joint


def evaluate_episode(trajectory_path):
    path = Path(trajectory_path).resolve()
    trajectory = transport._strict_json_loads(path.read_text(), label="joint multi-shot episode")
    if trajectory.get("mode") != "multi" or trajectory.get("expected_shots", 0) < 2:
        raise transport.RewardInfrastructureError("joint multi-shot reward needs a multi-shot episode")
    acceptance = joint.validate_acceptance(transport._load_config())
    global_review = transport.evaluate_episode(path, mode="multi")
    local_rewards, evaluations = [], []
    if global_review["policy_failure"]:
        local_rewards = [0.0] * trajectory["expected_shots"]
    else:
        for position, shot in enumerate(trajectory["shots"]):
            directory = path.parent / "joint_shot_rewards" / f"shot_{position:04d}"
            directory.mkdir(parents=True, exist_ok=True)
            single = copy.deepcopy(trajectory)
            single.update(mode="single", expected_shots=1, completed_shots=int(bool(shot.get("completed"))),
                          shots=[copy.deepcopy(shot)], shot_video_paths=[shot["output_video_path"]])
            single["shots"][0]["episode_position"] = 0
            single_path = directory / "trajectory.json"
            transport._atomic_write_json(single_path, single)
            evaluation_path = directory / "reward_joint_evaluation.json"
            result = joint.evaluate_offline(single_path, evaluation_path)
            local_rewards.append(result["reward"])
            evaluations.append({"path": str(evaluation_path), "sha256": transport._file_sha256(evaluation_path)})
    review = {**global_review, "schema_version": "reel.joint-multishot-online.v1", "reward_version": joint.VERSION,
              "online_applied": True, "acceptance": acceptance, "local_shot_rewards": local_rewards,
              "joint_shot_evaluations": evaluations,
              "aggregation": "rollout mixes joint local reward and episode reward using STORYMEM_RL_MULTI_LOCAL_MIX"}
    review_path = path.parent / "reward_review_joint_multi.json"
    transport._atomic_write_json(review_path, review)
    return {**review, "review_path": str(review_path)}


def validate_online_review(*, trajectory_path, expected_model):
    from .verify_stage import _validate_review
    path = Path(trajectory_path).resolve()
    trajectory = joint.quality._load(path)
    review = joint.quality._load(path.with_name("reward_review_joint_multi.json"))
    count = trajectory["expected_shots"]
    if (review.get("schema_version") != "reel.joint-multishot-online.v1"
            or review.get("reward_version") != joint.VERSION
            or review.get("mode") != "multi" or review.get("online_applied") is not True
            or review.get("acceptance") != joint.validate_acceptance()
            or len(review.get("joint_shot_evaluations", [])) != count):
        raise transport.RewardInfrastructureError("invalid joint multi-shot receipt")
    global_reward = _validate_review(path.with_name("reward_review.json"), trajectory_path=path,
        mode="multi", expected_shots=count, expected_model=expected_model,
        video_paths=[Path(shot["output_video_path"]).resolve() for shot in trajectory["shots"]],
        expected_reward_version="v1")
    rewards = []
    for position, record in enumerate(review["joint_shot_evaluations"]):
        directory = path.parent / "joint_shot_rewards" / f"shot_{position:04d}"
        evaluation_path = directory / "reward_joint_evaluation.json"
        if record != {"path": str(evaluation_path), "sha256": transport._file_sha256(evaluation_path)}:
            raise transport.RewardInfrastructureError("joint local receipt changed")
        evaluation = joint.quality._load(evaluation_path)
        local_review = {"schema_version": joint.ONLINE_SCHEMA, "reward_version": joint.VERSION,
            "mode": "single", "online_applied": True, "judge_called": evaluation["judge_called"],
            "policy_failure": evaluation["policy_failure"], "acceptance": review["acceptance"],
            "evaluation": evaluation, "reward": evaluation["reward"],
            "local_shot_rewards": [evaluation["reward"]]}
        rewards.append(joint.validate_online_review(local_review,
            trajectory_path=directory / "trajectory.json", expected_model=expected_model))
    if review["reward"] != global_reward or review["local_shot_rewards"] != rewards:
        raise transport.RewardInfrastructureError("joint multi-shot reward mismatch")
    return global_reward
