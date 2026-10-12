"""Collect single-shot candidate groups for one accumulated actor update."""

from __future__ import annotations

import asyncio
from typing import Any

from verl import DataProto


def validate_accumulation_config(config: Any) -> int:
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    rounds = int(config.data.train_batch_size)
    world_size = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    if rounds < 1 or int(actor.ppo_mini_batch_size) != rounds:
        raise ValueError("single-shot collection requires one or more rounds in one PPO mini-batch")
    if world_size < 2 or int(rollout.n) != world_size:
        raise ValueError("single-shot collection requires one candidate per data-parallel GPU")
    if (int(actor.ppo_micro_batch_size_per_gpu) != 1 or actor.use_dynamic_bsz
            or int(actor.ulysses_sequence_parallel_size) != 1 or int(actor.ppo_epochs) != 1):
        raise ValueError("single-shot accumulation requires fixed micro-batches of one, SP=1, PPO epochs=1")
    if str(config.algorithm.adv_estimator) != "grpo":
        raise ValueError("single-shot accumulation requires group-relative advantages")
    return rounds


def single_shot_rounds(prompts: DataProto, candidates: int, workers: int) -> list[DataProto]:
    if candidates < 2 or workers < 1 or candidates % workers:
        raise ValueError("each complete candidate group must divide evenly among agent workers")
    if not len(prompts) or len(prompts) % candidates or "uid" not in prompts.non_tensor_batch:
        raise ValueError("single-shot rounds require complete candidate groups with source UIDs")
    groups = prompts.split(candidates)
    seen = set()
    for group in groups:
        identifiers = {str(value) for value in group.non_tensor_batch["uid"]}
        if len(identifiers) != 1 or identifiers & seen:
            raise ValueError("each round must contain exactly one distinct shot UID with all its candidates")
        seen.update(identifiers)
    return groups


async def generate_accumulated_rollouts(manager: Any, prompts: DataProto) -> DataProto:
    rounds = validate_accumulation_config(manager.config)
    groups = single_shot_rounds(prompts, int(manager.rollout_config.n), len(manager.agent_loop_workers))
    if len(groups) != rounds:
        raise ValueError("the rollout round count differs from the optimizer accumulation batch")
    outputs = []
    for round_index, group in enumerate(groups, 1):
        prefix = (f"[storymem-accum] step={prompts.meta_info.get('global_steps', '?')} "
                  f"round={round_index}/{rounds} shots=1 candidates={len(group)}")
        print(f"{prefix} status=generating", flush=True)
        chunks = group.chunk(len(manager.agent_loop_workers))
        generated = await asyncio.gather(*[
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(manager.agent_loop_workers, chunks, strict=True)
        ])
        if sum(len(output) for output in generated) != len(group):
            raise ValueError("single-shot accumulation cannot expand or discard candidate rows")
        outputs.extend(generated)
        print(f"{prefix} status=completed", flush=True)
    output = DataProto.concat(outputs)
    metrics = [chunk.meta_info.pop("metrics") for chunk in outputs]
    output.meta_info = {"timing": manager._performance_metrics(metrics, output), **outputs[0].meta_info}
    print(f"[storymem-accum] step={prompts.meta_info.get('global_steps', '?')} "
          f"ready_for_actor candidates={len(output)} gradient_accumulation_steps={rounds}", flush=True)
    return output
