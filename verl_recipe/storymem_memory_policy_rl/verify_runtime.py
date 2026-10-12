#!/usr/bin/env python3
"""Fail-fast contract check for the exact VERL stack used by StoryMem RL.

This recipe deliberately depends on extensions that are not present in an
unmodified VERL 0.8 installation.  Checking only ``verl.__version__`` is
therefore unsafe: a stock wheel and the patched build report the same version.
The checks below inspect the installed implementation for the multi-output
agent-loop and trainer batch-alignment contracts, then (unless ``--source-only``
is used) exercise the Qwen3.5 processor and rollout-backend registrations.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Sequence


PATCH_CONTRACT = "storymem-verl-multishot-v2"
EXPECTED_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
EXPECTED_MODEL_TYPE = "qwen3_5"


class RuntimeContractError(RuntimeError):
    """The selected environment cannot execute this recipe correctly."""


def _require_fragments(path: Path, fragments: Sequence[str], *, label: str) -> None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeContractError(f"cannot read {label}: {path}: {exc}") from exc
    missing = [fragment for fragment in fragments if fragment not in source]
    if missing:
        rendered = ", ".join(repr(fragment) for fragment in missing)
        raise RuntimeContractError(
            f"{label} does not implement {PATCH_CONTRACT}; missing {rendered}"
        )


def verl_package_root() -> Path:
    spec = importlib.util.find_spec("verl")
    if spec is None or spec.origin is None:
        raise RuntimeContractError("VERL is not installed in the selected Python environment")
    root = Path(spec.origin).resolve().parent
    if not root.is_dir():
        raise RuntimeContractError(f"invalid VERL package root: {root}")
    return root


def verify_patched_verl(expected_version: str, *, require_lora: bool = False) -> dict[str, str]:
    try:
        version = importlib.metadata.version("verl")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeContractError("VERL distribution metadata is missing") from exc
    if version != expected_version:
        raise RuntimeContractError(
            f"expected VERL {expected_version}, found {version}; install the pinned patched build"
        )

    root = verl_package_root()
    if require_lora:
        _require_fragments(
            root.parent / "peft" / "tuners" / "lora" / "torchao.py",
            ("if type(target_base_layer.weight) in (torch.Tensor, torch.nn.Parameter):",),
            label="PEFT ordinary LoRA dispatcher",
        )
    _require_fragments(
        root / "workers" / "engine" / "fsdp" / "transformer_impl.py",
        ("entropy_logits = logits.reshape(-1, logits.shape[-1])",
         "entropy = self.compute_entropy_from_logits(entropy_logits)",
         "entropy = entropy.reshape(entropy_shape)",
         "logits_rmpad = logits_rows[0] if len(logits_rows) == 1 else torch.cat(logits_rows)",
         "inplace_backward=not (calculate_entropy or calculate_sum_pi_squared or distillation_use_topk)"),
        label="VERL padded entropy chunking and logits view",
    )
    _require_fragments(
        root / "workers" / "engine_workers.py",
        ('aggressive_empty_cache(force_sync=True)\n        log_gpu_memory_usage("Before resume weights", logger=logger)',),
        label="VERL rollout weight-resume cache release",
    )
    agent_loop = root / "experimental" / "agent_loop" / "agent_loop.py"
    trainer = root / "trainer" / "ppo" / "ray_trainer.py"
    main_ppo = root / "trainer" / "main_ppo.py"
    dataset = root / "utils" / "dataset" / "rl_dataset.py"
    async_sglang_server = (
        root
        / "workers"
        / "rollout"
        / "sglang_rollout"
        / "async_sglang_server.py"
    )
    if require_lora:
        _require_fragments(
            async_sglang_server,
            ("if self.lora_as_adapter:\n            args.update(",
             "if self.lora_as_adapter:\n            generate_request.lora_path = SGLANG_LORA_NAME",
             "self.rollout_mode == RolloutMode.COLOCATED or self.lora_as_adapter else False"),
            label="VERL/SGLang merged LoRA server and request handling",
        )

    # One input rollout may return K independently contextualized shot outputs.
    # The worker must flatten them while expanding every source non-tensor field.
    _require_fragments(
        agent_loop,
        (
            "nested_outputs = await asyncio.gather(*tasks)",
            "if isinstance(item, (list, tuple)):",
            "outputs.extend(item)",
            "source_indices.extend([i] * len(item))",
            "self._expand_non_tensor_batch(batch.non_tensor_batch, source_indices)",
            "output: AgentLoopOutput | list[AgentLoopOutput] = await agent_loop.run",
            "if isinstance(output, (list, tuple)):",
            "non_tensor_batch.update(extra_fields)",
        ),
        label="VERL agent-loop worker",
    )

    # The PPO trainer must accept the expanded K*N output directly.  Unioning it
    # with the smaller N-expanded source batch corrupts both row alignment and
    # the per-shot UID groups used by GRPO.
    _require_fragments(
        trainer,
        (
            "if len(batch) != len(gen_batch_output):",
            "merged_meta_info = dict(batch.meta_info)",
            "merged_meta_info.update(gen_batch_output.meta_info)",
            "gen_batch_output.meta_info = merged_meta_info",
            "batch = gen_batch_output",
            "adv_kwargs[\"index\"] = data.non_tensor_batch[\"uid\"]",
            "ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n",
            'self.global_steps in self.config.trainer.get("extra_save_steps", [])',
        ),
        label="VERL PPO trainer",
    )
    _require_fragments(
        main_ppo,
        (
            "from verl.trainer.ppo.ray_trainer import RayPPOTrainer",
            "trainer = RayPPOTrainer(",
            "trainer.fit()",
        ),
        label="VERL main_ppo entrypoint",
    )

    # Dataset filtering must load the identical tool configuration.  The
    # launcher also passes this path explicitly as data.tool_config_path.
    _require_fragments(
        dataset,
        (
            'self.tool_config_path = config.get("tool_config_path", None)',
            "load_all_tools(",
            'apply_kwargs["tools"] = self.tool_schemas',
        ),
        label="VERL RL dataset",
    )
    _require_fragments(
        async_sglang_server,
        (
            "scheduler_init_result,",
            "self.scheduler_info = scheduler_init_result.scheduler_infos[0]",
            '"video_data": video_data,',
        ),
        label="VERL/SGLang 0.5.10 async server adapter",
    )
    return {
        "contract": PATCH_CONTRACT,
        "verl_version": version,
        "verl_root": str(root),
    }


def verify_qwen35_stack(model_path: Path, backend: str) -> dict[str, str]:
    try:
        import transformers
        from transformers import AutoConfig
        from verl.utils.tokenizer import hf_processor
    except Exception as exc:
        raise RuntimeContractError(
            f"cannot import the Qwen3.5/VERL runtime ({type(exc).__name__}: {exc})"
        ) from exc

    try:
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
    except Exception as exc:
        raise RuntimeContractError(
            f"Transformers cannot load Qwen3.5 config ({type(exc).__name__}: {exc})"
        ) from exc
    architectures = tuple(getattr(config, "architectures", ()) or ())
    if getattr(config, "model_type", None) != EXPECTED_MODEL_TYPE:
        raise RuntimeContractError(
            f"checkpoint model_type is {getattr(config, 'model_type', None)!r}, "
            f"expected {EXPECTED_MODEL_TYPE!r}"
        )
    if EXPECTED_ARCHITECTURE not in architectures:
        raise RuntimeContractError(
            f"checkpoint architectures do not contain {EXPECTED_ARCHITECTURE}: {architectures!r}"
        )
    if getattr(transformers, EXPECTED_ARCHITECTURE, None) is None:
        raise RuntimeContractError(
            f"Transformers {transformers.__version__} does not expose {EXPECTED_ARCHITECTURE}"
        )

    try:
        processor = hf_processor(
            str(model_path), trust_remote_code=True, local_files_only=True, use_fast=True
        )
    except Exception as exc:
        raise RuntimeContractError(
            f"VERL failed to construct the Qwen3.5 processor ({type(exc).__name__}: {exc})"
        ) from exc
    if processor is None:
        raise RuntimeContractError(
            "verl.utils.tokenizer.hf_processor returned None for Qwen3.5; "
            "the selected VERL build lacks Qwen3.5 multimodal processor support"
        )
    if not callable(getattr(processor, "apply_chat_template", None)):
        raise RuntimeContractError("Qwen3.5 processor has no apply_chat_template method")
    if not callable(getattr(processor, "get_rope_index", None)):
        raise RuntimeContractError(
            "Qwen3.5 processor has no VERL-bound get_rope_index implementation"
        )

    # Verify the module launched by ``python -m verl.trainer.main_ppo`` resolves
    # to the same patched trainer implementation inspected above.
    try:
        main_ppo_module = importlib.import_module("verl.trainer.main_ppo")
        trainer_module = importlib.import_module("verl.trainer.ppo.ray_trainer")
        entrypoint_trainer = getattr(main_ppo_module, "RayPPOTrainer")
        patched_trainer = getattr(trainer_module, "RayPPOTrainer")
    except Exception as exc:
        raise RuntimeContractError(
            f"cannot resolve main_ppo trainer ({type(exc).__name__}: {exc})"
        ) from exc
    if entrypoint_trainer is not patched_trainer:
        raise RuntimeContractError(
            "verl.trainer.main_ppo does not reference verl.trainer.ppo.ray_trainer.RayPPOTrainer"
        )
    loaded_trainer_path = Path(inspect.getsourcefile(patched_trainer) or "").resolve()
    expected_trainer_path = (
        verl_package_root() / "trainer" / "ppo" / "ray_trainer.py"
    ).resolve()
    if loaded_trainer_path != expected_trainer_path:
        raise RuntimeContractError(
            f"main_ppo loaded trainer from {loaded_trainer_path}, expected {expected_trainer_path}"
        )

    if backend == "vllm":
        try:
            import vllm
            from vllm.model_executor.models import ModelRegistry
        except Exception as exc:
            raise RuntimeContractError(
                f"cannot import vLLM ({type(exc).__name__}: {exc})"
            ) from exc
        supported = set(ModelRegistry.get_supported_archs())
        if EXPECTED_ARCHITECTURE not in supported:
            raise RuntimeContractError(
                f"vLLM does not register {EXPECTED_ARCHITECTURE}"
            )
        backend_version = str(getattr(vllm, "__version__", "unknown"))
    elif backend == "sglang":
        try:
            import sglang
            registry_module = importlib.import_module("sglang.srt.models.registry")
        except Exception as exc:
            raise RuntimeContractError(
                f"cannot import SGLang ({type(exc).__name__}: {exc})"
            ) from exc
        registry = getattr(registry_module, "ModelRegistry", None)
        supported = False
        for owner in (registry, registry_module):
            if owner is None:
                continue
            for name in ("models", "_models", "_model_registry", "MODEL_REGISTRY"):
                candidate = getattr(owner, name, None)
                if isinstance(candidate, dict) and EXPECTED_ARCHITECTURE in candidate:
                    supported = True
        if not supported and registry is not None:
            for name in ("load_model_cls", "get_model_cls"):
                method = getattr(registry, name, None)
                if not callable(method):
                    continue
                try:
                    supported = method(EXPECTED_ARCHITECTURE) is not None
                except Exception:
                    supported = False
                if supported:
                    break
        if not supported:
            raise RuntimeContractError(
                f"SGLang does not register {EXPECTED_ARCHITECTURE}"
            )
        sglang_root = Path(inspect.getsourcefile(sglang) or "").resolve().parent
        _require_fragments(
            sglang_root / "srt" / "utils" / "aio_rwlock.py",
            ("self._cond = asyncio.Condition()", "self._lock = self._cond"),
            label="SGLang Python 3.10 RWLock adapter",
        )
        # The online agent loop owns separate prompt representations: raw
        # one-placeholder-per-media ids go to SGLang, and processor-expanded
        # ids go back to VERL for training.  No SGLang video-token regex patch
        # is part of the runtime contract; matching expanded Qwen3.5 frame
        # spans would incorrectly turn one timestamped video into many items.
        backend_version = str(getattr(sglang, "__version__", "unknown"))
    else:
        raise RuntimeContractError(f"unsupported rollout backend {backend!r}")

    return {
        "transformers_version": str(transformers.__version__),
        "processor_class": type(processor).__name__,
        "backend": backend,
        "backend_version": backend_version,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--backend", choices=("sglang", "vllm"), required=True)
    parser.add_argument("--expected-verl-version", default="0.8.0")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--require-lora", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise RuntimeContractError(f"model directory does not exist: {model_path}")
    result = verify_patched_verl(args.expected_verl_version, require_lora=args.require_lora)
    if not args.source_only:
        result.update(verify_qwen35_stack(model_path, args.backend))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeContractError as exc:
        raise SystemExit(f"[storymem-runtime][fatal] {exc}") from None
