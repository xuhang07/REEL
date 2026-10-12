#!/usr/bin/env python3
"""Run deterministic next-tool-call smoke tests on held-out Qwen3.5 SFT rows.

The input rows use LLaMA-Factory v1's native ``{type, value}`` message
representation.  This script deliberately reuses the same Renderer used for
training so tool definitions, media placeholders, and the chat template cannot
silently drift between training and evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


TOOL_CALL_RE = re.compile(
    r"\s*<tool_call>\s*<function=\s*([^\s<>]+)\s*>\s*(.*?)\s*</function>\s*</tool_call>\s*",
    re.DOTALL,
)
PARAMETER_RE = re.compile(r"<parameter=\s*([^<>]+?)\s*>\s*(.*?)\s*</parameter>", re.DOTALL)


def parse_case(value: str) -> tuple[int, int]:
    try:
        row_index, message_index = (int(part) for part in value.split(":", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("case must be ROW_INDEX:MESSAGE_INDEX") from exc
    if row_index < 0 or message_index < 0:
        raise argparse.ArgumentTypeError("case indexes must be non-negative")
    return row_index, message_index


def load_rows(path: Path, indexes: set[int]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index in indexes:
                rows[row_index] = json.loads(line)
                if len(rows) == len(indexes):
                    break
    missing = sorted(indexes - rows.keys())
    if missing:
        raise IndexError(f"validation rows not found: {missing}")
    return rows


def expected_call(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("role") != "assistant":
        raise ValueError("target message must have role=assistant")
    blocks = message.get("content")
    if not isinstance(blocks, list) or len(blocks) != 1 or blocks[0].get("type") != "tool_call":
        raise ValueError("target assistant message must contain exactly one tool_call block")
    value = json.loads(blocks[0]["value"])
    if not isinstance(value, dict) or not isinstance(value.get("name"), str) or not isinstance(value.get("arguments"), dict):
        raise ValueError("target tool_call must contain string name and object arguments")
    return value


def _parse_parameter_value(raw_value: str) -> Any:
    value = raw_value.strip()
    # Qwen3.5's official Jinja template renders top-level Python None/bools via
    # ``|string`` rather than JSON.  Normalize those native wire values before
    # JSON-Schema validation.
    literals = {"None": None, "True": True, "False": False}
    if value in literals:
        return literals[value]
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_qwen35_tool_call(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    cleaned = text.replace("<|im_end|>", "").strip()
    match = TOOL_CALL_RE.fullmatch(cleaned)
    if match is None:
        return None, ["output is not exactly one native Qwen3.5 tool call"]

    name, parameter_body = match.groups()
    arguments: dict[str, Any] = {}
    spans: list[tuple[int, int]] = []
    for parameter_match in PARAMETER_RE.finditer(parameter_body):
        key, raw_value = parameter_match.groups()
        key = key.strip()
        if key in arguments:
            errors.append(f"duplicate parameter: {key}")
        arguments[key] = _parse_parameter_value(raw_value)
        spans.append(parameter_match.span())

    residue_parts = []
    cursor = 0
    for start, end in spans:
        residue_parts.append(parameter_body[cursor:start])
        cursor = end
    residue_parts.append(parameter_body[cursor:])
    if "".join(residue_parts).strip():
        errors.append("unparsed content inside function block")

    return {"name": name, "arguments": arguments}, errors


def tool_schemas(tools_text: str) -> dict[str, dict[str, Any]]:
    tools = json.loads(tools_text)
    if not isinstance(tools, list):
        tools = [tools]
    schemas: dict[str, dict[str, Any]] = {}
    for item in tools:
        function = item.get("function", item)
        name = function.get("name")
        parameters = function.get("parameters")
        if isinstance(name, str) and isinstance(parameters, dict):
            schemas[name] = parameters
    return schemas


def validate_call(call: dict[str, Any] | None, tools_text: str) -> list[str]:
    if call is None:
        return ["tool call could not be parsed"]
    schemas = tool_schemas(tools_text)
    if call["name"] not in schemas:
        return [f"tool is not exposed: {call['name']}"]
    return [error.message for error in Draft202012Validator(schemas[call["name"]]).iter_errors(call["arguments"])]


def _unique_string_set(value: Any) -> frozenset[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    items = frozenset(value)
    if len(items) != len(value):
        return None
    return items


def argument_match_metrics(
    expected: dict[str, Any], predicted: dict[str, Any] | None
) -> dict[str, bool | None]:
    """Compare calls while relaxing only generate_image's new subject set.

    ``generate_image.arguments.subject_ids`` declares which registry subjects a
    new image should contain; its order has no runtime meaning.  Once the image
    exists, however, that chosen order becomes part of the evidence binding and
    every later copy remains strict.  Therefore this comparison is deliberately
    shallow: nested ``input_evidence[*].subject_ids`` and every non-generate
    argument are still compared byte-for-byte as decoded JSON values.
    """

    exact = predicted is not None and predicted.get("arguments") == expected.get(
        "arguments"
    )
    expected_is_generate = expected.get("name") == "generate_image"
    if not expected_is_generate:
        return {
            "exact_arguments_match": exact,
            "subject_ids_set_equivalent": None,
            "semantic_arguments_match": exact,
        }

    if predicted is None or predicted.get("name") != "generate_image":
        return {
            "exact_arguments_match": exact,
            "subject_ids_set_equivalent": False,
            "semantic_arguments_match": False,
        }
    expected_arguments = expected.get("arguments")
    predicted_arguments = predicted.get("arguments")
    if not isinstance(expected_arguments, dict) or not isinstance(
        predicted_arguments, dict
    ):
        return {
            "exact_arguments_match": exact,
            "subject_ids_set_equivalent": False,
            "semantic_arguments_match": False,
        }

    expected_subjects = _unique_string_set(expected_arguments.get("subject_ids"))
    predicted_subjects = _unique_string_set(predicted_arguments.get("subject_ids"))
    subject_ids_set_equivalent = (
        expected_subjects is not None
        and predicted_subjects is not None
        and expected_subjects == predicted_subjects
    )
    expected_remainder = {
        key: value for key, value in expected_arguments.items() if key != "subject_ids"
    }
    predicted_remainder = {
        key: value for key, value in predicted_arguments.items() if key != "subject_ids"
    }
    return {
        "exact_arguments_match": exact,
        "subject_ids_set_equivalent": subject_ids_set_equivalent,
        "semantic_arguments_match": (
            subject_ids_set_equivalent and expected_remainder == predicted_remainder
        ),
    }


def model_inputs_to_device(rendered: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    inputs: dict[str, Any] = {}
    sequence_keys = {"input_ids", "attention_mask", "mm_token_type_ids", "position_ids"}
    for key, value in rendered.items():
        if key in {"labels", "loss_weights"}:
            continue
        if isinstance(value, list):
            dtype = torch.long if key in sequence_keys else None
            value = torch.tensor(value, dtype=dtype)
            if key in sequence_keys:
                value = value.unsqueeze(0)
        if torch.is_tensor(value):
            value = value.to(device)
        inputs[key] = value
    return inputs


def main() -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from llamafactory.v1.core.rendering import Renderer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--case", action="append", type=parse_case, required=True, help="ROW_INDEX:MESSAGE_INDEX")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--video-max-pixels-per-frame", type=int, default=65536)
    parser.add_argument("--video-max-frames", type=int, default=12)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")

    rows = load_rows(args.validation_jsonl, {row_index for row_index, _ in args.case})
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    processor.image_processor.size["longest_edge"] = args.image_max_pixels
    processor.video_processor.max_frames = args.video_max_frames
    processor.video_processor.size["longest_edge"] = args.video_max_pixels_per_frame * args.video_max_frames
    renderer = Renderer(processor)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    ).eval()
    device = next(model.parameters()).device
    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = sorted({int(im_end_id), int(processor.tokenizer.eos_token_id)})
    pad_token_id = int(processor.tokenizer.pad_token_id)

    results: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case_number, (row_index, message_index) in enumerate(args.case, start=1):
        row = rows[row_index]
        if message_index >= len(row["messages"]):
            raise IndexError(f"message index {message_index} is out of range for row {row_index}")
        expected = expected_call(row["messages"][message_index])
        prefix = row["messages"][:message_index]
        rendered = renderer.render_messages(
            prefix,
            row["tools"],
            is_generate=True,
            enable_thinking=False,
        )
        model_inputs = model_inputs_to_device(rendered, device)
        prompt_tokens = int(model_inputs["input_ids"].shape[-1])
        media_counts = {
            media_type: sum(
                block.get("type") == media_type
                for message in prefix
                for block in message.get("content", [])
            )
            for media_type in ("image_url", "video_url", "audio_url")
        }

        started = time.monotonic()
        with torch.inference_mode():
            output_ids = model.generate(
                **model_inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=eos_ids,
                pad_token_id=pad_token_id,
            )
        elapsed = time.monotonic() - started
        generated_ids = output_ids[0, prompt_tokens:].detach().cpu()
        raw_output = processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        parsed, format_errors = parse_qwen35_tool_call(raw_output)
        schema_errors = validate_call(parsed, row["tools"])
        match_metrics = argument_match_metrics(expected, parsed)
        result = {
            "case": case_number,
            "row_index": row_index,
            "message_index": message_index,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": int(generated_ids.numel()),
            "elapsed_seconds": round(elapsed, 3),
            "media_counts": media_counts,
            "expected": expected,
            "predicted": parsed,
            "expected_name_match": parsed is not None and parsed["name"] == expected["name"],
            **match_metrics,
            "format_errors": format_errors,
            "schema_errors": schema_errors,
            "wire_format_valid": not format_errors,
            "schema_valid": not schema_errors,
            "ended_on_im_end": bool(generated_ids.numel() and int(generated_ids[-1]) == im_end_id),
            "raw_output": raw_output,
        }
        results.append(result)
        args.output.write_text(
            json.dumps({"model": str(args.model), "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "case": case_number,
                    "row_index": row_index,
                    "message_index": message_index,
                    "expected": expected["name"],
                    "predicted": parsed and parsed["name"],
                    "wire_format_valid": not format_errors,
                    "schema_valid": not schema_errors,
                    "exact_arguments_match": result["exact_arguments_match"],
                    "subject_ids_set_equivalent": result[
                        "subject_ids_set_equivalent"
                    ],
                    "semantic_arguments_match": result[
                        "semantic_arguments_match"
                    ],
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": int(generated_ids.numel()),
                    "elapsed_seconds": round(elapsed, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del model_inputs, output_ids, generated_ids
        torch.cuda.empty_cache()

    summary = {
        "cases": len(results),
        "wire_format_valid": sum(item["wire_format_valid"] for item in results),
        "schema_valid": sum(item["schema_valid"] for item in results),
        "expected_name_match": sum(item["expected_name_match"] for item in results),
        "exact_arguments_match": sum(item["exact_arguments_match"] for item in results),
        "subject_ids_set_cases": sum(
            item["subject_ids_set_equivalent"] is not None for item in results
        ),
        "subject_ids_set_equivalent": sum(
            item["subject_ids_set_equivalent"] is True for item in results
        ),
        "semantic_arguments_match": sum(
            item["semantic_arguments_match"] for item in results
        ),
        "ended_on_im_end": sum(item["ended_on_im_end"] for item in results),
    }
    args.output.write_text(
        json.dumps({"model": str(args.model), "summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
