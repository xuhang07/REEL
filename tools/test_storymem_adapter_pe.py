from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Sequence

import pytest

import storymem_adapter

from storymem_adapter.gemini import (
    GeminiEmptyOutput,
    GeminiInlineImage,
    GeminiOutputTruncated,
    GeminiTextResult,
    IsolatedGeminiTextClient,
)
from storymem_adapter.models import (
    BasePromptRequest,
    OpeningCondition,
    PromptEngineeringProfile,
    PromptOutputContract,
)
from storymem_adapter.models import sha256_bytes, sha256_json, sha256_text
from storymem_adapter.prompt_engineering import (
    AdapterPromptEngineer,
    PromptArtifactConflict,
    PromptArtifactCorrupt,
    PromptOutputRejected,
)
from storymem_adapter.v7_validation import V7PromptValidationError
from storymem_teacher.providers import ProviderHTTPError, ProviderProtocolError


def test_package_root_exposes_only_backend_adapter_boundary() -> None:
    assert storymem_adapter.__all__ == [
        "Echo15AdapterError",
        "Echo15BackendAdapter",
        "Echo15Capabilities",
        "Echo15Compilation",
        "Echo15RequestSpec",
    ]
    for private_name in (
        "AdapterPromptEngineer",
        "BasePromptArtifact",
        "BasePromptRequest",
        "IsolatedGeminiTextClient",
        "PromptEngineeringProfile",
        "WirePromptArtifact",
    ):
        assert not hasattr(storymem_adapter, private_name)


class _FakeGemini:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_text(
        self,
        *,
        system_instruction: str,
        user_text: str,
        images: Sequence[GeminiInlineImage] = (),
    ) -> GeminiTextResult:
        self.calls.append(
            {
                "system_instruction": system_instruction,
                "user_text": user_text,
                "images": tuple(images),
            }
        )
        if not self.responses:
            raise AssertionError("unexpected Gemini call")
        text = self.responses.pop(0)
        request_payload_sha256 = sha256_json(
            {
                "system_instruction": system_instruction,
                "user_text": user_text,
                "images": [image.content_sha256 for image in images],
            }
        )
        return GeminiTextResult(
            text=text,
            request_manifest={
                "request_payload_sha256": request_payload_sha256,
                "system_instruction_sha256": sha256_text(system_instruction),
                "user_turn_count": 1,
                "parts": [
                    {
                        "kind": "text",
                        "sha256": sha256_text(user_text),
                        "size_bytes": len(user_text.encode("utf-8")),
                    },
                    *[
                        {
                            "kind": "inline_image",
                            "mime_type": image.mime_type,
                            "sha256": image.content_sha256,
                            "size_bytes": len(image.data),
                        }
                        for image in images
                    ],
                ],
            },
            provenance={
                "request_payload_sha256": request_payload_sha256,
                "response_sha256": sha256_text("response-body"),
                "assistant_text_sha256": sha256_text(text),
                "raw_assistant_text": text,
                "conversation_reused": False,
                "user_turn_count": 1,
            },
        )


def _profile(name: str = "base") -> PromptEngineeringProfile:
    return PromptEngineeringProfile.from_text(
        profile_id=name,
        system_instruction=f"System contract for {name}.",
        output_contract=PromptOutputContract(max_characters=1999),
    )


def _request(*, raw: str = "Mira opens the observatory door.") -> BasePromptRequest:
    return BasePromptRequest(
        scope_id="story-7/shot-2",
        raw_prompt=raw,
        origin_plan_projection={"plan_id": "plan-1", "version": 1},
        initial_condition_projection=[
            {"binding_id": "binding-1", "effect": "identity_reference"}
        ],
    )


def test_profile_loads_exact_official_sources_and_hashes_them() -> None:
    root = Path(__file__).resolve().parents[1]
    ordinary = (
        root
        / "Director_Agent"
        / "pe"
        / "v7_cinematic_full"
        / "references"
        / "shot-prompt-writer.md"
    )
    tail = (
        root
        / "Director_Agent"
        / "pe"
        / "v7_cinematic_full"
        / "skills"
        / "i2v-tail-frame-prompt-rewriter"
        / "SKILL.md"
    )
    profile = PromptEngineeringProfile.from_sources(
        profile_id="official-v7-tail-frame",
        sources=(("ordinary-shot-v7", ordinary), ("tail-frame", tail)),
        output_contract=PromptOutputContract(max_characters=1999),
        separator="\n\n---\n\n",
    )

    assert profile.system_instruction.startswith(ordinary.read_text(encoding="utf-8"))
    assert profile.system_instruction.endswith(tail.read_text(encoding="utf-8"))
    assert [row["source_id"] for row in profile.source_manifest] == [
        "ordinary-shot-v7",
        "tail-frame",
    ]
    assert profile.manifest()["profile_sha256"] == profile.profile_sha256


def test_base_prompt_is_atomic_immutable_and_resume_calls_no_api(tmp_path: Path) -> None:
    gemini = _FakeGemini(["This video has 2 shots. generated base prompt."])
    engineer = AdapterPromptEngineer(root=tmp_path / "private", gemini=gemini)
    request = _request()

    first = engineer.ensure_base(request, profile=_profile())
    second = engineer.ensure_base(request, profile=_profile())

    assert first.to_dict() == second.to_dict()
    assert first.prompt_text == "This video has 2 shots. generated base prompt."
    assert len(gemini.calls) == 1
    assert gemini.calls[0]["user_text"] == request.raw_prompt
    assert gemini.calls[0]["images"] == ()
    mutable = first.to_dict()
    mutable["prompt"]["text"] = "tampered"
    assert first.prompt_text != "tampered"

    with pytest.raises(PromptArtifactConflict):
        engineer.ensure_base(_request(raw="A different request."), profile=_profile())
    assert len(gemini.calls) == 1


def test_passthrough_locks_exact_raw_prompt_without_api(tmp_path: Path) -> None:
    gemini = _FakeGemini([])
    engineer = AdapterPromptEngineer(root=tmp_path, gemini=gemini)
    request = BasePromptRequest(
        scope_id="story-1/shot-1",
        raw_prompt="  exact user bytes  ",
        origin_plan_projection={"plan_id": "p"},
        initial_condition_projection=[],
        mode="passthrough",
    )

    artifact = engineer.ensure_base(request, profile=None)

    assert artifact.prompt_text == "  exact user bytes  "
    assert gemini.calls == []
    assert artifact.to_dict()["request_manifest"]["provider_call_count"] == 0


def test_ordinary_condition_revisions_never_rewrite_prompt(tmp_path: Path) -> None:
    gemini = _FakeGemini(["base prompt"])
    engineer = AdapterPromptEngineer(root=tmp_path, gemini=gemini)
    base = engineer.ensure_base(_request(), profile=_profile())

    first = engineer.compile_wire(
        base=base,
        condition_projection=[{"binding": "portrait-a"}],
    )
    revised = engineer.compile_wire(
        base=base,
        condition_projection=[{"binding": "portrait-b"}],
    )

    assert first.prompt_text == base.prompt_text
    assert revised.prompt_text == base.prompt_text
    assert first.prompt_sha256 == base.prompt_sha256
    assert revised.prompt_sha256 == base.prompt_sha256
    assert first.artifact_id != revised.artifact_id
    assert first.derivation == revised.derivation == "identity"
    assert len(gemini.calls) == 1


def test_opening_rewrite_uses_image_and_cache_ignores_plan_revision(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nactual-frame-zero")
    gemini = _FakeGemini(
        [
            "base prompt",
            "Use the current image as the first frame of the video, then continue.",
        ]
    )
    engineer = AdapterPromptEngineer(root=tmp_path / "private", gemini=gemini)
    base = engineer.ensure_base(_request(), profile=_profile())
    opening_profile = PromptEngineeringProfile.from_text(
        profile_id="opening",
        system_instruction="Opening rewrite contract.",
        output_contract=PromptOutputContract(
            max_characters=1999,
            required_any_prefixes=(
                "Use the current image as the first frame of the video,",
            ),
        ),
    )
    opening = OpeningCondition(
        condition_id="condition-frame-zero",
        image_path=image,
        semantic_effect="exact_opening_state",
    )

    first = engineer.compile_wire(
        base=base,
        condition_projection={"plan": "v1", "opening": "condition-frame-zero"},
        opening=opening,
        opening_profile=opening_profile,
    )
    resumed = engineer.compile_wire(
        base=base,
        condition_projection={"plan": "v1", "opening": "condition-frame-zero"},
        opening=opening,
        opening_profile=opening_profile,
    )
    revised_plan = engineer.compile_wire(
        base=base,
        condition_projection={"plan": "v2", "opening": "condition-frame-zero"},
        opening=opening,
        opening_profile=opening_profile,
    )

    assert first.to_dict() == resumed.to_dict()
    assert revised_plan.artifact_id != first.artifact_id
    assert revised_plan.prompt_text == first.prompt_text
    assert first.derivation == "opening_image_rewrite"
    assert base.prompt_text == "base prompt"
    assert len(gemini.calls) == 2  # one base PE, one cached opening rewrite
    rewrite_call = gemini.calls[1]
    assert rewrite_call["images"][0].data == image.read_bytes()
    assert base.prompt_text in rewrite_call["user_text"]
    first_doc = first.to_dict()
    assert first_doc["lineage"]["opening_condition"]["sha256"] == sha256_bytes(
        image.read_bytes()
    )
    assert first_doc["rewrite_cache_key"] == revised_plan.to_dict()[
        "rewrite_cache_key"
    ]


def test_changed_opening_bytes_is_the_only_condition_change_that_calls_pe(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"first")
    gemini = _FakeGemini(["base", "wire one", "wire two"])
    engineer = AdapterPromptEngineer(root=tmp_path / "private", gemini=gemini)
    base = engineer.ensure_base(_request(), profile=_profile())
    profile = _profile("opening")
    opening = OpeningCondition(
        condition_id="opening",
        image_path=image,
        semantic_effect="temporal_continuation",
        mime_type="image/png",
    )
    first = engineer.compile_wire(
        base=base,
        condition_projection={"revision": 1},
        opening=opening,
        opening_profile=profile,
    )
    image.write_bytes(b"second")
    second = engineer.compile_wire(
        base=base,
        condition_projection={"revision": 2},
        opening=opening,
        opening_profile=profile,
    )

    assert first.prompt_text == "wire one"
    assert second.prompt_text == "wire two"
    assert first.to_dict()["rewrite_cache_key"] != second.to_dict()[
        "rewrite_cache_key"
    ]
    assert len(gemini.calls) == 3


def test_opening_cache_is_partitioned_by_full_validation_identity(
    tmp_path: Path,
) -> None:
    class SemanticValidator:
        def __init__(self, validation_id: str, semantic_contract: dict[str, Any]):
            self.validation_id = validation_id
            self.semantic_contract = semantic_contract

        def validate(
            self,
            text: str,
            *,
            raw_prompt: str,
            condition_effect: str | None,
            expected_caption_language: str | None,
        ) -> dict[str, Any]:
            if "VALID" not in text:
                raise V7PromptValidationError("missing VALID marker")
            # Deliberately omit the contract from the result.  Cache identity
            # must not depend on validators exposing it in their receipts.
            return {"validation_id": self.validation_id}

    image = tmp_path / "frame.png"
    image.write_bytes(b"same-opening-image")
    request = BasePromptRequest(
        scope_id="story-cache/shot-1",
        raw_prompt="Mira opens the observatory door.",
        origin_plan_projection={"plan_id": "cache-plan"},
        initial_condition_projection=[],
        mode="passthrough",
        expected_caption_language="English",
    )
    opening = OpeningCondition(
        condition_id="opening",
        image_path=image,
        semantic_effect="exact_opening_state",
        mime_type="image/png",
    )
    opening_profile = _profile("opening-cache")
    validators = (
        SemanticValidator("validator.same.v1", {"variant": "a"}),
        SemanticValidator("validator.different.v1", {"variant": "a"}),
        SemanticValidator("validator.same.v1", {"variant": "b"}),
    )
    wires = []
    clients = []
    base = None
    for validator in validators:
        client = _FakeGemini(["VALID opening rewrite"])
        engineer = AdapterPromptEngineer(
            root=tmp_path / "private",
            gemini=client,
            output_validator=validator,
        )
        if base is None:
            base = engineer.ensure_base(request, profile=None)
        wires.append(
            engineer.compile_wire(
                base=base,
                condition_projection={"opening": "same"},
                opening=opening,
                opening_profile=opening_profile,
            )
        )
        clients.append(client)

    cache_keys = [wire.to_dict()["rewrite_cache_key"] for wire in wires]
    assert len(set(cache_keys)) == 3
    assert [len(client.calls) for client in clients] == [1, 1, 1]
    receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "private" / "opening_rewrite_cache").glob("*.json")
    ]
    assert len(receipts) == 3
    identities = [receipt["validation_identity"] for receipt in receipts]
    assert {identity["output_validator_id"] for identity in identities} == {
        "validator.same.v1",
        "validator.different.v1",
    }
    assert {identity["semantic_contract_sha256"] for identity in identities} == {
        sha256_json({"variant": "a"}),
        sha256_json({"variant": "b"}),
    }
    for receipt in receipts:
        identity = receipt["validation_identity"]
        assert identity["raw_prompt_sha256"] == sha256_text(request.raw_prompt)
        assert identity["expected_caption_language"] == "English"
        assert identity["condition_effect"] == "exact_opening_state"
        assert receipt["validation_identity_sha256"] == sha256_json(identity)


def test_invalid_output_is_not_committed_and_corruption_is_not_overwritten(
    tmp_path: Path,
) -> None:
    gemini = _FakeGemini(["```wrapped\ntext```", "valid output"])
    engineer = AdapterPromptEngineer(
        root=tmp_path, gemini=gemini, max_generation_attempts=1
    )
    request = _request()

    with pytest.raises(PromptOutputRejected):
        engineer.ensure_base(request, profile=_profile())
    artifact = engineer.ensure_base(request, profile=_profile())
    path = next((tmp_path / "scopes").glob("*/base_prompt.json"))
    document = json.loads(path.read_text(encoding="utf-8"))
    document["prompt"]["text"] = "corrupt"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PromptArtifactCorrupt):
        engineer.ensure_base(request, profile=_profile())
    assert artifact.prompt_text == "valid output"
    assert len(gemini.calls) == 2


def test_provider_whitespace_is_canonicalized_but_raw_response_is_retained(
    tmp_path: Path,
) -> None:
    raw = "  This video has 2 shots.\nshot1: A.\tshot2: B.  \n"
    gemini = _FakeGemini([raw])
    engineer = AdapterPromptEngineer(root=tmp_path, gemini=gemini)

    artifact = engineer.ensure_base(_request(), profile=_profile())
    document = artifact.to_dict()

    assert artifact.prompt_text == "This video has 2 shots. shot1: A. shot2: B."
    assert document["provider_provenance"]["raw_assistant_text"] == raw
    assert document["provider_provenance"]["assistant_text_sha256"] == sha256_text(
        raw
    )
    assert document["output_normalization"] == {
        "method": "unicode_whitespace_collapse.v1",
        "changed": True,
        "raw_text_sha256": sha256_text(raw),
        "normalized_text_sha256": sha256_text(artifact.prompt_text),
    }


def test_profile_hard_character_limit_is_checked_after_normalization(
    tmp_path: Path,
) -> None:
    gemini = _FakeGemini(["x" * 2000 + "\n"])
    engineer = AdapterPromptEngineer(
        root=tmp_path, gemini=gemini, max_generation_attempts=1
    )

    with pytest.raises(PromptOutputRejected, match="hard limit"):
        engineer.ensure_base(_request(), profile=_profile())

    assert not list((tmp_path / "scopes").glob("*/base_prompt.json"))


class _MarkerValidator:
    validation_id = "test.marker-validator.v1"

    def validate(
        self,
        text: str,
        *,
        raw_prompt: str,
        condition_effect: str | None,
        expected_caption_language: str | None,
    ) -> dict[str, Any]:
        if "VALID" not in text:
            raise V7PromptValidationError("missing VALID marker")
        return {
            "validation_id": self.validation_id,
            "raw_prompt_sha256": sha256_text(raw_prompt),
            "condition_effect": condition_effect,
            "expected_caption_language": expected_caption_language,
        }


def test_bounded_correction_uses_fresh_turn_and_records_every_attempt(
    tmp_path: Path,
) -> None:
    gemini = _FakeGemini(["bad draft", "VALID corrected prompt"])
    engineer = AdapterPromptEngineer(
        root=tmp_path,
        gemini=gemini,
        output_validator=_MarkerValidator(),
        max_generation_attempts=2,
    )

    artifact = engineer.ensure_base(_request(), profile=_profile())

    assert artifact.prompt_text == "VALID corrected prompt"
    assert len(gemini.calls) == 2
    assert gemini.calls[0]["user_text"] == _request().raw_prompt
    assert "missing VALID marker" in gemini.calls[1]["user_text"]
    assert "bad draft" in gemini.calls[1]["user_text"]
    assert _request().raw_prompt in gemini.calls[1]["user_text"]
    provenance = artifact.to_dict()["provider_provenance"]
    assert provenance["attempt_count"] == 2
    assert [row["validation"]["accepted"] for row in provenance["attempts"]] == [
        False,
        True,
    ]

    resumed = engineer.ensure_base(_request(), profile=_profile())
    assert resumed.to_dict() == artifact.to_dict()
    assert len(gemini.calls) == 2


def test_correction_retains_all_prior_validation_failures(tmp_path: Path) -> None:
    class CumulativeValidator:
        validation_id = "test.cumulative-validator.v1"

        def validate(
            self,
            text: str,
            *,
            raw_prompt: str,
            condition_effect: str | None,
            expected_caption_language: str | None,
        ) -> dict[str, Any]:
            if text == "first rejected draft":
                raise V7PromptValidationError("missing camera movement")
            if text == "second rejected draft":
                raise V7PromptValidationError("missing identity anchor")
            return {"validation_id": self.validation_id}

    gemini = _FakeGemini(
        ["first rejected draft", "second rejected draft", "accepted prompt"]
    )
    engineer = AdapterPromptEngineer(
        root=tmp_path,
        gemini=gemini,
        output_validator=CumulativeValidator(),
        max_generation_attempts=3,
    )

    engineer.ensure_base(_request(), profile=_profile())

    second_turn = gemini.calls[1]["user_text"]
    third_turn = gemini.calls[2]["user_text"]
    assert "1. missing camera movement" in second_turn
    assert "missing identity anchor" not in second_turn
    assert "1. missing camera movement" in third_turn
    assert "2. missing identity anchor" in third_turn
    assert "1550-1750 characters inclusive" in second_turn
    assert "1550-1750 characters inclusive" in third_turn


def test_structured_correction_sequence_resolves_voice_prompt_conflict(
    tmp_path: Path,
) -> None:
    class SequencedValidator:
        validation_id = "test.structured-sequence-validator.v1"

        def validate(
            self,
            text: str,
            *,
            raw_prompt: str,
            condition_effect: str | None,
            expected_caption_language: str | None,
        ) -> dict[str, Any]:
            if text == "event-missing":
                raise V7PromptValidationError(
                    "raw-intent semantic preservation failed: event: missing discover, inspect",
                    violations=[
                        {
                            "code": "semantic.missing_concepts",
                            "category": "event",
                            "missing_concepts": ["discover", "inspect"],
                        }
                    ],
                )
            if text == "dialogue-missing":
                raise V7PromptValidationError(
                    "dialogue and declaration are missing",
                    violations=[
                        {
                            "code": "dialogue.requested_speech_missing",
                            "mode": "required_unspecified",
                            "expected_count": 1,
                            "actual_count": 0,
                        },
                        {"code": "dialogue.vocal_declaration"},
                    ],
                )
            return {"validation_id": self.validation_id}

        def correction_context(
            self,
            *,
            raw_prompt: str,
            condition_effect: str | None,
            expected_caption_language: str | None,
        ) -> dict[str, Any]:
            return {
                "dialogue_mode": "required_unspecified",
                "required_character_range": [1, 1999],
                "required_output_checklist": {
                    "dialogue": "compose_exactly_one_natural_line"
                },
            }

    raw_prompt = (
        "The chef briefly speaks; do not invent or display exact dialogue or infer "
        "acoustic content. FINAL REEL DIALOGUE RESOLUTION: "
        "mode=required_unspecified."
    )
    gemini = _FakeGemini(
        ["x" * 2001, "event-missing", "dialogue-missing", "accepted prompt"]
    )
    engineer = AdapterPromptEngineer(
        root=tmp_path,
        gemini=gemini,
        output_validator=SequencedValidator(),
        max_generation_attempts=4,
    )

    artifact = engineer.ensure_base(_request(raw=raw_prompt), profile=_profile())

    assert artifact.prompt_text == "accepted prompt"
    assert len(gemini.calls) == 4
    assert '"code":"length.above_maximum"' in gemini.calls[1]["user_text"]
    assert '"missing_concepts":["discover","inspect"]' in gemini.calls[2][
        "user_text"
    ]
    assert '"code":"dialogue.requested_speech_missing"' in gemini.calls[3][
        "user_text"
    ]
    assert '"code":"dialogue.vocal_declaration"' in gemini.calls[3]["user_text"]
    for call in gemini.calls[1:]:
        correction = call["user_text"]
        assert correction.rindex("FINAL CORRECTION AUTHORITY") > correction.rindex(
            "do not invent or display exact dialogue"
        )
        assert "it does not forbid speech" in correction
    ledger = artifact.to_dict()["provider_provenance"]["pe_attempt_ledger"]
    assert [row["outcome"] for row in ledger] == [
        "deterministic_validation_rejected",
        "deterministic_validation_rejected",
        "deterministic_validation_rejected",
        "accepted",
    ]
    assert ledger[1]["validation"]["violation_codes"] == [
        "semantic.missing_concepts"
    ]


def test_unexpected_validator_error_propagates_without_correction(
    tmp_path: Path,
) -> None:
    class BrokenValidator:
        validation_id = "test.broken-validator.v1"

        def validate(
            self,
            text: str,
            *,
            raw_prompt: str,
            condition_effect: str | None,
            expected_caption_language: str | None,
        ) -> dict[str, Any]:
            raise RuntimeError("validator implementation defect")

    gemini = _FakeGemini(["otherwise valid output", "must not be requested"])
    engineer = AdapterPromptEngineer(
        root=tmp_path,
        gemini=gemini,
        output_validator=BrokenValidator(),
        max_generation_attempts=2,
    )

    with pytest.raises(RuntimeError, match="validator implementation defect"):
        engineer.ensure_base(_request(), profile=_profile())

    assert len(gemini.calls) == 1
    assert not list((tmp_path / "scopes").glob("*/base_prompt.json"))


def test_overlength_correction_records_actual_length_and_safe_target_without_truncating(
    tmp_path: Path,
) -> None:
    rejected = "x" * 2050
    accepted = "y" * 1600
    gemini = _FakeGemini([rejected, accepted])
    engineer = AdapterPromptEngineer(
        root=tmp_path,
        gemini=gemini,
        max_generation_attempts=2,
    )
    profile = PromptEngineeringProfile.from_text(
        profile_id="bounded-length",
        system_instruction="Return a complete cinematic prompt.",
        output_contract=PromptOutputContract(
            min_characters=1500,
            max_characters=1999,
        ),
    )

    artifact = engineer.ensure_base(_request(), profile=profile)

    assert artifact.prompt_text == accepted
    correction = gemini.calls[1]["user_text"]
    assert "rejected draft is 2050 Unicode characters" in correction
    assert "1550-1750 characters inclusive" in correction
    assert "absolute permitted range is 1500-1999" in correction
    assert "Never slice, mechanically truncate" in correction
    assert f"REJECTED DRAFT:\n{rejected}" in correction


def test_correction_target_adapts_to_a_smaller_valid_profile(tmp_path: Path) -> None:
    gemini = _FakeGemini(["bad", "VALID"])
    engineer = AdapterPromptEngineer(
        root=tmp_path,
        gemini=gemini,
        output_validator=_MarkerValidator(),
        max_generation_attempts=2,
    )
    profile = PromptEngineeringProfile.from_text(
        profile_id="small-profile",
        system_instruction="Return a compact marker.",
        output_contract=PromptOutputContract(max_characters=10),
    )

    artifact = engineer.ensure_base(_request(), profile=profile)

    assert artifact.prompt_text == "VALID"
    correction = gemini.calls[1]["user_text"]
    assert "Rewrite the complete prompt to 1-10 characters inclusive" in correction
    assert "absolute permitted range is 1-10" in correction


def test_correction_construction_failure_keeps_the_attempt_ledger(
    tmp_path: Path,
) -> None:
    class BrokenCorrectionValidator(_MarkerValidator):
        validation_id = "test.broken-correction-validator.v1"

        def correction_context(
            self,
            *,
            raw_prompt: str,
            condition_effect: str | None,
            expected_caption_language: str | None,
        ) -> dict[str, Any]:
            raise RuntimeError("correction context defect")

    gemini = _FakeGemini(["bad draft"])
    engineer = AdapterPromptEngineer(
        root=tmp_path,
        gemini=gemini,
        output_validator=BrokenCorrectionValidator(),
        max_generation_attempts=2,
    )

    with pytest.raises(RuntimeError, match="correction context defect") as captured:
        engineer.ensure_base(_request(), profile=_profile())

    assert len(captured.value.attempts) == 1
    assert captured.value.attempts[0]["outcome"] == (
        "deterministic_validation_rejected"
    )
    assert captured.value.pe_generation_policy["passthrough_fallback"] is False


class _Response:
    status_code = 200
    headers = {"x-request-id": "request-123"}
    history: list[Any] = []
    url = "https://gemini.example.test/generateContent"

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body
        self.content = json.dumps(body).encode("utf-8")
        self.text = self.content.decode("utf-8")

    def json(self) -> dict[str, Any]:
        return copy.deepcopy(self._body)


class _Session:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        self.requests.append({"method": method, "url": url, **kwargs})
        return _Response(
            {
                "modelVersion": "gemini-test",
                "usageMetadata": {"promptTokenCount": 5},
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": "one plain prompt"}]},
                    }
                ],
            }
        )


def test_gemini_transport_is_one_fresh_user_turn_and_text_only() -> None:
    session = _Session()
    client = IsolatedGeminiTextClient(
        "https://gemini.example.test/generateContent",
        api_key="secret",
        model="gemini-test",
        max_attempts=1,
        session_factory=lambda: session,
    )
    image = GeminiInlineImage.from_bytes(mime_type="image/png", data=b"pixels")

    result = client.generate_text(
        system_instruction="private PE system",
        user_text="raw shot only",
        images=(image,),
    )

    assert result.text == "one plain prompt"
    assert len(session.requests) == 1
    payload = session.requests[0]["json"]
    assert payload["contents"] == [
        {
            "role": "user",
            "parts": [
                {"text": "raw shot only"},
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": "cGl4ZWxz",
                    }
                },
            ],
        }
    ]
    assert payload["generationConfig"]["responseMimeType"] == "text/plain"
    assert "tools" not in payload
    assert "messages" not in payload
    assert result.provenance["conversation_reused"] is False
    assert result.request_manifest["user_turn_count"] == 1
    assert result.request_manifest["transport"] == "gemini_native_generate_content"
    assert "secret" not in json.dumps(result.request_manifest)


def test_gemini_transport_excludes_thought_parts() -> None:
    class ThoughtSession(_Session):
        def request(self, method: str, url: str, **kwargs: Any) -> _Response:
            self.requests.append({"method": method, "url": url, **kwargs})
            return _Response(
                {
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {
                                "parts": [
                                    {"thought": True, "text": "hidden\nreasoning"},
                                    {"text": "final prompt only"},
                                ]
                            },
                        }
                    ]
                }
            )

    session = ThoughtSession()
    client = IsolatedGeminiTextClient(
        "https://gemini.example.test/generateContent",
        api_key=None,
        model="gemini-test",
        max_attempts=1,
        session_factory=lambda: session,
    )

    result = client.generate_text(system_instruction="system", user_text="shot")

    assert result.text == "final prompt only"
    response = result.provenance["response_manifest"]
    assert response["thought_part_count"] == 1
    assert response["final_text_part_count"] == 1
    assert "hidden" not in json.dumps(result.request_manifest)
    assert "hidden" not in result.provenance["raw_assistant_text"]


def test_gemini_transport_rejects_truncated_candidate() -> None:
    class TruncatedSession(_Session):
        def request(self, method: str, url: str, **kwargs: Any) -> _Response:
            return _Response(
                {
                    "candidates": [
                        {
                            "finishReason": "MAX_TOKENS",
                            "content": {"parts": [{"text": "partial output"}]},
                        }
                    ]
                }
            )

    client = IsolatedGeminiTextClient(
        "https://gemini.example.test/generateContent",
        api_key=None,
        model="gemini-test",
        max_attempts=1,
        session_factory=TruncatedSession,
    )

    with pytest.raises(GeminiOutputTruncated, match="did not finish cleanly") as captured:
        client.generate_text(system_instruction="system", user_text="shot")
    assert captured.value.provenance["response_manifest"]["finish_reason"] == (
        "MAX_TOKENS"
    )
    assert captured.value.provenance["request_manifest"]["parts"][0][
        "sha256"
    ] == sha256_text("shot")


def test_gemini_transport_marks_blank_stop_as_retryable_and_preserves_raw_text(
) -> None:
    raw_text = " \n\t "
    session = _SequenceSession(
        [_gemini_body(finish_reason="STOP", text=raw_text)]
    )
    client = IsolatedGeminiTextClient(
        "https://gemini.example.test/generateContent",
        api_key=None,
        model="gemini-test",
        max_attempts=1,
        session_factory=lambda: session,
    )

    with pytest.raises(GeminiEmptyOutput, match="no non-thought") as captured:
        client.generate_text(system_instruction="system", user_text="shot")

    provenance = captured.value.provenance
    assert provenance["failure_kind"] == "gemini_empty_output"
    assert provenance["retryable_for_fresh_pe_attempt"] is True
    assert provenance["response_manifest"]["finish_reason"] == "STOP"
    assert provenance["raw_assistant_text"] == raw_text
    assert provenance["assistant_text_sha256"] == sha256_text(raw_text)


class _SequenceSession:
    def __init__(self, bodies: Sequence[dict[str, Any]]) -> None:
        self.bodies = list(bodies)
        self.requests: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        self.requests.append({"method": method, "url": url, **kwargs})
        if not self.bodies:
            raise AssertionError("unexpected Gemini request")
        return _Response(self.bodies.pop(0))


def _gemini_body(*, finish_reason: str, text: str) -> dict[str, Any]:
    return {
        "modelVersion": "gemini-test",
        "candidates": [
            {
                "finishReason": finish_reason,
                "content": {"parts": [{"text": text}]},
            }
        ],
    }


def test_truncation_is_retried_as_fresh_pe_turn_and_persisted_in_ledger(
    tmp_path: Path,
) -> None:
    session = _SequenceSession(
        [
            _gemini_body(finish_reason="MAX_TOKENS", text="discarded partial"),
            _gemini_body(finish_reason="STOP", text="complete prompt"),
        ]
    )
    client = IsolatedGeminiTextClient(
        "https://gemini.example.test/generateContent",
        api_key=None,
        model="gemini-test",
        max_attempts=1,
        session_factory=lambda: session,
    )
    engineer = AdapterPromptEngineer(
        root=tmp_path,
        gemini=client,
        max_generation_attempts=2,
    )

    artifact = engineer.ensure_base(_request(), profile=_profile())

    assert artifact.prompt_text == "complete prompt"
    assert len(session.requests) == 2
    assert session.requests[0]["json"]["contents"] == session.requests[1]["json"][
        "contents"
    ]
    ledger = artifact.to_dict()["provider_provenance"]["pe_attempt_ledger"]
    assert [row["outcome"] for row in ledger] == [
        "retryable_provider_truncation",
        "accepted",
    ]
    first = ledger[0]
    assert first["failure"]["finish_reason"] == "MAX_TOKENS"
    assert first["failure"]["retryable_as_fresh_pe_attempt"] is True
    assert first["provider_provenance"]["response_sha256"].startswith("sha256:")
    assert first["provider_provenance"]["request_manifest_sha256"] == sha256_json(
        first["request_manifest"]
    )
    assert first["request_manifest"]["request_payload_sha256"] == first[
        "provider_provenance"
    ]["request_payload_sha256"]


def test_blank_stop_is_retried_as_fresh_pe_turn_and_persisted_in_ledger(
    tmp_path: Path,
) -> None:
    raw_blank = " \n\t "
    session = _SequenceSession(
        [
            _gemini_body(finish_reason="STOP", text=raw_blank),
            _gemini_body(finish_reason="STOP", text="complete prompt"),
        ]
    )
    client = IsolatedGeminiTextClient(
        "https://gemini.example.test/generateContent",
        api_key=None,
        model="gemini-test",
        max_attempts=1,
        session_factory=lambda: session,
    )
    engineer = AdapterPromptEngineer(
        root=tmp_path,
        gemini=client,
        max_generation_attempts=2,
    )

    artifact = engineer.ensure_base(_request(), profile=_profile())

    assert artifact.prompt_text == "complete prompt"
    assert len(session.requests) == 2
    assert session.requests[0]["json"]["contents"] == session.requests[1]["json"][
        "contents"
    ]
    ledger = artifact.to_dict()["provider_provenance"]["pe_attempt_ledger"]
    assert [row["outcome"] for row in ledger] == [
        "retryable_provider_empty_output",
        "accepted",
    ]
    first = ledger[0]
    assert first["failure"]["finish_reason"] == "STOP"
    assert first["failure"]["failure_kind"] == "gemini_empty_output"
    assert first["failure"]["retryable_as_fresh_pe_attempt"] is True
    assert first["provider_provenance"]["raw_assistant_text"] == raw_blank
    assert first["provider_provenance"]["assistant_text_sha256"] == sha256_text(
        raw_blank
    )
    assert artifact.to_dict()["request_manifest"]["generation_policy"][
        "retryable_failure_kinds"
    ] == ["gemini_output_truncated", "gemini_empty_output"]


def test_truncation_exhaustion_is_bounded_and_exposes_complete_ledger(
    tmp_path: Path,
) -> None:
    session = _SequenceSession(
        [
            _gemini_body(finish_reason="MAX_TOKENS", text="partial one"),
            _gemini_body(finish_reason="MAX_TOKENS", text="partial two"),
        ]
    )
    client = IsolatedGeminiTextClient(
        "https://gemini.example.test/generateContent",
        api_key=None,
        model="gemini-test",
        max_attempts=1,
        session_factory=lambda: session,
    )
    engineer = AdapterPromptEngineer(
        root=tmp_path,
        gemini=client,
        max_generation_attempts=2,
    )

    with pytest.raises(GeminiOutputTruncated) as captured:
        engineer.ensure_base(_request(), profile=_profile())

    assert len(session.requests) == 2
    assert len(captured.value.attempts) == 2
    assert all(
        row["outcome"] == "retryable_provider_truncation"
        for row in captured.value.attempts
    )
    assert captured.value.provenance["pe_attempt_count"] == 2
    assert captured.value.provenance["pe_attempt_ledger_sha256"] == sha256_json(
        captured.value.attempts
    )
    assert not list((tmp_path / "scopes").glob("*/base_prompt.json"))


def test_blank_stop_exhaustion_is_bounded_and_exposes_complete_ledger(
    tmp_path: Path,
) -> None:
    session = _SequenceSession(
        [
            _gemini_body(finish_reason="STOP", text=""),
            _gemini_body(finish_reason="STOP", text=" \n"),
        ]
    )
    client = IsolatedGeminiTextClient(
        "https://gemini.example.test/generateContent",
        api_key=None,
        model="gemini-test",
        max_attempts=1,
        session_factory=lambda: session,
    )
    engineer = AdapterPromptEngineer(
        root=tmp_path,
        gemini=client,
        max_generation_attempts=2,
    )

    with pytest.raises(GeminiEmptyOutput) as captured:
        engineer.ensure_base(_request(), profile=_profile())

    assert len(session.requests) == 2
    assert len(captured.value.attempts) == 2
    assert all(
        row["outcome"] == "retryable_provider_empty_output"
        for row in captured.value.attempts
    )
    assert captured.value.provenance["pe_attempt_count"] == 2
    assert captured.value.provenance["pe_attempt_ledger_sha256"] == sha256_json(
        captured.value.attempts
    )
    assert not list((tmp_path / "scopes").glob("*/base_prompt.json"))


def test_nonretryable_gemini_protocol_and_http_errors_fail_closed(
    tmp_path: Path,
) -> None:
    protocol_session = _SequenceSession(
        [_gemini_body(finish_reason="SAFETY", text="blocked")]
    )
    protocol_client = IsolatedGeminiTextClient(
        "https://gemini.example.test/generateContent",
        api_key=None,
        model="gemini-test",
        max_attempts=1,
        session_factory=lambda: protocol_session,
    )
    protocol_engineer = AdapterPromptEngineer(
        root=tmp_path / "protocol",
        gemini=protocol_client,
        max_generation_attempts=5,
    )

    with pytest.raises(ProviderProtocolError) as protocol_error:
        protocol_engineer.ensure_base(_request(), profile=_profile())
    assert not isinstance(protocol_error.value, GeminiOutputTruncated)
    assert not isinstance(protocol_error.value, GeminiEmptyOutput)
    assert len(protocol_session.requests) == 1
    assert protocol_error.value.attempts[0]["outcome"] == (
        "terminal_provider_failure"
    )
    assert protocol_error.value.attempts[0]["failure"][
        "retryable_as_fresh_pe_attempt"
    ] is False

    class HTTPErrorSession:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        def request(self, method: str, url: str, **kwargs: Any) -> _Response:
            self.requests.append({"method": method, "url": url, **kwargs})
            response = _Response({"error": "invalid request"})
            response.status_code = 400
            return response

    http_session = HTTPErrorSession()
    http_client = IsolatedGeminiTextClient(
        "https://gemini.example.test/generateContent",
        api_key=None,
        model="gemini-test",
        max_attempts=3,
        session_factory=lambda: http_session,
    )
    http_engineer = AdapterPromptEngineer(
        root=tmp_path / "http",
        gemini=http_client,
        max_generation_attempts=5,
    )

    with pytest.raises(ProviderHTTPError) as http_error:
        http_engineer.ensure_base(
            BasePromptRequest(
                scope_id="story-http/shot-1",
                raw_prompt="A different request.",
                origin_plan_projection={"plan_id": "plan-http"},
                initial_condition_projection=[],
            ),
            profile=_profile(),
        )
    assert len(http_session.requests) == 1
    assert len(http_error.value.attempts) == 1
    assert http_error.value.attempts[0]["outcome"] == "terminal_provider_failure"


def test_gemini_from_env_uses_sealed_responses_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "STORYMEM_PE_URL",
        "RECAPTION_URL",
        "RECAPTION_OPENAI_BASE_URL",
        "STORYMEM_PE_API_KEY",
        "RECAPTION_API_KEY",
        "API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    session = _Session()

    client = IsolatedGeminiTextClient.from_env(
        session_factory=lambda: session,
        max_attempts=1,
    )
    client.generate_text(system_instruction="system", user_text="shot")

    assert session.requests[0]["url"] == (
        "https://modelservice.jdcloud.com/v1/responses"
    )
