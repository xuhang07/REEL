from __future__ import annotations

import copy
import io
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from PIL import Image

from reel_h3_adapter import (
    H3PEError,
    H3R2AVPromptAdapter,
    HEADINGS,
    SKILL_PATH,
    load_h3_profile,
    normalize_h3_prompt,
    validate_h3_prompt,
    validate_observation,
)
from storymem_adapter.echo15 import Echo15AdapterError
from storymem_adapter.gemini import GeminiTextResult
from storymem_adapter.models import sha256_bytes, sha256_json, sha256_text


RAW_PROMPT = "Mira crosses the observatory carrying a brass astrolabe. No speech or music."


def make_state(tmp_path, effects=("identity_reference",), *, source_kind="uploaded_material", purpose=None):
    tables = {
        "uploaded_material": "uploaded_evidence",
        "retrieved_memory": "memory_bank",
        "sequence_context": "previous_evidence",
        "generated_evidence": "created_evidence",
        "generation_derived": "generation_evidence",
    }
    data = {
        "request_id": "request_mira",
        "shot_requirement": RAW_PROMPT,
        "shot_requirement_sha256": sha256_text(RAW_PROMPT),
        "registry": {"items": {"person_mira": {"name": "Mira"}}},
        "views": {},
        **{table: {} for table in tables.values()},
    }
    bindings = []
    requirements = []
    for index, effect in enumerate(effects, 1):
        image_path = tmp_path / f"source-{index}.png"
        buffer = io.BytesIO()
        Image.new("RGB", (32, 24), color=(index * 17 % 255, 40, 180)).save(buffer, format="PNG")
        payload = buffer.getvalue()
        image_path.write_bytes(payload)
        evidence_id = f"evidence_{index}"
        representation_id = f"representation_{index}"
        authorization_id = f"authorization_{index}"
        representation = {
            "representation_id": representation_id,
            "modality": "image",
            "mime_type": "image/png",
            "path": str(image_path),
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
        data[tables[source_kind]][evidence_id] = {
            "evidence_id": evidence_id,
            "source_kind": source_kind,
            "modality": "image",
            "subject_ids": ["person_mira"],
            "representations": {representation_id: representation},
            "purpose": purpose,
        }
        data["views"]["|".join((source_kind, evidence_id, representation_id, authorization_id))] = {}
        timeline = "opening" if effect in {"exact_opening_state", "temporal_continuation"} else "ending" if effect == "exact_ending_state" else "middle" if effect == "composition_anchor" else "timeless"
        bindings.append({
            "binding_id": f"binding_{index}",
            "requirement_ids": [f"requirement_{index}"],
            "source_kind": source_kind,
            "evidence_id": evidence_id,
            "representation_id": representation_id,
            "authorization_id": authorization_id,
            "modality": "image",
            "semantic_roles": ["identity"],
            "subject_ids": ["person_mira"],
            "intended_effect": effect,
            "timeline_role": timeline,
            "rank": index,
            "priority": 90,
            "required": True,
            "fallback": "forbidden",
        })
        requirements.append({
            "requirement_id": f"requirement_{index}",
            "description": "Keep person_mira recognizable during the requested action.",
            "subject_ids": ["person_mira"],
            "semantic_roles": ["identity"],
            "acceptable_modalities": ["image"],
            "priority": 90,
            "required": True,
        })
    data["accepted_evidence_plan"] = {
        "plan_id": "plan_mira_1",
        "request_id": data["request_id"],
        "raw_request_sha256": data["shot_requirement_sha256"],
        "requirements": requirements,
        "ranked_bindings": bindings,
        "generation_prompt": RAW_PROMPT,
    }
    seal_plan(data)
    return data


def seal_plan(data):
    plan = data["accepted_evidence_plan"]
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = sha256_json(plan)


def observation_for(pack):
    return {
        "status": "ready",
        "references": [
            {"label": reference["label"], "role": reference["role"],
             "visible_facts": ["A blue-toned rectangular image is visible."],
             "uncertainties": ["Motion and sound are not established by a still image."]}
            for reference in pack["references"]
        ],
        "request_mapping": [{"constraint": "Mira crosses the observatory.",
                             "reference_labels": [], "basis": "request_only"}],
        "conflicts": [],
        "blocked_reasons": [],
    }


def prompt_for(pack):
    definitions = [f"{subject['label']}: {subject['description']}." for subject in pack["subjects"]]
    definitions.extend(reference["definition"] for reference in pack["references"])
    summary = "[reference generation]"
    if any(reference["role"] != "reference" for reference in pack["references"]):
        summary += " + [keyframe completion]"
    summary += " Mira crosses the observatory, retaining the selected reference properties."
    sections = (
        "\n".join(definitions),
        summary,
        "\n".join(f"{reference['label']}: partially_preserved — retain the selected properties in their intended phase." for reference in pack["references"]),
        "At the beginning, Mira carries the brass astrolabe. Mira crosses the observatory naturally, retaining the selected appearance.",
        "N/A — no additional sound specified",
        "N/A — no non-diegetic music requested",
    )
    return "\n\n".join(f"{heading}:\n{body}" for heading, body in zip(HEADINGS, sections))


class FakeGemini:
    def __init__(self, transform=None):
        self.calls = []
        self.transform = transform

    def generate_text(self, **kwargs):
        self.calls.append(kwargs)
        pack = json.loads(kwargs["user_text"])
        text = json.dumps(observation_for(pack)) if pack["stage"] == "observe" else prompt_for(pack)
        if self.transform:
            text = self.transform(pack, text, len(self.calls))
        return GeminiTextResult(
            text=text,
            request_manifest={
                "model": "test-gemini",
                "system_instruction_sha256": sha256_text(kwargs["system_instruction"]),
                "user_turn_count": 1,
                "parts": [{"sha256": image.content_sha256} for image in kwargs["images"]],
            },
            provenance={},
        )


def adapter_for(tmp_path, client=None, **kwargs):
    return H3R2AVPromptAdapter(artifact_root=tmp_path / "adapter", gemini_client=client or FakeGemini(), **kwargs)


def test_twelve_images_do_not_use_echo_seven_slot_allocation(tmp_path):
    state = make_state(tmp_path, ("identity_reference",) * 12)
    prepared = adapter_for(tmp_path).prepare(state=state, raw_prompt=RAW_PROMPT)
    assert len(prepared.images) == 12
    assert [row["label"] for row in prepared.document["input"]["references"]] == [f"<Image {index}>" for index in range(1, 13)]
    assert len(prepared.document["input"]["subjects"]) == 1
    validate_h3_prompt(prompt_for(prepared.document["input"]), prepared)


@pytest.mark.parametrize("count", [0, 13])
def test_no_silent_padding_or_capacity_truncation(tmp_path, count):
    state = make_state(tmp_path, ("identity_reference",) * count)
    with pytest.raises(H3PEError, match="one to twelve"):
        adapter_for(tmp_path).prepare(state=state, raw_prompt=RAW_PROMPT)


def test_budget_is_a_host_parameter_not_a_different_planner(tmp_path):
    state = make_state(tmp_path, ("identity_reference",) * 3)
    snapshot = copy.deepcopy(state)
    with pytest.raises(H3PEError) as failure:
        adapter_for(tmp_path, max_references=2).prepare(state=state, raw_prompt=RAW_PROMPT)
    assert failure.value.code == "reference.capacity"
    assert state == snapshot
    assert len(adapter_for(tmp_path, max_references=12).prepare(state=state, raw_prompt=RAW_PROMPT).images) == 3
    assert state == snapshot


@pytest.mark.parametrize("source_kind", ["uploaded_material", "retrieved_memory", "sequence_context", "generated_evidence", "generation_derived"])
def test_actual_use_not_generation_origin_determines_role(tmp_path, source_kind):
    state = make_state(tmp_path, ("identity_reference", "exact_opening_state", "composition_anchor", "exact_ending_state"), source_kind=source_kind, purpose="reference_evidence")
    references = adapter_for(tmp_path).prepare(state=state, raw_prompt=RAW_PROMPT).document["input"]["references"]
    assert [reference["role"] for reference in references] == ["reference", "first_frame", "keyframe", "keyframe"]
    assert references[0]["definition"].startswith("<Image 1> is a reference image for <Subject 1>")
    assert references[1]["definition"] == "<Image 2> is the first frame of the target video."
    assert "middle" in references[2]["definition"]
    assert references[3]["definition"].endswith("at the end.")


def test_opening_creation_purpose_does_not_force_opening_use(tmp_path):
    state = make_state(tmp_path, source_kind="generated_evidence", purpose="opening_frame_evidence")
    prepared = adapter_for(tmp_path).prepare(state=state, raw_prompt=RAW_PROMPT)
    assert prepared.document["input"]["references"][0]["role"] == "reference"


def test_reference_at_opening_is_not_an_exact_first_frame(tmp_path):
    state = make_state(tmp_path)
    state["accepted_evidence_plan"]["ranked_bindings"][0]["timeline_role"] = "opening"
    seal_plan(state)
    assert adapter_for(tmp_path).prepare(state=state, raw_prompt=RAW_PROMPT).document["input"]["references"][0]["role"] == "reference"


@pytest.mark.parametrize("effects", [("exact_opening_state", "temporal_continuation"), ("exact_ending_state", "exact_ending_state")])
def test_competing_hard_frames_fail_before_pe(tmp_path, effects):
    client = FakeGemini()
    with pytest.raises(H3PEError):
        adapter_for(tmp_path, client).compile(state=make_state(tmp_path, effects), raw_prompt=RAW_PROMPT)
    assert client.calls == []


def test_exact_view_authorization_is_required(tmp_path):
    state = make_state(tmp_path)
    state["views"].clear()
    with pytest.raises(Echo15AdapterError) as failure:
        adapter_for(tmp_path).prepare(state=state, raw_prompt=RAW_PROMPT)
    assert failure.value.code == "evidence_not_authorized"


def test_authorized_bytes_cannot_change(tmp_path):
    state = make_state(tmp_path)
    (tmp_path / "source-1.png").write_bytes(b"tampered")
    with pytest.raises(Echo15AdapterError) as failure:
        adapter_for(tmp_path).prepare(state=state, raw_prompt=RAW_PROMPT)
    assert failure.value.code == "evidence_integrity_mismatch"


def test_image_must_really_decode_even_with_a_valid_hash(tmp_path):
    state = make_state(tmp_path)
    payload = b"not a PNG"
    (tmp_path / "source-1.png").write_bytes(payload)
    representation = state["uploaded_evidence"]["evidence_1"]["representations"]["representation_1"]
    representation.update(sha256=sha256_bytes(payload), size_bytes=len(payload))
    with pytest.raises(H3PEError) as failure:
        adapter_for(tmp_path).prepare(state=state, raw_prompt=RAW_PROMPT)
    assert failure.value.code == "reference.decode"


def test_stale_plan_is_not_compiled(tmp_path):
    state = make_state(tmp_path)
    stale = copy.deepcopy(state["accepted_evidence_plan"])
    stale["plan_id"] = "stale_plan"
    with pytest.raises(Echo15AdapterError) as failure:
        adapter_for(tmp_path).prepare(state=state, raw_prompt=RAW_PROMPT, evidence_plan=stale)
    assert failure.value.code == "stale_plan"


def test_reel_authored_prompt_cannot_be_bypassed(tmp_path):
    state = make_state(tmp_path)
    del state["accepted_evidence_plan"]["generation_prompt"]
    seal_plan(state)
    with pytest.raises(H3PEError) as failure:
        adapter_for(tmp_path).prepare(state=state, raw_prompt=RAW_PROMPT)
    assert failure.value.code == "input.generation_prompt"


def test_audio_input_is_not_silently_dropped(tmp_path):
    state = make_state(tmp_path)
    binding = state["accepted_evidence_plan"]["ranked_bindings"][0]
    binding.update(modality="audio", semantic_roles=["voice"], intended_effect="voice_reference")
    seal_plan(state)
    with pytest.raises(H3PEError) as failure:
        adapter_for(tmp_path).prepare(state=state, raw_prompt=RAW_PROMPT)
    assert failure.value.code == "reference.modality"


def test_subject_identifiers_and_paths_are_not_pe_content(tmp_path):
    state = make_state(tmp_path)
    state["accepted_evidence_plan"]["generation_prompt"] = "person_mira crosses the observatory with the brass astrolabe."
    seal_plan(state)
    prepared = adapter_for(tmp_path).prepare(state=state, raw_prompt=RAW_PROMPT)
    visible = json.dumps(prepared.document["input"])
    assert "person_mira" not in visible
    assert str(tmp_path) not in visible
    assert "evidence_1" not in visible
    assert "<Subject 1> crosses" in visible


def test_full_skill_is_system_instruction_in_both_isolated_stages(tmp_path):
    state = make_state(tmp_path, ("identity_reference", "exact_opening_state"))
    before = copy.deepcopy(state)
    client = FakeGemini()
    compilation = adapter_for(tmp_path, client).compile(state=state, raw_prompt=RAW_PROMPT)
    assert [json.loads(call["user_text"])["stage"] for call in client.calls] == ["observe", "rewrite"]
    assert all(call["system_instruction"] == SKILL_PATH.read_text() for call in client.calls)
    assert all(len(call["images"]) == 2 for call in client.calls)
    assert client.calls[0]["images"] == client.calls[1]["images"]
    assert json.loads(client.calls[1]["user_text"])["observation"]["status"] == "ready"
    assert state == before
    assert compilation.document["h3_submitted"] is False
    assert compilation.document["status"] == "pe_only"


@pytest.mark.parametrize("change", ["missing", "role", "order", "blocked", "unsupported_claim"])
def test_grounding_gate(tmp_path, change):
    prepared = adapter_for(tmp_path).prepare(state=make_state(tmp_path, ("identity_reference", "exact_opening_state")), raw_prompt=RAW_PROMPT)
    observation = observation_for(prepared.document["input"])
    if change == "missing":
        observation["references"].pop()
    elif change == "role":
        observation["references"][0]["role"] = "first_frame"
    elif change == "order":
        observation["references"].reverse()
    elif change == "blocked":
        observation["status"] = "blocked"
        observation["blocked_reasons"] = ["Cannot identify the requested subject."]
    else:
        observation["request_mapping"][0]["basis"] = "visible_reference"
    with pytest.raises(H3PEError):
        validate_observation(json.dumps(observation), prepared)


def test_blocked_grounding_never_reaches_rewrite(tmp_path):
    def transform(pack, text, count):
        value = json.loads(text)
        value["status"] = "blocked"
        return json.dumps(value)
    client = FakeGemini(transform)
    with pytest.raises(H3PEError) as failure:
        adapter_for(tmp_path, client).compile(state=make_state(tmp_path), raw_prompt=RAW_PROMPT)
    assert failure.value.code == "grounding.blocked"
    assert len(client.calls) == 1


@pytest.mark.parametrize("change", ["missing_section", "unknown_image", "picture_tag", "changed_role", "weak_opening", "timestamp", "shot", "private_path", "duplicate_image"])
def test_final_prompt_guards(tmp_path, change):
    prepared = adapter_for(tmp_path).prepare(state=make_state(tmp_path, ("identity_reference", "exact_opening_state")), raw_prompt=RAW_PROMPT)
    prompt = prompt_for(prepared.document["input"])
    if change == "missing_section":
        prompt = prompt.replace("summary:", "other:")
    elif change == "unknown_image":
        prompt += " Refer to <Image 13>."
    elif change == "picture_tag":
        prompt = prompt.replace("<Image 1>", "<Picture 1>")
    elif change == "changed_role":
        prompt = prompt.replace("<Image 1> is a reference image", "<Image 1> is the first frame")
    elif change == "weak_opening":
        prompt = prompt.replace("<Image 2>: partially_preserved", "<Image 2>: weak_reference")
    elif change == "timestamp":
        prompt += " At 00:05, Mira turns."
    elif change == "shot":
        prompt += " Shot 2 begins."
    elif change == "private_path":
        prompt += " Use /pfs2/private.png."
    else:
        prompt = prompt.replace("summary:", prepared.document["input"]["references"][0]["definition"] + "\n\nsummary:")
    with pytest.raises(H3PEError):
        validate_h3_prompt(prompt, prepared)


def test_dialogue_literals_keep_language_and_numeric_words(tmp_path):
    prepared = adapter_for(tmp_path).prepare(state=make_state(tmp_path), raw_prompt=RAW_PROMPT)
    prompt = prompt_for(prepared.document["input"])
    prompt = prompt.replace("N/A — no additional sound specified", '<Subject 1> (S1) says <d>[Chinese] 还剩3秒。</d> The sign reads "Shot 1, 5 seconds".')
    validate_h3_prompt(prompt, prepared)


@pytest.mark.parametrize("decade", ["1920s", "1970s", "1980s", "1990s", "2000s", "2020s"])
def test_decade_aesthetics_are_not_duration_instructions(tmp_path, decade):
    prepared = adapter_for(tmp_path).prepare(state=make_state(tmp_path), raw_prompt=RAW_PROMPT)
    prompt = prompt_for(prepared.document["input"]).replace("observatory naturally", f"observatory naturally with a {decade} film aesthetic")
    validate_h3_prompt(prompt, prepared)


@pytest.mark.parametrize("timing", ["5s", "5 s", "500ms", "1970 seconds", "1970 s"])
def test_duration_instructions_remain_rejected(tmp_path, timing):
    prepared = adapter_for(tmp_path).prepare(state=make_state(tmp_path), raw_prompt=RAW_PROMPT)
    prompt = prompt_for(prepared.document["input"]) + f" The action lasts {timing}."
    with pytest.raises(H3PEError) as failure:
        validate_h3_prompt(prompt, prepared)
    assert failure.value.code == "output.timing"


def test_format_repair_is_bounded_and_does_not_repeat_observation(tmp_path):
    def transform(pack, text, count):
        return "```" + text + "```" if count == 2 else text
    client = FakeGemini(transform)
    adapter_for(tmp_path, client).compile(state=make_state(tmp_path), raw_prompt=RAW_PROMPT)
    assert len(client.calls) == 3
    assert json.loads(client.calls[-1]["user_text"])["correction"]["code"] == "output.private_or_wrapped"


def test_identical_concurrent_calls_use_one_durable_pe_result(tmp_path):
    client = FakeGemini()
    adapter = adapter_for(tmp_path, client)
    state = make_state(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda unused: adapter.compile(state=state, raw_prompt=RAW_PROMPT), range(4)))
    assert len(client.calls) == 2
    assert len({result.prompt for result in results}) == 1
    assert len({result.artifact_path for result in results}) == 1


def test_resume_reuses_observation_after_rewrite_failure(tmp_path):
    client = FakeGemini(lambda pack, text, count: "invalid" if pack["stage"] == "rewrite" else text)
    state = make_state(tmp_path)
    with pytest.raises(H3PEError):
        adapter_for(tmp_path, client).compile(state=state, raw_prompt=RAW_PROMPT)
    resumed = FakeGemini()
    adapter_for(tmp_path, resumed).compile(state=state, raw_prompt=RAW_PROMPT)
    assert len(resumed.calls) == 1
    assert json.loads(resumed.calls[0]["user_text"])["stage"] == "rewrite"


def test_corrupt_cache_is_not_overwritten(tmp_path):
    client = FakeGemini()
    adapter = adapter_for(tmp_path, client)
    state = make_state(tmp_path)
    compilation = adapter.compile(state=state, raw_prompt=RAW_PROMPT)
    artifact = compilation.document
    artifact["prompt"] += " Tampered content."
    compilation.artifact_path.write_text(json.dumps(artifact))
    with pytest.raises(H3PEError) as failure:
        adapter.compile(state=state, raw_prompt=RAW_PROMPT)
    assert failure.value.code == "cache.corrupt"
    assert len(client.calls) == 2


def test_new_agent_prompt_invalidates_cached_pe(tmp_path):
    client = FakeGemini()
    adapter = adapter_for(tmp_path, client)
    state = make_state(tmp_path)
    first = adapter.compile(state=state, raw_prompt=RAW_PROMPT)
    state["accepted_evidence_plan"]["generation_prompt"] += " Keep Mira's clothing unchanged."
    seal_plan(state)
    second = adapter.compile(state=state, raw_prompt=RAW_PROMPT)
    assert first.artifact_path != second.artifact_path
    assert len(client.calls) == 4


def test_skill_profile_hash_covers_exact_file_bytes():
    profile = load_h3_profile()
    assert profile.system_instruction == SKILL_PATH.read_text()
    manifest = json.loads(profile.source_manifest_json)
    assert manifest[0]["sha256"] == sha256_bytes(SKILL_PATH.read_bytes())
    assert profile.output_contract.single_paragraph is False


def test_reference_only_prompt_cannot_claim_a_keyframe(tmp_path):
    prepared = adapter_for(tmp_path).prepare(state=make_state(tmp_path), raw_prompt=RAW_PROMPT)
    prompt = prompt_for(prepared.document["input"]).replace("[reference generation]", "[reference generation] + [keyframe completion]")
    with pytest.raises(H3PEError) as failure:
        validate_h3_prompt(prompt, prepared)
    assert failure.value.code == "output.task_type"


@pytest.mark.parametrize("payload", ["null", "[]", "broken JSON"])
def test_malformed_cache_fails_closed(tmp_path, payload):
    adapter = adapter_for(tmp_path)
    state = make_state(tmp_path)
    compilation = adapter.compile(state=state, raw_prompt=RAW_PROMPT)
    compilation.artifact_path.write_text(payload)
    with pytest.raises(H3PEError) as failure:
        adapter.compile(state=state, raw_prompt=RAW_PROMPT)
    assert failure.value.code == "cache.corrupt"


def test_canary_is_read_only_and_never_claims_h3_output(tmp_path):
    from reel_h3_pe_canary import run_case
    state = make_state(tmp_path)
    state["phase"] = "complete"
    source = tmp_path / "policy_state.json"
    source.write_text(json.dumps(state))
    original = source.read_bytes()
    services = SimpleNamespace(safe_error=lambda error: str(error))
    result = run_case(source, tmp_path / "canary", adapter_for(tmp_path), services)
    assert result["status"] == "pe_passed"
    assert result["h3_submitted"] is False
    assert result["source_state_unchanged"] is True
    assert source.read_bytes() == original


def test_canary_never_reads_active_turn_into_generation(tmp_path):
    from reel_h3_pe_canary import run_case
    state = make_state(tmp_path)
    state["phase"] = "planning"
    source = tmp_path / "policy_state.json"
    source.write_text(json.dumps(state))
    client = FakeGemini()
    services = SimpleNamespace(safe_error=lambda error: str(error))
    result = run_case(source, tmp_path / "canary", adapter_for(tmp_path, client), services)
    assert result["status"] == "pe_failed"
    assert result["h3_submitted"] is False
    assert client.calls == []


def test_inline_code_on_labels_is_normalized_without_touching_literals(tmp_path):
    prepared = adapter_for(tmp_path).prepare(state=make_state(tmp_path), raw_prompt=RAW_PROMPT)
    plain = prompt_for(prepared.document["input"])
    wrapped = plain.replace("<Image 1>", "`<Image 1>`").replace("<Subject 1>", "`<Subject 1>`")
    assert normalize_h3_prompt(wrapped) == plain
    literal = '<d>[English] Keep `<Image 1>` unchanged.</d> The sign reads "`<Subject 1>`".'
    assert normalize_h3_prompt(literal) == literal
    validate_h3_prompt(normalize_h3_prompt(wrapped), prepared)


def test_definition_punctuation_does_not_change_reference_role(tmp_path):
    prepared = adapter_for(tmp_path).prepare(state=make_state(tmp_path, ("exact_opening_state",)), raw_prompt=RAW_PROMPT)
    prompt = prompt_for(prepared.document["input"])
    prompt = prompt.replace("<Image 1> is the first frame of the target video.", "<Image 1> is the first frame of the target video, showing the requested initial state.")
    validate_h3_prompt(prompt, prepared)


def test_framing_terms_are_not_mistaken_for_added_shot_segmentation(tmp_path):
    prepared = adapter_for(tmp_path).prepare(state=make_state(tmp_path), raw_prompt=RAW_PROMPT)
    prompt = prompt_for(prepared.document["input"])
    prompt = prompt.replace("At the beginning,", "In a medium shot at the beginning,")
    validate_h3_prompt(prompt, prepared)


def test_rejected_pe_is_private_diagnostic_not_a_video_result(tmp_path):
    client = FakeGemini(lambda pack, text, count: "invalid" if pack["stage"] == "rewrite" else text)
    with pytest.raises(H3PEError):
        adapter_for(tmp_path, client).compile(state=make_state(tmp_path), raw_prompt=RAW_PROMPT)
    records = [json.loads(path.read_text()) for path in (tmp_path / "adapter" / "rejected_pe").glob("*.json")]
    assert len(records) == 2
    assert all(record["h3_submitted"] is False for record in records)
    assert all(record["status"] == "rejected_private_pe_output" for record in records)


def test_ready_with_conflicts_requires_reassessment_not_silent_acceptance(tmp_path):
    def transform(pack, text, count):
        if count == 1:
            observed = json.loads(text)
            observed["conflicts"] = ["An occluded property is established by another reference, so the difference is resolved."]
            return json.dumps(observed)
        return text
    client = FakeGemini(transform)
    adapter_for(tmp_path, client).compile(state=make_state(tmp_path), raw_prompt=RAW_PROMPT)
    assert len(client.calls) == 3
    correction = json.loads(client.calls[1]["user_text"])["correction"]
    assert correction["code"] == "grounding.inconsistent_status"
    assert json.loads(client.calls[1]["user_text"])["stage"] == "observe"


def test_reassessment_cannot_override_an_unresolved_conflict(tmp_path):
    def transform(pack, text, count):
        observed = json.loads(text)
        observed["status"] = "ready" if count == 1 else "blocked"
        observed["conflicts"] = ["Two incompatible identities remain unresolved."]
        return json.dumps(observed)
    client = FakeGemini(transform)
    with pytest.raises(H3PEError) as failure:
        adapter_for(tmp_path, client).compile(state=make_state(tmp_path), raw_prompt=RAW_PROMPT)
    assert failure.value.code == "grounding.blocked"
    assert [json.loads(call["user_text"])["stage"] for call in client.calls] == ["observe", "observe"]


def test_blank_lines_do_not_change_reference_binding(tmp_path):
    prepared = adapter_for(tmp_path).prepare(state=make_state(tmp_path, ("identity_reference", "identity_reference")), raw_prompt=RAW_PROMPT)
    prompt = prompt_for(prepared.document["input"])
    prompt = prompt.replace("\n<Image 2>", "\n\n  <Image 2>")
    validate_h3_prompt(prompt, prepared)


def test_bare_protocol_labels_keep_the_same_indices_when_serialized(tmp_path):
    prepared = adapter_for(tmp_path).prepare(state=make_state(tmp_path), raw_prompt=RAW_PROMPT)
    prompt = prompt_for(prepared.document["input"])
    bare = prompt.replace("<Image 1>", "Image 1").replace("<Subject 1>", "Subject 1")
    assert normalize_h3_prompt(bare) == prompt
    with pytest.raises(H3PEError):
        validate_h3_prompt(normalize_h3_prompt(bare + " Use Image 13."), prepared)
    literal = '<d>[English] Say Image 1.</d> The sign reads "Subject 2".'
    assert normalize_h3_prompt(literal) == literal


@pytest.mark.parametrize("prefix", ["<Image 1>: ", "<Image 1> "])
def test_repeated_label_prefix_is_only_serialization_not_a_new_reference(tmp_path, prefix):
    prepared = adapter_for(tmp_path).prepare(state=make_state(tmp_path), raw_prompt=RAW_PROMPT)
    prompt = prompt_for(prepared.document["input"])
    repeated = prompt.replace("<Image 1> is", prefix + "<Image 1> is")
    assert normalize_h3_prompt(repeated) == prompt


def test_adapter_owns_task_type_even_when_a_reference_is_scoped_to_opening(tmp_path):
    state = make_state(tmp_path, ("continuity_reference",))
    state["accepted_evidence_plan"]["ranked_bindings"][0]["timeline_role"] = "opening"
    seal_plan(state)
    prepared = adapter_for(tmp_path).prepare(state=state, raw_prompt=RAW_PROMPT)
    assert prepared.document["input"]["summary_prefix"] == "[reference generation]"
    assert prepared.document["input"]["references"][0]["role"] == "reference"


@pytest.mark.parametrize("delimiter", [": ", " ", " — "])
def test_retention_delimiter_does_not_change_preservation_mode(tmp_path, delimiter):
    prepared = adapter_for(tmp_path).prepare(state=make_state(tmp_path), raw_prompt=RAW_PROMPT)
    prompt = prompt_for(prepared.document["input"])
    prompt = prompt.replace("<Image 1>: partially_preserved", f"<Image 1>{delimiter}partially_preserved")
    validate_h3_prompt(prompt, prepared)


def test_canary_output_cannot_be_nested_in_production(tmp_path, monkeypatch):
    from reel_h3_pe_canary import main
    production = tmp_path / "production"
    production.mkdir()
    (production / "pipeline_run.json").write_text("{}")
    source = tmp_path / "source.json"
    source.write_text("{}")
    output = production / "nested" / "canary"
    monkeypatch.setattr("sys.argv", ["reel_h3_pe_canary", "--states", str(source), "--output-root", str(output)])
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code == 2
    assert not output.exists()
