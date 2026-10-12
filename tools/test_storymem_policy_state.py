from __future__ import annotations

import json

import pytest

from storymem_policy.state import MemoryPolicyState, StateConflictError


def _create_state(tmp_path):
    return MemoryPolicyState.create(
        tmp_path / "policy_state.json",
        session_id="session",
        request_id="request",
        shot_requirement="Keep the subject in the same room.",
        condition_budget={
            "budget_id": "budget",
            "max_total_conditions": 1,
            "max_image_conditions": 1,
            "max_video_conditions": 0,
            "max_audio_conditions": 0,
            "max_text_conditions": 0,
        },
    )


def test_state_save_rejects_stale_concurrent_writer(tmp_path) -> None:
    first = _create_state(tmp_path)
    stale = MemoryPolicyState.load(first.path)

    first.data["counters"]["generation"] = 1
    first.save()

    stale.data["counters"]["generation"] = 2
    with pytest.raises(StateConflictError, match="changed since it was loaded"):
        stale.save()

    persisted = json.loads(first.path.read_text(encoding="utf-8"))
    assert persisted["counters"]["generation"] == 1


def test_state_create_refuses_to_overwrite_existing_state(tmp_path) -> None:
    state = _create_state(tmp_path)
    with pytest.raises(StateConflictError):
        _create_state(tmp_path)
    assert MemoryPolicyState.load(state.path).data == state.data


def test_state_reload_refreshes_data_and_cas_token(tmp_path) -> None:
    current = _create_state(tmp_path)
    stale = MemoryPolicyState.load(current.path)

    current.data["counters"]["generation"] = 1
    current.save()
    stale.reload()
    assert stale.data == current.data

    stale.data["counters"]["generation"] = 2
    stale.save()
    assert MemoryPolicyState.load(stale.path).data["counters"]["generation"] == 2
