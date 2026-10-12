"""REEL prompt authoring when the submitted caption goes directly to Echo."""
import hashlib

from tools.reel_sft_prompt import TRAINING_SYSTEM_PROMPT as _SFT_PROMPT

TRAINING_SYSTEM_PROMPT_VERSION = "reel.joint-prompt-memory.policy-direct.v1"
TRAINING_SYSTEM_PROMPT = _SFT_PROMPT.replace(
    "A separate adapter enhances the submitted draft and maps semantic conditions to the selected video backend.",
    "Your submitted generation_prompt is sent unchanged to the video generator. There is no external prompt enhancement or rewrite. The adapter only maps your semantic evidence conditions to the backend.",
).replace(
    "The adapter still performs backend-specific prompt enhancement; do not embed channel assignments, implementation layouts, seeds, or provider syntax in the draft or rationale.",
    "You own the complete final caption, including camera and shot structure. The adapter preserves your text exactly; do not embed channel assignments, implementation layouts, seeds, or provider syntax in the draft or rationale.",
)
TRAINING_SYSTEM_PROMPT += """

## Direct caption budget

- From the first submission, the complete generation_prompt must fit within 1024 Echo Gemma tokens, including special tokens. This is a token limit, not a character limit.
- Write concise, complete natural language. Preserve the original intent, exact dialogue and grounded continuity. Do not pad, truncate, add a fixed shot count, or expect another model to improve or compress your draft.
- The host returns the exact token count when a draft exceeds the limit. Revise your own complete caption and resubmit the same pending plan version; a rejected submission has not generated a video or accepted a plan.
- If the request leaves shot structure open, choose framing and cuts that best realize the request. Preserve a justified no-cut repair. There are no mandatory ID anchors, fixed declarations, default dialogue, or Director v7 formatting requirements.
"""


def training_system_prompt_hash():
    return "sha256:" + hashlib.sha256(TRAINING_SYSTEM_PROMPT.encode()).hexdigest()
