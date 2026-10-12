---
name: i2v-tail-frame-prompt-rewriter
description: Rewrite the complete prompt for an upcoming outer shot when the previous outer shot's final frame is supplied as the condition image and must become the first frame of an I2V continuation. Preserve the original prompt language, content, and active ordinary shot-prompt contract; change only the wording needed to anchor and continue from the condition image.
---

# I2V Tail-Frame Prompt Rewriter

Rewrite only when the current shot continues from the previous outer shot and that previous shot's final frame is supplied as the condition image. Do not use this rewrite for a fresh cut or without a condition image. Do not select, extract, upload, or generate the image or video.

Treat the condition image as the authoritative visual state at frame zero. Rewrite the already complete ordinary prompt for the upcoming shot; do not invent a different shot.

## First-frame contract

- Keep the output language exactly the same as the original prompt. Do not infer it again from the conversation and do not translate or mix languages.
- Make the first sentence exactly language-matched:
  - Chinese: `以当前图片作为视频首帧，并基于首帧中已有的人物、物体、环境、构图、机位、光线和动作状态自然延续。`
  - English: `Use the current image as the first frame of the video, and continue naturally from the characters, objects, environment, composition, camera position, lighting, and action state already shown in it.`
- When the ordinary prompt has internal cuts, place this first-frame sentence before the normal cut-count sentence, then retain every `shotN:` segment. Begin `shot1:` from the exact visible state in the condition image.
- Keep every visible character's identity, face, body, hair, clothing, position, pose, gaze, expression, and held objects consistent at the first frame. Keep the environment, props, composition, camera position, lighting, weather, and time-of-day consistent at the first frame.
- Continue the next action from that state. Do not make a present character enter again, reset a pose, repeat a completed action, rebuild the same establishing view, teleport, disappear, or appear from nowhere. Do not introduce a contradictory first-frame camera angle or background.
- Allow action, expression, blocking, camera movement, focus, and framing to evolve only after the matching first frame has been established. Make each change physically and temporally continuous from what the image shows.

## Preserve the ordinary prompt contract

The active ordinary shot-prompt system contract remains fully binding. This skill overrides only the opening order and the initial visual state required for condition-image I2V.

- Preserve the original story beat, shot intent, character IDs and bindings, dialogue text and speaker, action outcome, setting, style, sound effects, and BGM. Change only wording that must become relative to the condition image.
- Preserve the single continuous paragraph output, caption-language lock, no-language-mixing rule, cut-count syntax, all `shotN:` labels, internal-cut structure, and total length limits.
- Preserve all character identity, clothing, voice-anchor, first-appearance, later compact-description, dialogue-placement, lip-sync, action-order, silence, and timing requirements.
- Preserve all shot-scale, camera-position, camera-movement, composition, lighting, background, atmosphere, realism, renderability, sound, music, safety, no-OCR, no-readable-text, no-subtitle, and required segment-closing rules.
- Preserve technical tokens such as `ID_A`, `shot1:`, and `OCR` exactly as required by the ordinary prompt contract.
- Output only the complete rewritten generation prompt. Do not output analysis, reasoning, headings, field names, JSON, Markdown fences, alternatives, or commentary.

## Rewrite boundary

The permitted changes are limited to the language-matched first-frame sentence, state-relative action phrasing, and any first-segment visual wording required to match the supplied condition image. Do not revise unrelated content or relax any ordinary system-prompt requirement.
