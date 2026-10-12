# StoryMem tool hand-off adapter

This final resource maps the official Director PE output boundary onto the
StoryMem tool protocol. It does not replace, summarize, relax, or reinterpret
the official prompt-writing rules returned earlier in this bundle. If wording
here appears to conflict with an official resource, follow the official
resource for all generator-prompt semantics.

## Input mapping

- The official writer's current-shot input is the immutable raw USER request.
  The equivalent of its story-level conversation, cast sheet, and previous
  prompt is supplied only by the current `story_prompt_state` and the registry
  and media observations bound to this skill receipt. Do not ask the USER to
  paste any of that state.
- Treat only actually viewed media as visual or acoustic evidence. Registry
  text and search results are retrieval metadata, not proof of media content.
- The official `ID_X` namespace remains human-only. A registry `ENTITY_N` tag
  identifies a recurring non-human entity such as an animal or robot; never
  turn it into an `ID_X` person. Refer to that entity naturally in the prompt
  and use viewed entity media when available. `ENTITY_N` is registry metadata,
  not required generator syntax.
- Structured condition and memory-slot bindings are delivered to the
  generator separately. Do not mention file paths, URLs, internal handles,
  slot numbers, or phrases such as `reference image 1` in prompt prose.

## Operation mapping

- For `video.compose`, author one complete ordinary prompt under the official
  shot-prompt writer. If the condition is `temporal_continuation`, internally
  apply the official tail-frame rewriter in this same authoring pass and
  register only its final complete result. This is not a second model or a
  second prompt artifact.
- For `video.revise`, start from the exact parent artifact and change only what
  the bound assessment and retry axes require while keeping every still-valid
  official constraint. Register one complete replacement prompt, never a
  patch, explanation, or abbreviated delta. If temporal continuation remains
  bound, the returned tail-frame skill remains fully applicable.
- The official instruction to output only the prompt means: put those exact
  prompt bytes in the immediately following `register_prompt.text` call. Do
  not emit them as chat prose. `register_prompt` validates and stores the text
  verbatim; it never rewrites it.
