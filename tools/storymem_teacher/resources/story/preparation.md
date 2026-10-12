# StoryMem story-plan output adapter

This resource defines the exact hand-off from the offline story planner to the
per-shot StoryMem trajectories. It is an output adapter, not an interactive
Director workflow and not a generator prompt writer.

## Invocation and output mapping

- All available story material is in `source_story` in the JSON request. Treat
  it as planning input. This call has no tools and must not simulate tool calls,
  workspace state, user confirmations, media inspection, or generation.
- Return exactly one JSON object matching `required_output`; do not call tools,
  emit Markdown, or include prose outside that object.
- Preserve the source premise, named subjects, requested ordering, setting,
  style, ending, and requested number of outer shots. Resolve only ambiguity
  necessary to make coherent approximately ten-second beats. Do not write a
  generator-ready image or video caption here and do not optimize wording for
  any named media provider.
- Lock `caption_language` and aligned `dialogue_language` once under the
  offline planner's language policy. Preserve explicit speech or silence. Every
  `raw_request` must resolve this intent explicitly with either the exact
  language-matched no-dialogue sentence or the full intended dialogue marker
  defined by the planner. Do not manufacture dialogue merely because a person
  is visible or because a later generator may support audio.
- A speaking shot must activate at least one source-declared human material.
  Person-free and entity-only shots use the exact no-dialogue marker, so the
  downstream Director never invents an unregistered `ID_A` speaker.

## Canonical material taxonomy

- `person` means an actual human being only. Assign human tags `ID_A`, `ID_B`,
  and so on in declared/first-appearance order. A stylized depiction of a
  human is still a person, but an animal, robot, android, alien, creature,
  mascot, or anthropomorphic non-human is not.
- `entity` means a recurring visually identifiable non-human subject. Assign
  `ENTITY_1`, `ENTITY_2`, and so on. This includes pets and other animals,
  robots and androids, fantasy creatures, and anthropomorphic non-humans.
  Never assign an `ID_X` tag or `category="person"` to one of them, even if it
  speaks, wears clothing, has a face, or is called a character in the source.
- Use `scene`, `object`, and `style` for locations, props, and visual style.
  A recurring non-human protagonist is `entity`, not `object`.
- Preserve each source character or entity's natural name/role in `aliases`
  and its stable visible facts in `description`. Do not silently replace one
  source subject with another or merge a human and a non-human.

## Shot hand-off

- Produce one `raw_request` per outer shot. It contains only the natural
  next-shot story beat and explicit user constraints, including intended
  dialogue, audio, or silence. Speech intent is never omitted: use exactly one
  language-matched dialogue/no-dialogue marker from the planner. It contains no PE boilerplate,
  generator syntax, memory catalog, upload schedule, hidden budget, previous
  prompt, tool instruction, or canonical material tag.
- Use natural character, entity, object, and place names in every
  `raw_request`. Keep `ID_A`, `ENTITY_1`, scene/object/style tags, and all other
  registry identifiers only in `active_material_tags` and `prompt_state`.
- Put the language lock, stable facts, visible scene state, explicit state
  transitions, first-seen information, and previous-prompt lineage in the
  structured plan fields. Media-derived facts still require a real later
  `view_memories` observation before a shot agent may rely on them.
- `active_material_tags` contains every declared material actually used in that
  beat and no undeclared tag. For `person` materials it is exactly the set of
  people visible in the shot (a speaking person is visible under the frozen
  Director profile); do not include an off-screen person merely because the
  request mentions them. Keep stable identity/appearance/voice facts separate
  from transient pose, expression, action, and shot-specific state.
