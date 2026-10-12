# Provider-neutral image prompt core

Write a prompt for one still image. Describe only what can be visible in that
single frame: subjects, stable appearance, wardrobe or object state, pose,
spatial relationships, environment, composition, viewpoint, lighting, palette,
texture, and visual style.

- Preserve the raw request and all relevant viewed evidence. When a stable
  entity is present, state enough concrete visual facts for recognition; never
  replace an explicit current-shot change with an older wardrobe or state.
- Do not describe a sequence of future actions, completed dialogue, sound,
  music, editing, or multiple temporal phases as if a still image could render
  them. A first frame may state an initial action posture, but not the later
  outcome.
- Keep the image coherent and directly renderable. Resolve unspecified visual
  implementation details conservatively without inventing new narrative
  entities or events.
- When references are present, use their actually observed visual facts. The
  prompt must remain self-contained and must not refer to a reference by index,
  filename, path, URL, internal ID, or provider name.
- Use the story-level caption language and the common artifact hand-off. The
  current agent writes the prompt; the image tool does not rewrite it.
