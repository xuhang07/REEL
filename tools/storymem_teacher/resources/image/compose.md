# Image compose operation

Use this operation once, just in time, after the image purpose and complete
input-reference plan are known and every selected reference has been viewed.

- Author the first executable prompt for this image generation group directly
  from the raw request, registry state, viewed evidence, image core, and loaded
  purpose/reference-context resources.
- There is no parent prompt artifact or candidate assessment. Do not fabricate
  either one and do not create an intermediate draft artifact.
- Register the complete text with `operation="compose"`, the same purpose,
  evidence, and `has_reference_inputs` value as this skill application.
- Once registration succeeds, use that artifact to open the image generation
  group; do not register a sibling first.
