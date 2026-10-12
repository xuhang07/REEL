## ECHO ENCODER BUDGET — FINAL OUTPUT AUTHORITY

Apply this budget from the first generation of the caption, including base
captions and image-conditioned opening captions. It overrides every earlier
length target and every example that would exceed this budget.

- The COMPLETE final caption must contain at most **1024 Echo Gemma tokens**,
  including special tokens, cut-count prefix, all shot labels, identity and
  clothing anchors, exact dialogue, required declarations and any opening-frame
  preamble. This is one shared budget for the whole caption, never per segment.
- Plan the complete caption inside this budget before writing it. Allocate room
  for all required elements first, then write concise scene and camera prose.
  Do not first write an overlong caption and rely on another PE compression call.
- Chinese captions have NO 1500-character minimum. English captions also have
  NO minimum character count. There is NO separate 1999-character ceiling in
  this Echo path: the exact 1024-token limit is authoritative for both languages.
  Do not expand a short request to reach an earlier character target. Prefer
  concise, complete prose and use room only for required meaning, never
  decorative padding or invented events. Tokens and characters are different
  units; do not confuse a character count with the encoder token budget.
- Preserve the immutable user intent, exact dialogue, continuity, required
  anchors and declarations. Return one complete paragraph with no explanations,
  token counts, truncation or omitted required content.
- The host checks the complete caption with the exact Echo tokenizer. An
  over-budget answer is rejected; it is never clipped or passed to Echo.
- Preserve the complete source cast from the private immutable semantic
  contract. There is no fixed limit on the number of people; retain every
  required canonical character ID. Never drop source characters, invent IDs,
  or turn a comparison-only reference into a new visible action. All source roles and required anchors
  still share the same 1024-token caption budget.
