# Image revise operation

Use this operation only after a real image candidate was inspected and its
immutable assessment authorized a retry whose axes require a new prompt
artifact.

- Read the assessed candidate, failed or uncertain checks, parent artifact,
  raw request, and current viewed reference context. Retain compliant content
  and make only changes justified by the assessment and declared axes.
- `axes` must exactly match the assessment. A `prompt` axis requires a textual
  correction. A `references` axis requires a revision artifact when reference
  presence changes between empty and non-empty, even if the final text happens
  to be identical.
- Do not use this operation for a seed-only retry, or for replacing references
  while they remain non-empty and the current prompt is still semantically
  valid; reuse the accepted parent artifact in those cases.
- Register exactly one child with `operation="revise"`, the same purpose as its
  parent, the full new context, and the authorizing assessment. Do not run a
  separate rewriter or create a chain of per-axis rewrites.
