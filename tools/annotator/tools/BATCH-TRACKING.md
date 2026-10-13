# Private batch tracking

These tools operate on explicitly supplied local inputs; private data and generated
annotations must remain outside the repository. Never stage datasets or outputs.

- `batch-track.py`: preparation by default; `--run` explicitly starts inference.
  Imports the same `Native` stdio bridge as the UI. CPU-only, one object per
  chronological sweep. `--experimental-efficient-video` is a separate explicit
  opt-in, not a quality certification. Use a fresh output directory when executable,
  model, source hashes, seeds, or run parameters change.
- `test-batch-tracking.py`: synthetic RLE, assignment, missing/duplicate/switch tests.
- `test-batch-resume.py`: fake-native test of completed-track reuse, deterministic
  interrupted-track replay and actual frontend persistence/COCO roundtrip.
- `verify-batch-project.mjs <project.json>`: validates through actual frontend modules,
  checks project persistence and COCO export/reload, writes sibling `.coco.json`.
- `evaluate-tracking.py --project ... --audit ... --audit-sha256 ... --output ...`:
  fixed IoU >= 0.5 all-admissible trajectory-pair counts and per-window global
  identity assignment for standard IDF1; maximum-cardinality frame matching
  for coverage/ID switches, explicit cross-window links, frozen cohort/gates.

Runner arguments: `--serve EXE --model GGUF --images DIRECTORY --seeds JSON
--output PRIVATE_DIRECTORY --first 0 --last 9`; optional
`--seed-ids 1,2 --max-frames 5 --threads 4`.
Preparation hashes all source frames, model and executable and writes a manifest
and work estimate. The inclusive `--first`/`--last` bounds are required; the synthetic example
selects `00000000.jpg` through `00000009.jpg`. Seed input
contains a `seeds` array with `proposed_fruit_id`, `frame_index`, `source_filename`
and `bbox_xyxy`. Coordinates and draft identity choices must be independently
reviewed before using them. IDs are never joined automatically.

Every frame writes a compact per-track JSON checkpoint and latency/status. Native
memory is not serialized: interruption requires replay of the incomplete track
from its seed, while complete tracks are skipped. Project assembly includes all
source frames, including empty ones. No latency/quality value is fabricated.
Missing detections remain unresolved. Re-entry starts a new segment and is
identity-unresolved; a retained proposed fruit ID is not verified re-identification.
No backward propagation or exhaustive fruit coverage is claimed.

Evaluation is box-based against an assistant visual audit, not human ground truth
or a segmentation reference. Only the predeclared audit-seeded cohort active in
each reviewed window is scored. Unknown fruits/excluded targets are not negatives.
Duplicate cohort predictions are counted; all-track overlaps are separately
reported for review, never auto-merged. Raw and manually corrected projects must
be evaluated separately (`--kind`); intervention metadata must remain truthful.
Frozen reference hash, threshold and gates must not be changed to improve results.

## Protocol implementation correction, before private inference

Independent review found that the initial implementation accumulated only the
selected per-frame Hungarian matches before its global trajectory assignment.
That is a two-stage identity-consistency diagnostic, not standard IDF1. For two
nearby overlapping reference boxes and two predictions whose box positions swap
between frames, all four trajectory pairings can remain IoU >= 0.5. The old
implementation reported IDF1 = 0.5 despite a valid fixed global assignment
matching every observation. Standard IDF1 is 1.0 in this exact regression case.

The corrected primary IDF1/IDP/IDR count every admissible trajectory-pair overlap
at the unchanged IoU >= 0.5 threshold, then maximize total IDTP by one global
one-to-one identity assignment within each frozen audit window. Each
(reference ID, predicted ID) pair contributes at most once per frame, so
duplicate predictions cannot inflate IDTP; duplicate eligible outputs still
increase the prediction denominator. The old two-stage result is retained as
`two_stage_diagnostic` and per-window diagnostic fields. Per-frame
maximum-cardinality matching still determines geometry coverage, adjacent ID
switches and explicit cross-window identity checks. Thus the overlap regression
has primary IDF1 = 1.0, diagnostic IDF1 = 0.5 and two observed frame-match switches.
The zero-switch gate remains independent and fails for that example.

This is a pre-inference implementation correction, not a threshold relaxation.
All quality gates remain unchanged. Eligibility is still frame-specific: a
frozen observation is eligible; an explicit exclusion or an unlisted
(frame, fruit) pair is unknown, never known absence. Predictions for an unknown
pair are not added to the scoring denominator just because that fruit has
observations elsewhere in the window. All-track duplicate overlaps remain a
separate review report. No reference contents or frozen hashes are changed.

The evaluator now rejects duplicate or overlapping observation/exclusion records,
unknown reference IDs, inconsistent windows/frames/association endpoints,
duplicate project source-frame indices, and invalid/nonfinite/out-of-bounds
boxes. Duplicate prediction IDs are retained as measurable errors rather than
silently collapsed. Production project validation separately rejects duplicate
IDs before export. Regression tests cover the exact overlapping-box example,
unknown versus excluded observations, duplicate outputs, malformed evidence,
nonfinite geometry and duplicate source-frame indices.
