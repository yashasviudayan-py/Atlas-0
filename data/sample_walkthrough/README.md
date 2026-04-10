# Sample Walkthrough Fixture

This directory contains the committed upload-report regression fixture for
`ATLAS-0`.

- `frames/`: five deterministic walkthrough frames rendered from a synthetic
  living-room scene with a tall lamp, a blue vase near a table edge, and a low
  stack on the opposite side of the room.
- `expected_report.json`: the minimum report characteristics that must remain
  true when the upload pipeline changes.

The benchmark and regression test use these frames directly so the expected
hazards stay stable without requiring an external camera recording.
