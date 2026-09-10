# TRExFitter atomic-operation SFT examples

`trexfitter_atomic_examples.jsonl` is a small, hand-reviewed seed dataset. Each
row uses chat `messages` plus metadata that records the atomic operation, the
support boundary, and the TRExFitter v1.10.0 source locations used to check the
answer.

These examples are teaching material, not yet a production training corpus.
Before scaling them, add automated checks for JSON schema, referenced config
snippets, numerical answers, source revision, train/evaluation leakage, and
differential behavior against the pinned TRExFitter executable.

Recommended lifecycle for a new row:

1. Isolate one analyst operation and its preconditions.
2. Derive the answer from the pinned source, documentation, or an executable
   Hyy fixture.
3. Add a unit or differential test when the answer is machine-checkable.
4. Record whether the operation is `verified`, `unsupported`, or `downstream`.
5. Have a physicist review the analysis intent separately from syntax.
