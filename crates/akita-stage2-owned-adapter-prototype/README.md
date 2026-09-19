# Opt-in resident Metal Stage2

This Apple Silicon adapter retains the coefficient witness in a thread-local Metal owner from compact entry through coefficient folds. The default PCS API remains CPU-only. The supported field is Prime64Offset59/Ext2 and supported digit bases are 4, 8, 16, 32 and 64, subject to checked geometry and representation admission.

`AkitaCommitmentScheme::batched_prove_resident_stage2` is strict: an unsupported relation state is an error. The separate `batched_prove_hybrid_stage2` API selects the original CPU executor for ReducedDense relations before Stage2 transcript mutation; QuotientFactored uses the resident executor. Neither API catches a native error and replays on CPU. Device errors abort the proof attempt. The hybrid API does not claim complete GPU coverage.

The owner is neither Send nor Sync. Compact input is copied at admission, checked canonical factors and sparse descriptors are uploaded per round, and only messages and the final initialized witness are returned. The implementation retains correctness guard scans and is not a zero-copy or performance-validated backend. The optional `resident-stage2-observer` feature exposes execution counters.

## Build and validate

On Apple Silicon macOS with Apple Command Line Tools, Cargo builds the bundled Objective-C++ owner and embeds the bundled Metal shaders. No private archive, network download, or absolute dependency path is required:

```sh
cargo test --locked -p akita-pcs --test resident_bound6 \
  --no-default-features \
  --features parallel,transcript-blake2b,resident-stage2-observer \
  bound6_full_pcs_hybrid_matches_cpu_and_strict_rejects_reduced \
  -- --exact --nocapture --test-threads=1
```

The test defaults to the immutable application catalog in `crates/akita-pcs/tests/fixtures/resident_bound6.aks` (SHA256 `ff99e17486d9d3e2cf7cd4b9ec481a8893f746ba80ffa8ad1d485e225846496c`). `AKITA_RESIDENT_TEST_CATALOG` can select another compatible artifact explicitly. This fixture is test data, not a replacement for upstream default schedules.

For an explicitly prepared matching native archive, `AKITA_STAGE2_NATIVE_LIB_DIR` remains an optional absolute-directory override. The standalone `native/build.py` builds an archive and fixture client into a new output directory. Cargo links Metal, Foundation and C++. Native execution on other platforms is unsupported; CPU-only callers should omit resident features.

## Validation scope

The post147 bound6 test uses the unchanged public schedule policy, an NV14 polynomial, independently evaluated opening claims, two complete CPU/hybrid proofs, exact serialized proof equality, two original-verifier passes, and the next transcript challenge. The actual route is native root basis8, native first suffix basis64, and preselected CPU ReducedDense final suffix basis64. The test also requires the strict API's expected ReducedDense rejection. This publication checkout passed that test with its newly built native object, which was byte-identical to the previously audited native object.

Focused tests in the prover and sumcheck crates cover retained-owner transcript parity, additional terms, admission rejection and error propagation. Earlier native and pre147 campaigns are historical evidence only; they do not establish complete correctness of this upstream graph. No throughput or speedup claim is made. Full repository Clippy/CI matrices, schedule regeneration and additional platform validation remain required before merge. The experimental ABI is not a stable cross-version contract.
