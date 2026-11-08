# Compute backends (Metal/GPU)

> **Status:** stub. Part of the initial Akita Book scaffold.

The CPU cutover has landed; the open work is a Metal (and broader GPU) backend
behind the explicit compute-operation traits, plus hybrid scheduling.

The Metal crate also has a specialized inner commitment for 16-entry one-hot
chunks over the 128-bit field at ring degree 512. It replaces repeated key
additions with small-prime NTTs of 16-bit key limbs and reconstructs exactly
the same field result. Key preparation is reusable across witness lengths.
The transformed key uses twice the raw key storage; packed source indices
use one byte instead of four. This is a compute kernel, not a complete Metal
prover. The implementation and its bounds are recorded in the specification
below.

## Sources to fold in

- `specs/akita-compute-backend-metal.md` (Metal tail)
- `docs/compute-backends.md` (current boundary)
