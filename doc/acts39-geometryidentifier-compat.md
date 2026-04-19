# Acts 39 GeometryIdentifier Compatibility

This branch carries a small build-compatibility fix for environments using
`Acts 39.x`.

## Problem

`traccc` JSON readers in:

- `io/src/json/read_conditions_config.cpp`
- `io/src/json/read_digitization_config.cpp`

were written against a `GeometryIdentifier` API that provides:

- `withVolume(...)`
- `withLayer(...)`
- `withSensitive(...)`

The Spack environment used for CUDA validation on this machine provides
`Acts 39.0.0`, where the available mutating API is:

- `setVolume(...)`
- `setLayer(...)`
- `setSensitive(...)`

Without this compatibility patch, clean builds fail before the CUDA test binary
can be rebuilt.

## Scope

This branch is intentionally separate from the CUDA ambiguity resolver bugfix
branch so that the ambiguity PR can stay focused on the resolver correctness
change and the test re-enable.
