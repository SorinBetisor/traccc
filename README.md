# traccc - Sorin Betisor thesis fork

This is a fork of [acts-project/traccc](https://github.com/acts-project/traccc) maintained by **Sorin Betisor** as part of a Bachelor thesis in Computer Science at Maastricht University, in collaboration with Nikhef.

The thesis topic is GPU ambiguity resolution in accelerator-based track reconstruction pipelines - specifically making the ambiguity resolution stage deterministic, benchmarkable, and GPU-native within the traccc ecosystem.

---

## What this fork adds on top of upstream traccc

### Standalone resolver benchmarks

Two dedicated executables that run only the ambiguity resolution step, decoupled from the full tracking chain:

- `traccc_benchmark_resolver` — CPU greedy baseline
- `traccc_benchmark_resolver_cuda` — GPU resolver benchmark with multiple algorithm backends selectable at runtime

Both support loading pre-frozen JSON event dumps or generating synthetic inputs, repeated timed runs with warmup, and structured key=value output for scripting.

### Ambiguity resolution I/O

`io/src/ambiguity_io.cpp` — serialise and deserialise the full resolver input state (track container + config) to/from JSON. Measurement identifiers are remapped to a dense `[0, N-1]` range on load so GPU kernels can use them as direct array indices. The `--dump-ambiguity-input` flag on `traccc_seq_example` writes a frozen snapshot immediately before resolution, making CPU/GPU comparisons reproducible on identical inputs.

### Parallel Batch Greedy (PBG)

A GPU-native greedy variant that processes conflict-free batches of tracks in parallel rather than sequentially. Exposed via `--parallel-batch` on the CUDA benchmark. Useful in low-conflict-density regimes with O(1k–10k) candidates; degrades at high density due to outer-iteration overhead.

### Explicit conflict graph resolver (Luby MIS + Jones-Plassmann)

`device/cuda/src/ambiguity_resolution/` — builds an explicit COO conflict graph from the track container and resolves it using either Luby randomised MIS or Jones-Plassmann graph colouring. Both are exposed via `--conflict-graph=mis|jp|both`. At high pile-up (µ≥400, n≥2000 candidates) JP delivers up to 85% higher throughput than the greedy baseline.

### CUDA graph reuse

The greedy eviction kernel loop can be captured as a CUDA graph on the first outer iteration and re-executed with updated launch parameters on subsequent iterations, avoiding repeated kernel-launch overhead. Enabled with `--reuse-eviction-graph` (baseline greedy path only). Gives 10–34% latency reduction depending on event size.

### Per-kernel profiling

`--profile-kernels` records CUDA event timings for each kernel inside the greedy eviction loop, reported as `greedy_*_ms` fields in the benchmark output. Useful for identifying which sub-step dominates at a given workload size.

### Determinism validation

`--determinism-runs=N` re-runs each enabled GPU backend N extra times and asserts bit-identical selected-track sets across all runs. Used to verify that all GPU resolver paths produce stable output.

### Correctness metrics

Per-run output includes FNV-1a hashes of the selected track set, CPU/GPU hash comparison (`hash_match`), Jaccard similarity between CPU and GPU selections, and duplicate-rate reporting. The `--truth-file` option accepts a TSV of per-track particle associations and enables efficiency and fake-rate reporting.

---

## Building

Same as upstream traccc. Requires CMake ≥ 3.25, CUDA toolkit, and the acts-project dependencies. On Nikhef Stoomboot:

```bash
spack env activate /path/to/spack/environments/traccc
cmake -S . -B build -DTRACCC_BUILD_CUDA=ON -DTRACCC_BUILD_BENCHMARKS=ON
cmake --build build -j$(nproc)
```

---

## Upstream

The original traccc project lives at [https://github.com/acts-project/traccc](https://github.com/acts-project/traccc) and is developed by the ACTS collaboration. This fork does not intend to diverge from upstream in any permanent way; the additions are thesis prototypes.