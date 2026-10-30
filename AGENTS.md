# Bench-clpeak contribution guide

## Scope

This repository integrates upstream clpeak into Crucible. Keep the benchmark
client-only and preserve upstream measurement behavior.

## Source and software stack

- Upstream pin: `krrishnarraj/clpeak` commit
  `b5da52485bb6873aef54ff30c3fe81037e523eac`.
- Do not vendor clpeak source in this repository.
- Keep Workshop build inputs separate from runtime parameters.
- The selected userenv image owns the CUDA, ROCm, oneAPI, OpenCL, Vulkan, and
  CPU software stack. Do not add benchmark-specific SDK installers.
- Let clpeak's CMake feature probes decide which backends a selected userenv
  can build. Do not turn missing SDKs into benchmark-level fallbacks.
- Record the effective compiler, SDK, driver, backend, and source provenance.

## Runtime and results

- Preserve raw clpeak JSON output.
- Do not accept a metric from a nonzero exit, invalid JSON, or a manifest that
  disagrees with the raw result.
- Keep device, backend, test, and metric identifiers in CDM breakout names.
- Treat lower-is-better readings as latency-class metrics and use average
  aggregation unless a more specific semantic is introduced.
- Do not claim a userenv or architecture support without a Workshop build and
  native run record for that exact target.

## Validation

Run `python3 -m unittest discover -s tests -v` before handoff. Also validate
JSON schemas and Python/shell syntax when the full Crucible checkout is
available. Local unit tests do not replace a GPU image build or native run.

