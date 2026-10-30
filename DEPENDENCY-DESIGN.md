# Dependency and execution design

## Responsibility split

| Layer | Responsibility |
|---|---|
| Userenv image | OS, compiler, CUDA/ROCm/oneAPI SDK, driver-compatible userspace libraries |
| Workshop | Acquire pinned clpeak source, copy the build driver, build and install the binary |
| Runtime wrapper | Select tests, devices, and time budgets; execute the installed binary; validate output |
| Post-processing | Parse clpeak JSON and emit CDM metrics with device/test/variant breakout names |

## Why the GPU stack belongs in the userenv

Crucible executes remotehosts engines through Podman or a chroot derived from
the selected userenv image. GPU device nodes and the host kernel driver can be
exposed, but the userspace stack used by the benchmark is the one inside the
image.

Therefore:

1. A hardware-comparison run should pin the image's GPU stack as part of the
   benchmark definition.
2. A run that intends to characterize a SUT's installed stack must select a
   userenv image matching that stack, or use an explicit host-stack mode.
3. Silent mixing of host and container GPU libraries is not supported.

This design uses one clpeak source revision and one build driver for every
supported userenv. CMake detects the available SDKs in the image and enables
only those backends. This keeps vendor/version changes at the userenv layer.

OpenCL is disabled because upstream clpeak fetches the OpenCL-SDK `main`
branch at configure time when a complete system SDK is not present. Vulkan is
disabled because its build requires `glslc` and a current shader toolchain.
Both backends would undermine source pinning or broaden the required userenv
software stack. The initial integration therefore targets CUDA, ROCm/HIP,
oneAPI/SYCL, and CPU.

The upstream Git submodules in the source archive are used by the disabled
GUI and mobile build paths. The Linux CLI build does not require them when
OpenCL and Vulkan are disabled.

## Build inputs

The Workshop build:

1. Installs generic build tools.
2. Copies `clpeak-build.py` into the image.
3. Downloads the pinned clpeak source archive.
4. Configures clpeak with GUI, mobile AI, OpenCL, and Vulkan disabled.
5. Builds and installs the CLI.
6. Writes a provenance JSON file beside the installed binary.

## Runtime inputs

The runtime wrapper:

1. Validates the requested clpeak flags and time budgets.
2. Constructs a clpeak command with `--output`.
3. Runs the installed binary under a watchdog.
4. Captures stdout, stderr, exit status, and process timestamps.
5. Parses and validates the resulting JSON.
6. Writes a run manifest and removes accepted results on failure.

## Output contract

The engine working directory contains:

- `clpeak-result.json` — raw upstream result
- `clpeak-client-stderrout.txt` — combined native output
- `clpeak-run-manifest.json` — effective parameters, command, status, timing,
  source/build provenance, and parsed-result digest
- `clpeak-error.txt` — present only after a wrapper failure

Post-processing indexes every valid metric from the raw JSON. Metric breakout
names include backend, device, device index, test, test variant, metric, and
unit. Unsupported, skipped, and errored readings are not indexed.
Higher-is-better readings use throughput aggregation; lower-is-better readings
use latency aggregation and average aggregation.

