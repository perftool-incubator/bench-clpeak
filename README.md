# bench-clpeak

`bench-clpeak` integrates [clpeak](https://github.com/krrishnarraj/clpeak)
into Crucible as a client-only GPU and CPU microbenchmark suite.

clpeak measures arithmetic throughput, memory bandwidth, cache behavior,
kernel-launch and memory latency, vendor-library GEMM, attention, and a
transformer-decoder composite across CUDA, ROCm/HIP, oneAPI/SYCL, and native
CPU backends.

## Current scope

- Upstream source pin: `krrishnarraj/clpeak` commit
  `b5da52485bb6873aef54ff30c3fe81037e523eac`.
- Client-only workload; no server role is used.
- The selected userenv image owns the GPU software stack. This repository
  builds the same pinned clpeak source and lets CMake feature-detect CUDA,
  ROCm/HIP, oneAPI/SYCL, and CPU support in that image.
- Initial runtime parameters select clpeak categories, devices, and time
  budgets. They do not install or change GPU software.
- Raw clpeak JSON and a run manifest are preserved. Metrics are indexed into
  CDM with backend, device, device index, test, test variant, metric, and unit
  breakout names; unsupported, skipped, and errored readings are not indexed.
- `clpeak-get-runtime` reports the effective `deadline`; the client forwards
  termination signals to native clpeak and rejects interrupted samples.
- CUDA runs record `nvcc` and `nvidia-smi` versions in the manifest and reject
  a CUDA 12.8.1 image paired with a host driver older than 570.124.06.

## GPU software stacks

The benchmark does not attempt to replicate every possible host installation
inside its own Workshop recipe. Instead, use a userenv image that already
contains the intended GPU stack:

- `default` is suitable for CPU smoke and integration testing.
- `cuda-12.8` uses NVIDIA CUDA 12.8.1 and requires a host Linux NVIDIA driver
  of 570.124.06 or newer. See `userenvs/cuda-12.8.json`.
- `rocm` is intended for an image containing the desired ROCm/HIP stack.
- `oneapi` is intended for an image containing the desired Intel oneAPI/DPC++
  and oneMKL stack.

Those userenv definitions are deployment inputs, not benchmark parameters.
The same clpeak source revision and build driver are used for all of them.
CMake enables a backend only when its SDK is present, which keeps the
benchmark repository independent of a particular vendor release.

OpenCL and Vulkan are intentionally disabled in the initial integration.
Upstream's OpenCL build fetches the OpenCL-SDK `main` branch, and its Vulkan
backend requires `glslc`; enabling either would add an unpinned or
userenv-specific build dependency. They can be revisited with pinned source
and shader-toolchain requirements later.

## Runtime parameters

| Parameter | Default | Description |
|---|---|---|
| `tests` | `all` | Comma-separated clpeak category or test flags |
| `devices` | *(empty)* | Optional comma-separated `backend:index` selector |
| `max-time-gpu` | `250` | Milliseconds a GPU may remain busy per calibration step |
| `max-time-cpu` | `2000` | Milliseconds a CPU test may use |
| `deadline` | `900` | Wrapper watchdog in seconds |

`tests=all` runs every test compiled into the binary. Individual flags such as
`compute`, `bandwidth`, `latency`, `ai`, `gemm`, `attention`,
`transformer-block`, `cache-bandwidth`, and `memory-latency` are passed to
clpeak as `--<flag>`.

## Files

- `workshop.json` — build requirements and supported userenv selections
- `rickshaw.json` — Crucible client-only execution contract
- `multiplex.json` — runtime parameter validation
- `benchmark-metadata.json` — CDM source and output manifest
- `clpeak-build.py` — pinned source build and provenance writer
- `clpeak-client.py` — runtime wrapper, validation, and manifest writer
- `clpeak-post-process.py` — JSON parser and CDM metric indexer
- `clpeak_common.py` — shared parsing and validation helpers
- `examples/` — representative remotehosts run files
- `tests/` — focused parser and wrapper unit tests

## Validation

Run the focused unit tests:

```sh
python3 -m unittest discover -s tests -v
```

These tests do not build clpeak or execute a GPU workload. A complete
acceptance run still requires:

1. A Workshop image build for the selected GPU userenv.
2. A native clpeak execution on the intended GPU.
3. Validation of raw JSON, run manifest, exit status, and CDM indexing in a
   full Rickshaw run.

Local validation completed on AlmaLinux 9 with the declared CPU-only package
set: the pinned source built through `clpeak-build.py`, a native CPU compute
run produced 50 valid metrics, and `clpeak-post-process.py` emitted all 50 to
CDM through Toolbox.

The pinned CUDA 12.8.1 Rocky Linux 9 image also builds the CUDA backend. Its
recorded AMD64 digest is
`sha256:e7cd1646b9d769b14b8f010377e5797f196f1376a7830d5a179086e1405b5ce8`.
Without a host GPU and injected driver libraries, the resulting binary reaches
the expected `libcuda.so.1` load failure rather than silently reporting an
empty device inventory.

