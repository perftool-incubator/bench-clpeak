# bench-clpeak userenvs

`cuda-12.8.json` is intended for submission to Rickshaw's userenv inventory.
It uses the public NVIDIA CUDA 12.8.1 Rocky Linux 9 development image. The
image provides the CUDA compiler and development libraries; Crucible adds the
benchmark's generic build tools when composing the final image.

The pinned public tag resolves to AMD64 image digest
`sha256:e7cd1646b9d769b14b8f010377e5797f196f1376a7830d5a179086e1405b5ce8`.
Rickshaw's current userenv format references an image and tag rather than a
digest, so this digest is recorded here for provenance.

The host must run an NVIDIA Linux driver compatible with CUDA 12.8 Update 1.
NVIDIA's release notes require Linux driver `570.124.06` or newer. Podman's
`nvidia.com/gpu=all` device selector must be configured so the NVIDIA Container
Toolkit injects that host driver's device nodes and user-mode libraries into
the container. The benchmark records the effective toolkit and driver versions
in its run manifest and rejects an incompatible driver before accepting metrics.

Compatibility is directional: a newer host driver can run the older CUDA 12.8.1
toolkit, but a driver older than 570.124.06 cannot run this image's CUDA
runtime. The image does not include the host driver or a CUDA compatibility
package. Consequently, without NVIDIA Container Toolkit injection it builds
successfully but fails at `libcuda.so.1`, which is the expected no-GPU
smoke-test result.
