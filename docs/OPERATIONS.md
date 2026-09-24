# Local execution record

The host is an NVIDIA L40S (46,068 MiB reported by nvidia-smi), not a CPU-only environment. Sandbox-only inspection could not see the driver; authorized host execution exposed CUDA.

At the user's explicit request, the existing Qwen3-1.7B router, Gemma model server and GPU-backed gateway were terminated before training. GPU occupancy fell from approximately 42.5 GiB to zero. They were not restarted. No old model weights, datasets or workspace source files were deleted or edited.

This project lives independently at `/mnt/data/s1-mor-moe`. The existing Python runtime is reused without changing its dependencies. On this host GPU commands require the same authorized host execution context; sandbox CUDA invisibility is not evidence of a broken driver.

The project does not expose a public service. The optional API starts on loopback port 8792 and serializes inference calls. It is a research server, not an authenticated multi-user deployment. Model load time, concurrent load, transport latency and HTTP queueing are outside the reported warm Python benchmarks. A FastAPI ASGI smoke test verifies success and validation-error responses.

Training uses FP32 model/optimizer state and BF16 autocast on CUDA. All model-serving workloads were stopped for the final latency tests. The trained checkpoint does not depend on an active server and can be loaded later using the README commands.
