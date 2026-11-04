# Qwen3.6 35B-A3B on AMD Strix Halo

[![Repository validation](https://github.com/LeoStehlik/qwen3.6-35b-a3b-strix-halo/actions/workflows/validate.yml/badge.svg)](https://github.com/LeoStehlik/qwen3.6-35b-a3b-strix-halo/actions/workflows/validate.yml)

Reconstructable deployment and prefill experiments for Qwen3.6-35B-A3B,
ROCmFP4 weights, native MTP, and F16 KV on a Ryzen AI Max+ 395 / Radeon 8060S.
The runtime is **llama.cpp Vulkan/RADV**. ROCmFP4 names the weight format and
specialized kernels here; these measurements are **not vLLM or a HIP backend**.

The goal is sustained long-context prefill near 1,000 tok/s while preserving the
model, tools, and useful context. The largest measured improvement is a
multi-row attention port, still experimental because full-model numerical
equivalence failed on the 64K code prompt.

## Current result — 2026-11-04 UTC

| 65,536-token workload | MR off | MR on | Throughput gain |
|---|---:|---:|---:|
| Synthetic target-only, full warmup | 740.35 tok/s | **919.33 tok/s** | **24.17%** |
| Cold server request with native MTP | 638.97 tok/s | **817.07 tok/s** | **27.87%** |

Two timed runs per configuration. Same weights, F16 KV, batch/ubatch
2048/1024. Server prefill fell from 102.57 to 80.21 seconds; short generation
measurements stayed near 48–49 tok/s. Attention compute in the diagnostic
profiles fell from 38.20 to 16.62 seconds, approximately 2.30× faster.

**These are two different benchmark tracks.** The earlier ~741 tok/s baseline
belongs to the synthetic track. Server tests include MTP and use a fixed source
prompt with a 2K warmup. Neither result establishes sustained 1,000 tok/s,
150K-context performance, or end-to-end OpenClaw task throughput.

**Deployment status:** the validated MTP sparse-logit build remains the normal
server. MR attention is disabled. Its 30 operator tests and MTP regression pass,
and one prose perplexity check changed only +0.03%, but the 64K numerical gate
failed. Read [validation and limitations](docs/VALIDATION.md) before enabling MR.

## Start here

- [Francis handoff](handoff.md): current state, exact local locations, recovery,
  next experiments, and operational constraints.
- [Deployment cookbook](docs/DEPLOYMENT.md): pinned source, models, build and
  foreground launch. Repository setup does not restart any service.
- [Experiment history](docs/EXPERIMENTS.md): each step, including rejected ideas.
- [Code and patch guide](docs/CODE.md): ordered changes and runtime flags.
- [Benchmark reproduction](docs/BENCHMARKS.md): fixtures, commands and conditions.
- [Validation](docs/VALIDATION.md): successes, failures and promotion requirements.
- [Structured evidence](results/): sanitized individual runs and summaries.

```bash
git clone https://github.com/LeoStehlik/qwen3.6-35b-a3b-strix-halo.git
cd qwen3.6-35b-a3b-strix-halo
python3 scripts/prepare-source.py --profile stable
bash scripts/build.sh work/stable
python3 scripts/verify-models.py "$HOME/models/qwen36-rocmfpx"
python3 scripts/serve.py --dry-run
```

After reviewing the launch command and ensuring the intended port is free:

```bash
python3 scripts/serve.py
```

Default binding is `127.0.0.1:8082`; model ID is `tiercel-qwen36-a3b-fast`.
The launch uses the existing model files and does not download another model.
The 131,072-token served context is the tested configuration, not a claim that
150K prompts fit. Preserve checkpoint reuse and account for the context limit
in clients; no 8,192-token output cap is introduced.

## Relationship to the Qwen3.8 repository

The earlier [Qwen3.8-27B + DFlash2 deployment](https://github.com/LeoStehlik/qwen3.8-27b-strix-halo-dflash2)
remains a separate setup. This repository carries the Qwen3.6-specific source
patches, MTP work and prefill evidence. Its historical Qwen3.6 overview is useful
background, but the service state and benchmark decisions here are newer.

## Reproducibility and attribution

GitHub Actions checks the repository's JSON, fixture and patch manifests,
parses every Python file, smoke-tests the portable command-line interfaces and
reconstructs each source profile from the pinned upstream commit. It does not
run GPU benchmarks, validate model quality or prove the current deployment
state; those remain the evidence-separated checks documented above.
NumPy is needed only by the numerical and perplexity analysis helpers; install
it with `python3 -m pip install -r requirements-analysis.txt`.

The base is `charlie12345/rocmfp4-llama` commit
`4795079b04f5e0ada6e5d2e85b12bac1e27e7873`. Patch checksums and reconstructed
file checksums are recorded in [the manifest](patches/manifest.json).
The MR kernel derives from Nathan Wilson's Strix Halo work; this repository
ports and tests it on the ROCmFP4/MTP path, rather than claiming its invention.
See [third-party notices](THIRD_PARTY_NOTICES.md) for exact commits and licenses.

Original deployment scripts and documentation are MIT licensed. Upstream code
retains its MIT notices. Model weights, binaries, real agent conversations and
large tensor captures are not distributed. Code and documentation were prepared
with substantial Codex assistance for this independent deployment experiment;
this is not an upstream llama.cpp PR or an upstream-supported release.
