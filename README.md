# Qwen3.6 35B-A3B on AMD Strix Halo

[![Repository validation](https://github.com/LeoStehlik/qwen3.6-35b-a3b-strix-halo/actions/workflows/validate.yml/badge.svg)](https://github.com/LeoStehlik/qwen3.6-35b-a3b-strix-halo/actions/workflows/validate.yml)

Reconstructable deployment and prefill experiments for Qwen3.6-35B-A3B,
ROCmFP4 weights, native MTP, and F16 KV on a Ryzen AI Max+ 395 / Radeon 8060S.
The runtime is **llama.cpp Vulkan/RADV**. ROCmFP4 names the weight format and
specialized kernels here; these measurements are **not vLLM or a HIP backend**.

The goal is sustained long-context prefill near 1,000 tok/s while preserving the
model, tools, and useful context. Multi-row attention plus two follow-up
optimizations now reach about **956 tok/s** on the 64K server/MTP fixture.
The optimized endpoint is ready for the user's real OpenClaw prompt trial.

The first user-triggered Telegram request subsequently processed 50,437 prompt
tokens at **1,029.18 tok/s**, then generated 4,408 tokens at **70.20 tok/s** with
production checkpoints enabled. This is one real workload observation, separate
from the controlled 65K comparison; answer quality is for the user to assess.

## Current result — 2026-11-04 UTC

The final-build comparison keeps multi-row attention (MR) enabled in both modes:

| 65,536-token server/MTP request | MR only | MR + new optimizations | Change |
|---|---:|---:|---:|
| Prefill throughput | 821.97 tok/s | **955.71 tok/s** | **+16.27%** |
| Prefill time | 79.73 s | **68.57 s** | −11.16 s |
| Short generation check | 49.17 tok/s | **52.35 tok/s** | +6.48% |

The new changes tile convolution-input copies and omit unused MTP draft
vocabulary projections. Two earlier combined trials measured 957.56 and
952.95 tok/s. Their MR-only controls varied from 776.01 to 833.70 tok/s;
all samples are retained rather than selecting the slowest control. The table
uses one additional pair on the exact final build now serving. Same weights,
F16 KV, batch/ubatch 2048/1024; no profiler or reused prompt tokens.
[Individual measurements and correctness checks](results/prefill-next/summary.json).

**Deployment status:** MR and both new optimizations are enabled on the local
trial endpoint. Health, an actual completion, function arguments and cached
tool-result continuation passed. The user will test real OpenClaw prompts;
no synthetic OpenClaw agent turn or gateway restart was performed. The serving
unit is transient; see [Francis's handoff](handoff.md) for operational details.

The user reviewed and accepted the observed MR numerical variance as sufficient
to continue optimization and practical testing. The earlier equivalence failure
remains recorded. New operator tests and draft-state/output comparisons pass;
these do not establish broad capability equivalence. Read the
[numerical investigation](docs/NUMERICAL-INVESTIGATION.md) and
[validation scope](docs/VALIDATION.md).

## Earlier attention improvement

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

These earlier measurements isolate the attention change and must not be mixed
with the new server pair to claim a synthetic speedup. No new target-only
synthetic benchmark was run for the follow-up optimizations.

## Start here

- [Francis handoff](handoff.md): current state, exact local locations, recovery,
  next experiments, and operational constraints.
- [Deployment cookbook](docs/DEPLOYMENT.md): pinned source, models, build and
  foreground launch. Repository setup does not restart any service.
- [Experiment history](docs/EXPERIMENTS.md): each step, including rejected ideas.
- [Code and patch guide](docs/CODE.md): ordered changes and runtime flags.
- [Benchmark reproduction](docs/BENCHMARKS.md): fixtures, commands and conditions.
- [Validation](docs/VALIDATION.md): successes, failures and promotion requirements.
- [Numerical investigation](docs/NUMERICAL-INVESTIGATION.md): repeatability,
  layer tracing, causal router replay and long-context reference/quality checks.
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
The commands above retain the stable fallback. To reproduce the current trial,
prepare/build profile `prefill-next` and launch with `--optimized-prefill`;
see the [cookbook](docs/DEPLOYMENT.md).
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
