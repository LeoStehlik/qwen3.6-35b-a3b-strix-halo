# Qwen3.6 35B-A3B on AMD Strix Halo

[![Repository validation](https://github.com/LeoStehlik/qwen3.6-35b-a3b-strix-halo/actions/workflows/validate.yml/badge.svg)](https://github.com/LeoStehlik/qwen3.6-35b-a3b-strix-halo/actions/workflows/validate.yml)

Reconstructable deployment and prefill experiments for Qwen3.6-35B-A3B,
ROCmFP4 weights, native MTP, and F16 KV on a Ryzen AI Max+ 395 / Radeon 8060S.
The runtime is **llama.cpp Vulkan/RADV**. ROCmFP4 names the weight format and
specialized kernels here; these measurements are **not vLLM or a HIP backend**.

**Status — 2026-11-05: daily OpenClaw acceptance withdrawn.** Real use exposed
an output-budget-exhausting reasoning loop, failure to retrieve accessible
channel history, and unsupported recovery explanations. The user is returning
to Qwen3.8-27B dense; Qwen3.6 is stopped and disabled at boot, with all artifacts
retained for offline experiments. Qwen3.8 GPU serving and direct API checks pass. See the [setback findings](docs/OPERATIONAL-SETBACK.md)
and [current handoff](handoff.md).

The measured **1,045 and 1,049 tok/s** cold 64K server/MTP result remains valid.
The frozen `prefill-stage2-2026-11-04` tag, code and evidence are preserved.
Passing bounded tool/state checks did not establish reliable autonomous agent
behavior. The root cause of the live failures has not been isolated.

## Preserved performance result — 2026-11-04 UTC

Two runs per configuration, same weights, F16 KV, batch/ubatch 2048/1024,
MR attention and native MTP. The control includes the previously accepted tiled
concat and unused draft-head changes. Profilers, prompt reuse and checkpoints
are disabled for these timings.

| 65,536-token server/MTP request | Previous baseline | Expert selector + fused convolution |
|---|---:|---:|
| Prefill samples | 958.91, 954.49 tok/s | **1,045.40, 1,048.68 tok/s** |
| Mean throughput | 956.70 tok/s | **1,047.04 tok/s (+9.44%)** |
| Mean prefill time | 68.50 s | **62.59 s (5.91 s saved)** |
| Short generation check | 52.33 tok/s | **52.45 tok/s** |

Expert row selection alone added 5.93%; convolution fusion alone added 3.20%.
Their independent elapsed-time savings predicted about 1,048 tok/s, matching
the combined result. All ten uninstrumented runs have the same output hash and
draft counts. [Experiment details](docs/EXPERT-PREFILL.md),
[all measurements](results/prefill-stage2/server-runs.json).

**Historical deployment, 2026-11-04 23:54 UTC:** `prefill-stage2` was accepted.
That daily-use acceptance was withdrawn on November 5.
At promotion, the enabled boot unit was aligned with the running binary,
arguments and optimization flags; lingering was enabled. The transient process
continued without interruption. No restart or reboot was performed to change
persistence; a reboot test was not performed. Direct tool, continuation and displaced-cache checks
passed at 11K. See [Francis's handoff](handoff.md) and
[persistence evidence](results/combined-profile/persistence.json).

Initial real traffic on the promoted pair processed **73,093 tokens at 1,012.02
tok/s** and a later **79,548-token pass at 993.07 tok/s**, with production
checkpoints enabled. Follow-ups restored checkpoints beyond 72K. These are
[workload observations](results/prefill-stage2/live-traffic-observations.json),
not paired benchmarks or an answer-quality verdict.

The previous step reached about 956 tok/s, up from 821.97 in its paired control.
On that previous build, a user-triggered Telegram request processed 50,437 tokens
at 1,029.18 tok/s and generated 4,408 tokens at 70.20 tok/s. Those historical
workloads are separate from the new controlled comparison. See
[experiment history](docs/EXPERIMENTS.md).

The user accepted the earlier MR numerical variance as sufficient to continue
optimization and practical testing. Its failed equivalence gate remains recorded.
New operator, row-ownership and state checks pass; these do not establish broad
capability equivalence. Read the [numerical investigation](docs/NUMERICAL-INVESTIGATION.md)
and [validation scope](docs/VALIDATION.md). Performance testing of this pair is
complete. The subsequent real-use failure is documented separately; it does
not establish a regression in a particular kernel.

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

- [Operational setback](docs/OPERATIONAL-SETBACK.md): real-use failures, evidence
  limits, withdrawal of daily-use acceptance and the move back to Qwen3.8.
- [Francis handoff](handoff.md): current state, exact local locations, recovery,
  next experiments, and operational constraints.
- [Deployment cookbook](docs/DEPLOYMENT.md): pinned source, models, build and
  foreground launch and optional persistent systemd setup.
- [Combined-build profile](docs/COMBINED-PROFILE.md): long-context continuation
  bottlenecks, native counters and the next measured opportunity.
- [Upstream candidates and fork strategy](docs/UPSTREAM.md): what merits a PR,
  existing overlapping work and what we maintain downstream.
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
python3 scripts/prepare-source.py --profile prefill-stage2
bash scripts/build.sh work/prefill-stage2
# New installation only: downloads about 22 GB from a pinned public revision.
python3 scripts/download-models.py "$HOME/models/qwen36-rocmfpx"
python3 scripts/verify-models.py "$HOME/models/qwen36-rocmfpx"
python3 scripts/serve.py --dry-run
```

After reviewing the launch command and ensuring the intended port is free:

```bash
python3 scripts/serve.py
```

Default binding is `127.0.0.1:8082`; model ID is `tiercel-qwen36-a3b-fast`.
Source preparation, build and launch default to **prefill-stage2** for research
reproduction; this is no longer the accepted daily OpenClaw deployment.
The earlier fallback remains explicit: prepare/build `stable`, then launch with
`--profile stable`. `--expert-prefill` remains a compatible alias for the default.
Only the explicit download command fetches models; launch never does.
For persistent boot setup, follow the [cookbook](docs/DEPLOYMENT.md).
The 131,072-token served context is the tested configuration, not a claim that
150K prompts fit. Preserve checkpoint reuse and account for the context limit
in clients; no 8,192-token output cap is introduced.

## Relationship to the Qwen3.8 repository

The earlier [Qwen3.8-27B + DFlash2 deployment](https://github.com/LeoStehlik/qwen3.8-27b-strix-halo-dflash2)
is the next deployment/optimization target after the real-use setback. This
repository retains the Qwen3.6-specific source patches, MTP work and prefill
evidence. Do not transfer MoE kernel policies or measured bottleneck shares to
the dense model without profiling its actual execution path.

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
