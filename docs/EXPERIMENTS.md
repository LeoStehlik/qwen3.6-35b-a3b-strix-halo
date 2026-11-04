# Experiment history

All dates are UTC. Rows with retained JSON evidence link to it. A negative
result remains part of the record; gains from different benchmark tracks or
different baselines must not be added together.

## 1. Establish the Qwen3.6 deployment — 2026-10-31

The separate ROCmFP4 fork and 35B-A3B model were built with explicit
`-O3 -DNDEBUG`, Vulkan, F16 target/draft KV and native MTP width 5. The initial
deployment measured 838.91 tok/s prefill and 101.16 tok/s decode on a
28,761-token prompt. Tool selection, streaming and vision were checked.
Repeated-prefix restoration reduced the roughly 28.8K prompt's prefill from
36.65 seconds to 2.13–2.17 seconds with 27,731 tokens reused.

These are **historical 28.8K results**, not the later 64K baseline. Their original
record is in the [Qwen3.8 companion repository](https://github.com/LeoStehlik/qwen3.8-27b-strix-halo-dflash2/blob/main/docs/QWEN36-ROCMFP4-MTP.md).

## 2. Establish the long-context prefill configuration

The retained working configuration uses one slot, batch 2048, ubatch 1024,
F16 KV, flash attention, KV contiguity, MoE row lists, and convolution-tail graph
preparation. The first patch preserves these changes and their operator tests.

Earlier interactive sweeps considered ubatch 512 and 4096 and batch/ubatch
4096/1024. Their raw run records were not found while creating this repository,
so no reconstructed timing table is presented. The choice 2048/1024 is the
configuration reproduced in the retained experiments below. There is no claim
that each inherited optimization has an independently measured contribution.

One slot is a concurrency setting, not a limit of one output token or a way to
split a single sequential agent request across four slots. This work retained
the model and focused on the backend rather than reducing OpenClaw's tools.

## 3. Indirect MoE dispatch — 2026-11-03

Ported Nathan Wilson's indirect dispatch sized from expert row-list counts.
176 CPU-reference cases passed. Two full-warmup 64K synthetic trials per mode:

| Off | On | Change | Decision |
|---:|---:|---:|---|
| 741.01 tok/s | 741.57 tok/s | +0.08% | No demonstrated gain; disabled |

The code remains in patch 0001, controlled by `GGML_VK_MMID_INDIRECT=0` in
normal serving. [Evidence](../results/indirect-dispatch/results.json).

## 4. Qwen3.8 batch comparison — 2026-11-03

One cold 65,536-token server trial per setting, same tokenized source prompt.
Qwen3.8 used Q8 KV and DFlash2; Qwen3.6 used F16 KV and native MTP. This compares
complete stacks, not isolated attention implementations or model quality.

| Stack | Batch/ubatch | Prefill | Generation |
|---|---|---:|---:|
| Qwen3.8 | 2048/2048 | 367.33 tok/s | 16.06 tok/s |
| Qwen3.8 | 2048/1024 | 354.59 tok/s | 16.14 tok/s |
| Qwen3.6, before sparse-logit fix | 2048/1024 | 627.77 tok/s | 47.71 tok/s |

The Qwen3.6 ubatch choice did not transfer into a gain for Qwen3.8 in this
single-run comparison. [Evidence](../results/qwen38-comparison/results.json).

## 5. Separate MTP hidden states from vocabulary logits — 2026-11-03

MTP needs target hidden states for every prompt token. The old server enabled
generic embeddings to obtain them, also causing vocabulary logits for every
token. At 65,536 tokens and vocabulary 248,320, the dense F32 logit output is
60.625 GiB versus 0.5 GiB of required hidden states. Those byte counts are a
work estimate, not a promised elapsed-time saving.

Patch 0002 keeps all hidden rows while requesting only the logits required for
sampling and speculative verification. It preserves physical batch splitting
and input row order, including zero-logit batches.

| Server/MTP off | On | Gain | Decision |
|---:|---:|---:|---|
| 634.91 tok/s | 655.60 tok/s | +3.26% | Enabled in the validated server |

Two trials per mode, identical benchmark outputs and draft counts. Seven
hidden-state comparisons and 31 logit comparisons passed, including checkpoint
restore and multiple sequences. Runtime cache checks reused 11,278 tokens.
[Evidence](../results/mtp-sparse-logits/summary.json).

## 6. Smaller per-expert MoE tiles — 2026-11-03

Tested 32×32 rather than 64×64 tiles for the effective ROCmFP4 fast path.
32 routing reference cases and the MTP regression passed; all four server
outputs matched. Performance was worse:

| Track | Off | On | Change |
|---|---:|---:|---:|
| Synthetic | 742.18 | 712.37 | −4.02% |
| Server/MTP | 652.31 | 629.00 | −3.57% |

Units are tok/s. The separate `moe-tiles` profile preserves this experiment;
it is absent from the stable/MR patch sequence and was not adopted.
[Evidence](../results/smaller-moe-tiles/summary.json).

## 7. Measure attention rather than infer from copy volume — 2026-11-03

Patch 0003 adds timestamp measurements for attention repacking, mask work and
core attention, plus existing operator profiling. It found 64 target graphs
with 640 attention calls and 64 draft graphs with 64 calls on the 64K request.

Target plus draft attention compute took 38.20 seconds; repacking took 0.99
seconds. Repacking's large byte count therefore did not make it the primary
wall-time target. Attention represented about 37% of the instrumented request;
its share grew with context depth. Target convolution-input concatenation also
cost approximately 8.44 seconds.

Instrumentation slowed this server comparison from 661.45 to 634.11 tok/s,
about 4.31% more elapsed prefill time. Profiled results are diagnostic, never
the headline throughput benchmark. [Evidence](../results/attention-profile/summary.json).

With an approximate 40% attention share, Amdahl's law predicts total speedup
`1 / (0.6 + 0.4 / attention_speedup)`: 1.5× attention gives +15.4%, 2× gives
+25%, and 3× gives +36.4%. This motivated an experiment, not a guaranteed gain.

## 8. Multi-row F16 attention port — 2026-11-03/04

Ported Nathan Wilson's multi-row coopmat1 kernel and subsequent scheduling
changes into the pinned ROCmFP4 fork. H=2 groups more queries per workgroup,
reuses K/V, and uses a padded transposed V layout. Dispatch is restricted to the
validated F16, head-size-256, RADV/Strix-compatible prefill path; unsupported
shapes and decode fall back to the existing implementation.

| Track | Off | On | Gain |
|---|---:|---:|---:|
| Synthetic 64K | 740.35 tok/s | 919.33 tok/s | +24.17% |
| Server/MTP 64K | 638.97 tok/s | 817.07 tok/s | +27.87% |

Synthetic samples: 740.804, 739.903 versus 917.719, 920.939. Server samples:
644.045, 633.887 versus 819.976, 814.165. Server prefill saved 22.36 seconds.
The 128-token generation check averaged 47.91 versus 48.89 tok/s.

Core attention in a separate candidate profile took 16.62 seconds versus the
previous profile's 38.20 seconds (2.30×). Repacking rose to 1.34 seconds.

**Decision: retain as an experiment, not the serving default.** Operator tests
and a prose perplexity comparison are encouraging, but the predefined model
numerical-equivalence gate failed. Full investigation and limits are in
[VALIDATION.md](VALIDATION.md). [Measurements](../results/multi-row-attention/summary.json).

## 9. Diagnose the numerical discrepancy — 2026-11-04

Three fresh-process captures per mode, including the retained earlier capture,
were bit-identical within each mode. Layer tracing was neutral to terminal
outputs. The first difference occurred at layer 3 attention, followed by
near-tie crossings in MoE expert routing. Diagnostic replay of baseline router
scores reduced 64K mean KL from 0.60334 to 0.01468 (about 41×), while baseline
replay into itself stayed identical. Replay pins both expert choices and weights.

On identical layer-3 inputs near the end of the 64K prompt, sampled MR attention
was about 36× closer to an independent FP64 reference than the existing kernel.
A separate 64K text check scored 32,767 positions: PPL 5.810504 → 5.806155,
−0.07486%, within the predeclared 1% margin. The apparent improvement is within
uncertainty. No kernel or serving-default change was made.

**Decision: the dominant numerical mechanism is supported; retain the failed
equivalence gate and keep MR experimental.** Full methods, quality checks and
limits are in [the investigation](NUMERICAL-INVESTIGATION.md).

## 10. Tile convolution inputs and skip unused draft heads — 2026-11-04

After reviewing the evidence, the user accepted the observed numerical variance
and requested another 10–20% prefill improvement followed by real OpenClaw use.
This explicitly superseded the earlier requirement to reproduce old-kernel
outputs. The failed equivalence gate and task observations remain in the record.

Two changes build on MR H=2 without changing model weights or KV precision:

1. Patch 0005 copies recurrent convolution inputs through a padded shared-memory
   tile, coalescing transposed activation reads and output writes. Values are
   copied as 32-bit words. Unsupported layouts, types and shapes fall back.
2. Patch 0006 omits the MTP draft vocabulary projection when no logits or generic
   embeddings are requested. Hidden/state computation and actual drafting retain
   their original paths. This complements patch 0002's target-side sparse logits.

The initial component experiment used the same 65,536-token server fixture,
2K warmup, F16 KV, batch/ubatch 2048/1024 and native MTP width 5. MR was on
throughout; prompt reuse, checkpoints and profilers were off. In execution order:

| Configuration | Prefill tok/s | Prefill seconds | Generation tok/s |
|---|---:|---:|---:|
| MR only, first control | 833.70 | 78.61 | 49.05 |
| Tiled concat only | 908.23 | 72.16 | 48.93 |
| Unused draft-head skip only | 863.23 | 75.92 | 52.37 |
| Both, trial 1 | 957.56 | 68.44 | 52.25 |
| Both, trial 2 | 952.95 | 68.77 | 52.52 |
| MR only, second control | 776.01 | 84.45 | 49.03 |

The controls varied materially; there is no established cause. The combined
trials were consistent and about 14.6% faster than the faster initial control.
Component timings are single observations, not independently precise effects.
GPU temperature rose from 41°C before the first run to 63–67°C after runs;
post-run power was approximately 108–110 W. Raw snapshots are retained.

After adding a workgroup-count guard for oversized tensors, the final build
passed the 27 targeted concat cases again. A fresh paired timing on that exact
build measured **821.97 → 955.71 tok/s (+16.27%)**, reducing prefill
**79.73 → 68.57 seconds**. Its short generation check was 49.17 → 52.35 tok/s.
This pair reproduces the useful gain on the binary selected for serving.
There is no new target-only synthetic measurement or sustained 1K claim.

139/139 CONCAT cases passed with the new path enabled; the 27 added cases
also passed disabled. The target MTP hidden/output regression passed. A new
draft regression captured nine required logit/hidden rows plus saved draft KV
state; head-skip off/on were bit-identical through prefill, drafting and restore.
All eight server runs had identical output hashes and draft counts.

**Decision: enable MR and both optimizations on the local trial endpoint for
the user's real prompts.** At 22:07 UTC, health, an actual completion, function
selection/arguments and cached tool-result continuation passed. No OpenClaw
agent turn or gateway restart was initiated. The user then began their own
Telegram request. Practical task quality remains for that test to establish.

The default launcher still provides the stable fallback. The live trial uses
`--optimized-prefill` and source profile `prefill-next`. Both new switches can
be disabled independently for diagnosis. Reconstructable patches, exact build
hashes, test summaries and all timings are retained in
[the evidence](../results/prefill-next/summary.json).

## Next work

1. Observe the user's real OpenClaw prompts, tool loops and cache behavior;
   assess useful answers without requiring old-kernel wording.
2. Measure longer contexts before extrapolating 64K throughput to 128K or 150K.
   The current served context is still 131,072.
3. If further performance work is requested, profile the combined build;
   repeated routing and remaining attention copies are unmeasured candidates.
4. Treat boot persistence and larger served context as separate operational
   changes; the current trial service is transient.
