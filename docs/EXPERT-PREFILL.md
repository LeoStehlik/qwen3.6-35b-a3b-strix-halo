# Expert-kernel prefill investigation — 2026-11-04

Status: both selected changes promoted to the existing endpoint at 23:21:56 UTC.
Repeated timings and direct state/tool/cache checks passed. Ordinary OpenClaw
answer quality is pending user acceptance; no further performance sweep is planned.

## Measurement and rejected hypotheses

The existing model, F16 KV, 2048/1024 batch/ubatch, MR attention and MTP remain
unchanged. The accepted baseline already includes tiled concat and unused draft
head elimination. Measurements here are server/MTP or explicitly isolated
operators; they are not the earlier synthetic target-only track.

A fresh instrumented 64K run put expert matrix work at 20.48 seconds.
Counting experts and building row lists together cost about 0.27 seconds:
reusing those lists has a ceiling below 0.4% of total prefill, so that proposal
was discarded. The active path is cooperative-matrix FP4-to-F16 staging with F32
activations, not the integer Q8-activation matrix kernel.

Actual counts were captured after each expert matrix call, excluding warmup and
decode: 7,872 calls, including target and draft. Each projection/ubatch is a new
expert instance; these percentages are not percentages of unique model experts.

| Rows per expert | Share of expert instances |
|---|---:|
| 0 | 12.09% |
| 1–8 | 28.13% |
| 9–16 | 14.53% |
| 17–32 | 17.10% |
| 33–64 | 14.72% |
| 65–128 | 8.72% |
| 129–256 | 3.55% |
| 257–1024 | 1.15% |

Among active instances, 67.98% have at most 32 rows. The existing 64-row tiles
are 44.99% filled; 32-row tiles would be 65.02% filled. These are logical matrix
utilization figures, not measured hardware occupancy. No actual DRAM bandwidth
or runtime occupancy counters were collected.

## Controlled kernel experiments

The 64×32, 128-thread variant reduced isolated call time by roughly 24–33% at
8–32 rows. At 48/64/128 rows it lost to the existing 64×64 kernel. At 96 rows
it was slightly faster, but this did not justify selecting it everywhere.
The primary policy therefore uses the narrow kernel at ≤32 rows and the old
kernel above that. The unmeasured 33–47 interval uses the conservative old path.

Compiler statistics for the same pipeline specialization show 144→108 vector
registers and 12→10 KiB shared memory. The compiler's subgroup-residency ceiling
increases 10→12 per SIMD. Neither shader spills registers. These are compiler
properties, not a measurement of achieved occupancy during the model run.

Arithmetic scale decoding, arithmetic codebook decoding and both together
were slower. K=64 and K=128 tiles were also slower. None is selected for serving.
Deeper K passed operator tolerance but was rejected before model-level drift
validation; it must not be described as model-equivalent.

## Correctness

An exhaustive direct-value test covers all 256 packed bytes × all 256 scale
bytes, both nibble positions, and aligned K=64/unaligned K=32 paths. Each of four
arithmetic modes passed 1,048,576 output checks with zero mismatches, including
reserved scale bytes. The initial CPU matmul comparison was unsuitable for an
exact decode oracle because it requantizes one-hot activations to Q8. The direct
scalar oracle avoids that extra rounding; its zero-error requirement was kept.

The primary, tail and compact-only dispatch variants each passed 26 row-boundary
cases and 12 balanced/skewed cases against the CPU reference. Additional tests
explicitly use 1/32/48/64-row experts, zero-row gaps, non-contiguous active IDs,
and first/final expert IDs. Four CPU-reference layouts and 334,080 exact ownership
checks passed per mode, including compact off. Every output was poisoned with
NaN before the exact check; output hashes are identical across modes.

The first compact implementation incorrectly inferred expert count from grid.z
after changing dispatch geometry. Operator tests caught it before any model run.
Compact mode now carries the logical expert count explicitly; the original
dispatch retains its original interpretation. The failed evidence is retained.

## Independent controls

Compact dispatch lists only live expert/tile pairs. It is independently
switchable from tile selection. A second experimental selector uses narrow tiles
only for ≤32-row tails while preserving full 64-row tiles; it is measured
separately from the primary per-expert policy.

Convolution fusion is independently switchable. Initial 40-case tests passed
with model-layout weights in both modes. The first model trial was vetoed by
allocator overlap, despite a stale profiler fusion label. That reporting bug
was corrected. The revised candidate uses the elided concat allocation as scratch
when needed and copies after completion, with a dispatch barrier. It retains
the original path when eligibility or safe scratch conditions do not hold.
The convolution result below is measured independently before combination.

## Repeated 64K results and promotion

These are cold 65,536-token server/MTP requests with a 2K warmup, 128 generated
tokens, F16 KV, batch/ubatch 2048/1024 and all previous accepted optimizations.
No profiler, reused prompt or production checkpoints participate in timing.

| Configuration | Two prefill samples (tok/s) | Mean gain over control |
|---|---:|---:|
| Previous accepted baseline | 958.91 / 954.49 | — |
| Primary expert selector | 1,012.94 / 1,014.01 | +5.93% |
| Fused convolution only | 986.28 / 988.37 | +3.20% |
| Both selected changes | **1,045.40 / 1,048.68** | **+9.44%** |

Mean prefill time falls from 68.50 to 62.59 seconds, saving 5.91 seconds.
Independent component elapsed-time savings predict 62.54 seconds, about
1,048 tok/s; their combination is consistent with that estimate. Mean generation
is essentially unchanged: 52.33 versus 52.45 tok/s. All ten uninstrumented runs,
including the controls below, have identical output hashes and 63 drafted / 58
accepted tokens. This is evidence for this fixed fixture, not all model tasks.

Compact-only dispatch measured 954.66 tok/s, effectively flat. It supports
selective execution but is not independently a demonstrated gain. The tail-only
selector measured 1,018.04 tok/s once, only about 0.4% above the primary policy;
that difference is inconclusive and does not justify promoting another policy.
Arithmetic decoding and deeper K remain disabled. The selected K=32 path keeps
the existing accumulation precision and order.

The saved combined diagnostic profile confirms actual convolution fusion and
both expert dispatches. Expert matrix time fell from about 20.48 to 17.35 seconds
(target plus draft). The fused target convolution chain costs 1.25 seconds.
These instrumented timings explain the result; they are excluded from headline
throughput. Readback/profiling overhead reduced the diagnostic run to 999.50
tok/s, which must not be averaged into the uninstrumented samples.

The combined `test-mtp-hidden` regression passed seven cases, including two
physical prompt ubatches, partial tail, verification, checkpoint restore,
multiple sequences and interleaved hidden-row order. Required hidden and
ordinary-versus-MTP sparse-logit comparisons have zero error. This checks state
contracts within the optimized backend, not full off/on model equivalence.
The separate all-output-versus-sparse-kernel diagnostic is retained in the
[state evidence](../results/prefill-stage2/state-check.json).

On the live production configuration, an 11,243-token tool request produced
exactly `get_validation_record({"key":"prefill-ready"})`. Its tool-result
continuation returned `42`, reusing 11,239 tokens. An unrelated request displaced
the session; replay then restored 11,310 cached tokens and again answered `42`.
Server logs confirm RAM state loading and recurrent checkpoint restoration.
The first readiness harness used an incorrect log phrase and unnecessarily
restored the previous listener; after correcting the assertion to the actual
server message, promotion and checks passed. No model/kernel fix was needed.
This functional check is not a 64K displaced-cache quality evaluation.

## First real OpenClaw requests

At 23:26:41 UTC, the promoted pair had completed a 73,093-token request at
1,012.02 tok/s and a later full 79,548-token pass at 993.07 tok/s. Follow-ups
restored checkpoints at 72,065 and 73,325 tokens. Production cache/checkpoints
were enabled; generation ranged 49.26–62.73 tok/s on these requests. No server
errors were observed, and further traffic remained active. These are aggregate
user-workload observations, not controlled comparisons or a quality assessment.
The user has not yet supplied the new pair's answer-quality verdict.

## Reproduce the selected configuration

Prepare/build source profile `prefill-stage2` (patches 0001–0007), then use
`scripts/serve.py --expert-prefill`. It retains the model, precision, context,
batch settings and all earlier optimizations. Added switches:

```text
GGML_VK_MMID_COMPACT=1
GGML_VK_MMID_SELECTOR=1
GGML_VK_SSM_CONV_FUSED=1
GGML_VK_MMID_FAST_TILE=0
GGML_VK_MMID_FAST_DEQUANT=0
GGML_VK_MMID_FAST_BK=32
```

`--disable-expert-selector` and `--disable-conv-fusion` control the two additions
independently. Source defaults remain off; the stable and `prefill-next` profiles
remain reproducible. `GGML_VK_MMID_INDIRECT=0` refers to the older maximum-row-grid
experiment; compact tile-list dispatch uses its own indirect commands.

For future reproduction only, see [benchmark commands](BENCHMARKS.md).
The experiment is complete; do not interrupt real traffic to repeat it unchanged.
[Structured results](../results/prefill-stage2/summary.json),
[correctness](../results/prefill-stage2/correctness.json),
[live checks](../results/prefill-stage2/live-readiness.json).
