# Validation, failures and deployment decision

As of 2026-11-04 23:54 UTC, the user has accepted the selected expert-row kernel
and fused convolution as the persistent/default baseline, on top of the earlier
MR/tiled-concat/unused-head profile. Direct correctness, state, tool and cache
checks passed. Production requests continue. This is deployment acceptance,
not proof of equivalent quality on every task. No OpenClaw gateway restart was
part of this work.

The enabled boot unit matches the live binary, full argv and selected flags;
systemd validation and lingering checks passed without changing the live PID.
No reboot test was performed. Combined-build profiling finished with unchanged
inference binaries; its parser reproduces the saved aggregate analysis.
Hardware traffic counters are uncalibrated and do not establish DRAM saturation.
See [combined profile](COMBINED-PROFILE.md) and
[persistence evidence](../results/combined-profile/persistence.json).

## Expert selection and convolution fusion

- Row-count policy: ≤32 rows use 64×32/128 threads; larger experts retain 64×64.
  Each compact/selector variant passed 26 row-boundary and 12 routing cases.
  The explicit 1/32/48/64-row boundary test passed four CPU-reference layouts
  and 334,080 exact ownership checks per mode, including compact off.
- Exhaustive FP4 decode testing passed 1,048,576 exact checks per arithmetic
  mode, all four modes, including every scale byte. Arithmetic decoding was
  slower and is not selected; K64/K128 were also rejected before model tests.
- Convolution fusion passed 40 CPU-reference cases both off and on. The final
  allocator-overlap handling is additionally exercised by full model and state
  checks. Failed initial fixture/oracle assumptions remain documented.
- Seven `test-mtp-hidden` cases passed with both changes enabled, including
  checkpoint restoration, multiple sequences and speculative verification.
  Required hidden and ordinary-versus-MTP sparse logits have zero error.
- Two combined runs measured 1,045.40 and 1,048.68 tok/s, versus 958.91 and
  954.49 controls: +9.44% mean throughput, 5.91 seconds saved at 64K.
  All ten uninstrumented runs have identical output hashes and draft counts.
- Live exact tool name/arguments and tool-result continuation passed on an
  11,243-token prompt. After displacement by an unrelated request, 11,310 tokens
  restored from RAM and the continuation again returned the correct `42`.
  Logs confirm restored recurrent checkpoints, including token 11,310.
  This establishes the tested 11K scenario, not general 64K cache/task quality.

[Full experiment](EXPERT-PREFILL.md),
[correctness evidence](../results/prefill-stage2/correctness.json),
[state checks](../results/prefill-stage2/state-check.json),
[endpoint readiness](../results/prefill-stage2/live-readiness.json).

The sections below retain earlier checks and their limits. User acceptance of
MR variance remains in force; exact old-kernel wording is not a task-quality gate.

## Previous tiled-concat and unused-head checks — 22:07 UTC

- 139/139 CONCAT CPU-reference cases passed with the tiled path enabled.
  The 27 added cases also passed with it disabled, with exact zero error.
  They cover odd extents, padded views, nonzero offsets, outer dimensions,
  the 8192-channel model shape, and type/shape fallbacks. After adding a
  workgroup-count guard, all 27 passed again on the final build.
- `test-mtp-hidden` passed with MR and both optimizations enabled, checking
  hidden rows, selected logits, verification, multiple sequences and restore.
- The new `test-mtp-unused-head` compared head-skip off/on in separate processes:
  two physical prompt ubatches, a short zero-output tail, draft steps, state
  restoration and a multi-output batch. All nine requested vocabulary rows,
  hidden rows and the saved draft KV state were bit-identical. This isolates
  the new head elimination; it does not require old and MR attention to agree.
- All eight measured server runs produced identical output hashes and the same
  draft counts (63 drafted, 58 accepted). Combined prefill was 953–958 tok/s.
  The final-build pair measured 821.97 → 955.71 tok/s (+16.27%).
- The live endpoint passed health, an actual `42` completion, exact function
  selection/arguments and a tool-result continuation answering `42`, with
  2,743 cached tokens. Production cache/checkpoint settings were enabled.
  The log confirms a small context checkpoint restored at token 2,743.
  This does not test long-context displacement/restoration or complete
  real-agent quality; the user supplies the real OpenClaw test.

[Evidence](../results/prefill-next/summary.json),
[draft-state comparison](../results/prefill-next/draft-head-comparison.json),
[endpoint readiness](../results/prefill-next/live-readiness.json).

The sections below retain the earlier attention investigation and its limits.

## Earlier attention checks

- 30 attention cases per mode against the CPU F32 oracle: Qwen dimensions,
  q=63/64/65/75/1024, KV up to 65,536, interleaved/padded views, masks, sinks,
  ALiBi and softcap. Some deliberately exercise fallback.
- MTP regression: hidden capture, selected logits, speculative verification,
  multiple sequences and checkpoint restore within the candidate path. This
  does not prove equality between old and new attention kernels.
- Repeat server output was identical **within each mode**. Baseline and MR
  output differed from one another.
- One WikiText excerpt at 12,288 context, 6,143 scored tokens: PPL 4.5824288
  versus 4.5837880, +0.02966%. Approximate 95% paired NLL interval: −0.001901 to
  +0.002494. This met the recorded 1% perplexity margin on that sample, not a
  general capability-equivalence criterion.

## CPU reference correction

Initially two 65K cases failed the original CPU oracle for **both** Vulkan
kernels, error around 0.00134 versus tolerance 0.0005. The CPU F16 path
accumulated values in F16 despite requested F32 precision. A diagnostic F32
accumulator fixed both comparisons: 30/30 passed without relaxing tolerance.
Default CPU behavior is unchanged unless `GGML_CPU_FA_F32_REFERENCE` is set.

## Failed full-model gate

The test sampled 128 next-token distributions across the separate 65,536-token
code prompt. Predefined thresholds: mean KL <0.001, maximum logit NRMSE <0.02,
greedy agreement ≥95%, hidden-state NRMSE <0.02.

| Metric | Baseline vs MR |
|---|---:|
| Mean KL | 0.60334 |
| Maximum logit NRMSE | about 0.9148 |
| Greedy agreement | 108/128 = 84.375% |
| Hidden-state NRMSE | about 0.3511 |

The gate failed. These are internal probabilities on one code prompt, not 128
independent tasks or a task-failure rate. Different output does not establish
worse quality; faster inference does not establish preserved quality either.

## Investigation

First-layer Q/K/V/mask inputs were bit-identical. MR versus old attention output
NRMSE was 0.000588. On selected rows/heads, an independent FP64 softmax/value
calculation found MR closer to the oracle: roughly 0.0000949 versus 0.000488.
This describes the captured first layer, not all layers or a 64K guarantee.

The old coopmat1 kernel used an F16 output accumulator; MR uses F32. A
diagnostic F32-output variant of the old kernel passed 30/30 operator tests,
but MR still differed at model level (mean KL 0.64754, 110/128 greedy matches).
Disabling the three MR scheduling options produced identical captured
first-layer output to normal MR.

The follow-up [numerical investigation](NUMERICAL-INVESTIGATION.md) now supports
MoE routing as a major amplifier of deterministic attention-rounding differences.
Three captures per mode were bit-identical. Tracing preserved terminal outputs;
layers 0–2 stayed identical and the first difference appeared at full-attention
layer 3. Small score changes cross near-ties in its top-eight expert selection.

Replaying all baseline router scores into MR, a diagnostic that pins both
choices and weights, reduced 64K mean KL from 0.60334 to 0.01468. Baseline
replay into itself was bit-identical. Residual drift remains; this intervention
is not a runtime fix and does not pass every original equivalence threshold.

At the last ubatch of the 64K prompt, MR was about 36× closer to an independent
FP64 reference on 25 selected query/head pairs in layer 3. On a separate 64K
WikiText chunk, PPL changed from 5.810504 to 5.806155 (−0.07486%), meeting the
predeclared 1% margin across 32,767 scored positions. This explains much of the
discrepancy without establishing broad model-quality equivalence.

**The numerical gate remains failed and unchanged.** The new evidence makes
"different from the old kernel" an inadequate stand-alone quality verdict.
Neither that observation nor the speedup is sufficient for promotion.

A new 65K+ synthetic source/tool scenario passed tool selection, exact argument
retrieval and tool-result continuation in both modes. Both failed the source
comparison's expected PR set. Both reused 65,880 cached tokens after displacement,
but MR's wrong answer changed after restoration while baseline's stayed the same.
This is an unresolved output-consistency observation; successful token reuse
must not be reported as successful end-to-end cache validation. See the
[recorded task results](../results/numerical-investigation/task-comparison.json).
With MTP disabled, both modes returned identical cold/restored/repeated text on
this scenario. The remaining interaction involves MTP or its execution pattern;
that control does not identify a serialization bug or establish state equality.

## Updated acceptance direction

The user subsequently accepted the measured numerical variance as sufficient
to resume performance work and then test through OpenClaw. The failed exact
numerical-equivalence result remains recorded; it is not by itself a task-quality
failure. New changes must retain operator correctness and required state/output
behavior. Judge practical adoption with assessable tasks, valid tool execution,
cache/continuation behavior and performance, allowing different valid answers.
Do not repeat open-ended equivalence forensics without evidence of malfunction.

## Earlier proposed checks (historical)

1. Preserve the deterministic tracing/replay evidence and investigate any new
   discrepancy that is not explained by the observed rounding/routing mechanism.
2. Evaluate several text/code samples at 32K and 64K+, plus retrieval, source
   comparison and tool-selection tasks with assessable answers.
3. Test multi-turn tool-result continuation, prefix displacement/restoration
   and speculative rollback with production checkpoints enabled.
4. Repeat paired timing and record workload/power/clock conditions after fixes.
5. Establish quality acceptance criteria before examining new results.

The earlier MR production-serving validation script in the local cache was
**not executed**. The follow-up endpoint checks described above are a separate,
smaller API/tool/prefix-reuse check. The launcher without an explicit mode still
uses stable code; the live trial uses `--optimized-prefill`.

Evidence: [summary](../results/multi-row-attention/summary.json),
[numeric comparison](../results/multi-row-attention/numeric-comparison.json),
[FP32 oracle](../results/multi-row-attention/numeric-oracle-comparison.json),
[first-layer check](../results/multi-row-attention/initial-comparison.json),
[perplexity](../results/multi-row-attention/perplexity-comparison.json).
