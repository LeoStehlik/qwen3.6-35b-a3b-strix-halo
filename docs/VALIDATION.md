# Validation, failures and deployment decision

The MR speedup is repeatable on the tested 64K workload. Its deployment decision
is separate: **MR remains disabled in normal serving**. The prior MTP build was
restored, health returned OK and a completion answered `42`. No OpenClaw gateway
restart was part of this experiment or repository work.

## Passed checks

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

## Before promotion

1. Preserve the deterministic tracing/replay evidence and investigate any new
   discrepancy that is not explained by the observed rounding/routing mechanism.
2. Evaluate several text/code samples at 32K and 64K+, plus retrieval, source
   comparison and tool-selection tasks with assessable answers.
3. Test multi-turn tool-result continuation, prefix displacement/restoration
   and speculative rollback with production checkpoints enabled.
4. Repeat paired timing and record workload/power/clock conditions after fixes.
5. Establish quality acceptance criteria before examining new results.

The prepared MR production-serving validation script in the local cache was
**not executed**. The default launcher uses stable code; `--experimental-mr`
is an opt-in switch, not a claim of promotion.

Evidence: [summary](../results/multi-row-attention/summary.json),
[numeric comparison](../results/multi-row-attention/numeric-comparison.json),
[FP32 oracle](../results/multi-row-attention/numeric-oracle-comparison.json),
[first-layer check](../results/multi-row-attention/initial-comparison.json),
[perplexity](../results/multi-row-attention/perplexity-comparison.json).
