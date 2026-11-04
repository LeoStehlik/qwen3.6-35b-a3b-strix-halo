# Explaining the 64K numerical discrepancy — 2026-11-04

The dominant mechanism is now supported by a controlled experiment: small
attention-rounding differences alter near-tied MoE routing scores, and those
changes amplify through the model. This is a deterministic difference between
implementations on the tested input. It is not evidence that the faster kernel
has worse attention accuracy: its sampled long-context output was substantially
closer to an independent reference.

**The original numerical-equivalence gate still fails. MR remains disabled in
normal serving.** This investigation does not change model weights, runtime
kernels, serving defaults, or the gate's thresholds. In particular, diagnostic
router replay is not a proposed inference algorithm.

## 1. Establish repeatability and trustworthy instrumentation

Two new fresh-process 65,536-token captures per mode matched the earlier capture
bit for bit: all three saved logit arrays and hidden arrays agree within each
mode, including across the host reboot. The between-mode difference reproduced.
These target-model captures do not use MTP or server prefix reuse, so neither is
necessary to produce the discrepancy.

A standalone evaluation callback then recorded layer outputs, attention outputs,
attention residuals and MoE router scores. It samples nine activation rows per
1,024-token ubatch and saves every router score. At both 2K and 64K, tracing
preserved the uninstrumented terminal logits and hidden states bit for bit in
each mode. This controls for the callback's changes to graph execution.

Evidence: [repeatability](../results/numerical-investigation/repeatability-summary.json),
[64K trace control](../results/numerical-investigation/trace-neutrality-64k.json).

## 2. Locate the first difference

Layers 0–2, including their recurrent computation, remain identical across every
ubatch. The first difference is the first full-attention layer, layer 3. Its
first-ubatch sampled attention NRMSE is 0.000526; the corresponding router-score
NRMSE is about 0.000153.

The top-eight expert sets inferred from those scores change at only three of
the first 1,024 token positions: 64, 450 and 925. Each swaps the old eighth and
ninth ranked experts. Their original score margins are 0.000910, 0.000224 and
0.000130, respectively, smaller than the rounding-induced score changes.
Selected sets are inferred from scores, not independently captured expert-ID
tensors; these first swaps have distinct scores, not ties.

Differences propagate into later routing and activations. In the first ubatch,
the number of changed expert sets rises from 3 at layer 3 to 19 at layer 4,
57 at layer 7 and 336 at layer 39. These are routing differences, not a count
of incorrect answers.

Evidence: [layer summary](../results/numerical-investigation/layer-routing-summary.json),
[first score crossings](../results/numerical-investigation/first-routing-changes.json).

## 3. Test the mechanism by replaying baseline router scores

During a separate diagnostic run, replace every MR router-score tensor with its
captured baseline counterpart. This fixes **both expert choices and their mixing
weights**. It does not isolate discrete selection alone. All 2,560 expected
router tensors were replayed in each 64K run. Replaying the baseline into itself
leaves terminal logits and hidden states bit-identical, providing a control.

| 64K comparison against baseline | Ordinary MR | MR with baseline router scores |
|---|---:|---:|
| Mean KL, 128 sampled distributions | 0.603343 | 0.014677 |
| Hidden-state NRMSE | 0.351134 | 0.072415 |
| Greedy top-token matches | 108/128 | 124/128 |

Fixing router scores reduces mean KL about **41×** and hidden NRMSE about
**4.85×**. At 2K, the corresponding reductions are about 444× and 7.09×.
This establishes routing as a major amplifier. It does not remove all drift or
prove every residual numerical difference benign. The replay still does not
meet all original equivalence thresholds.

Evidence: [64K replay](../results/numerical-investigation/router-replay-64k-comparison.json),
[2K replay](../results/numerical-investigation/router-replay-comparison.json).

## 4. Check long-context attention against an independent reference

Capture layer 3 in the final 1,024-query ubatch of the 64K prompt. Both modes
have bit-identical Q, K, V and mask inputs there. Compute softmax and the weighted
value sum in FP64 using the kernel's actual scaled-F16 query and stored F16 K/V.
This checks attention on its quantized inputs, not an unquantized whole model.

For 25 selected query/head pairs near the end of the context:

| Implementation | NRMSE against FP64 reference |
|---|---:|
| Existing attention | 0.00697791 |
| MR attention | 0.000193534 |

MR is about **36× closer** on this sample. The earlier first-ubatch check also
favored MR, about 5×. The old kernel's F16 output accumulation versus MR's F32
accumulation is a relevant precision difference. A prior F32-output variant of
the old kernel still differed from MR at model level, so changing that one
accumulator does not reproduce MR's complete arithmetic order.

These observations argue against treating the old kernel's exact output as a
quality oracle. They are not proof of correctness at every layer and shape.
The existing 30 operator cases remain separate supporting evidence.

Evidence: [long-context reference](../results/numerical-investigation/deep-attention-comparison.json).

## 5. Measure next-token quality at 64K

Use the first 65,536-token chunk of the WikiText-2 raw test split and score the
last 32,767 next-token positions. The input, weights and settings are paired;
no router replay or reference-precision switches are enabled. The 1% PPL margin
was recorded before the run. The dataset is identified by SHA256 and is not
redistributed here.

| Baseline PPL | MR PPL | Relative change |
|---:|---:|---:|
| 5.810504 | 5.806155 | −0.07486% |

Mean NLL change is −0.0007488. The approximate 95% interval using 128-token
blocks is [−0.0017567, +0.0002590]. This passes the specified 1% margin on this
sample; the small apparent improvement is within uncertainty. Blocks from one
text chunk are not independent documents, and this is not a broad model-quality
or agent-capability guarantee.

Evidence: [predeclared protocol](../results/numerical-investigation/long-perplexity-protocol.json),
[paired result](../results/numerical-investigation/perplexity-comparison.json).

## 6. Exercise a long source comparison and tool continuation

A synthetic source bundle contains 65,768 context tokens, with audit facts at
the beginning, middle and end. The expected answers were fixed before either
mode ran. Disposable servers used the production MTP, F16 KV, batch/ubatch,
prompt cache and checkpoint settings; decoding requested temperature 0 and
thinking disabled. This is one smoke scenario, not an agent benchmark.

| Check | Baseline | MR |
|---|---|---|
| Identify only the credited, merged PRs present in the release | **Failed** | **Failed** |
| Select the lookup tool and exact repository/ref from the middle | Passed | Passed |
| Combine artifact/code from tool result with nonce from the context | Passed | Passed |
| Reuse prefix after displacement by an unrelated request | 65,880 tokens | 65,880 tokens |

The expected PR set was `[21481, 21577]`. Baseline added `21602`, credited to
someone else. MR's cold answer added `21601`, whose commit was outside the
release. After restoring the cached prefix, MR instead added `21602`; baseline
retained its original answer. Both restored comparisons therefore also failed
the expected-answer check. No output was truncated.

**Cache reuse is not a passed output-consistency test.** MR's cold/restored
answer difference is an additional serving observation, not evidence by itself
of serialized-state corruption. Both restored requests evaluated a four-token
tail. The tests need to distinguish saved state, MTP/verification, batch-shape
numerics and sampling before attributing the cause. The earlier prepared
production-serving script was not run; these are new, separately recorded tests.

The harness initially hit TCP port reuse between modes after its baseline server
exited. Its free-port probe was corrected to allow TIME_WAIT reuse while still
rejecting a live listener; the MR half resumed with the original frozen input
and protocol. The failed quality answers were retained.

Evidence: [task protocol](../results/numerical-investigation/task-protocol.json),
[responses and checks](../results/numerical-investigation/task-comparison.json),
[cold/restored observation](../results/numerical-investigation/task-observations.json).

### MTP-off follow-up control

Repeat the exact context/question in each mode with speculative decoding set to
`none`, retaining production cache/checkpoints and greedy settings. Compare the
cold answer, restoration after displacement, and another restored repeat.
Both modes returned identical text in all three requests: the same wrong set
`[21481, 21577, 21602]`. Each restored request reused 65,880 tokens and evaluated
four tokens. The non-MTP control therefore did not reproduce the MR answer change.

This narrows the observation to an interaction involving MTP or its execution
pattern in this scenario. Disabling MTP can change graph/batch behavior, so it
does not prove a particular checkpoint-serialization defect or establish hidden
state equality. The next diagnostic should teacher-force the same continuation
and compare target logits/hidden states with identical token batches, then
separate draft state, speculative verification and batching. Neither MTP nor
checkpoints were disabled on the live endpoint.

Evidence: [follow-up protocol](../results/numerical-investigation/cache-control-protocol.json),
[control responses](../results/numerical-investigation/cache-control-comparison.json).

## Reproduction and limits

The standalone trace source and analysis scripts are under
[`benchmarks/diagnostics`](../benchmarks/diagnostics/README.md). They link to the
existing MR build; no further kernel patch was made for this investigation.
Raw tensors stay local. Expect several GiB of captures per mode.

The original failed gate is preserved as a numerical-equivalence test. The new
evidence explains why its failure alone does not establish quality loss. Before
promotion, expand independent quality coverage across text/code, long-context
retrieval and tool tasks, and validate real multi-turn workloads. Do not change
the gate after seeing a favorable speed result or declare broad equivalence
from one perplexity sample. This investigation adds no new throughput claim.
