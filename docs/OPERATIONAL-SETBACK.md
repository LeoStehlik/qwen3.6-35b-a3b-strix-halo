# Operational acceptance withdrawn — 2026-11-05

The user has retired Qwen3.6-35B-A3B from daily OpenClaw use and authorized a
return to Qwen3.8-27B dense + DFlash2. Keep this model, source, builds and frozen
performance tag for offline research. This is an operational decision based on
failed real tasks, not proof that the base model cannot perform them.

## What failed

1. **A real Discord turn exhausted its output budget without an answer.**
   The recorded request reused 32,768 tokens and processed 5,680 new tokens at
   1,003.16 tok/s. Generation reached 32,768 tokens at 86.04 tok/s and ended with
   `stopReason=length`. Its only assistant output was a thinking part: 532
   nonempty lines but only 15 distinct lines, with one line repeated 260 times.
   There was no final-answer part. Nine tool actions had already occurred;
   OpenClaw correctly warned that actions might have happened despite the failed
   response. Fast prefill and decode did not produce useful completion.
2. **After a session reset, it failed to recover accessible channel history.**
   Qwen tried session-history tools, including invalid arguments, and treated an
   empty/reset session record as evidence that earlier channel messages were
   unavailable. It asked the user to paste them. The messages still existed in
   Discord. On the subsequent model/runtime switch, the other agent retrieved
   them with `openclaw message read --channel discord ...` through shell access.
   Qwen also had shell access: this was not an established access restriction.
3. **The recovery discussion added unsupported explanations.**
   In the user-supplied follow-up transcript, Qwen attributed the other agent's
   successful retrieval to itself, alternated between incompatible diagnoses,
   and asserted causes without testing them. Repeated acknowledgments did not
   resolve the task or establish why it failed.

The first two observations come from read-only inspection of local inference
and OpenClaw records. The third comes from the conversation supplied by the
user. Only aggregate measurements and a paraphrased account are published;
private messages, thinking text, identifiers and tool payloads remain local.
[Sanitized incident measurements](../results/operational-setback/summary.json).

## What the evidence does and does not establish

- The failed generation's input was approximately 38,448 tokens, below the
  131,072-token served context. Earlier context pressure and successful
  compaction were separate events; do not label this an input-overflow failure.
- The history comparison changed both model and surrounding runtime: Qwen used
  the embedded OpenClaw loop; Astra used a Codex thread. It was not a controlled
  model-only A/B. Both paths could execute shell commands.
- No controlled replay has separated model capability, tool presentation,
  instructions, sampling, template, quantization, MTP or custom-kernel effects.
  None of those is a demonstrated root cause. Temperature can affect behavior;
  no temperature experiment was performed here.
- This is a 35B-total / approximately 3B-active MoE, not a plain 3B model.
  Active-parameter count alone does not diagnose either failure.
- The user need not prove an intrinsic model limit before rejecting a deployment
  that hands achievable work back to them. They chose not to pursue task-specific
  prompt coaching or a sampling sweep as the next project.

## What remains valid

The frozen `prefill-stage2-2026-11-04` tag and all kernel measurements remain
unchanged. Cold 64K server/MTP throughput was 1,045.40 and 1,048.68 tok/s; the
selected pair added 9.44% over its paired control. All ten Stage 2 outputs and
draft counts matched **within that comparison**. Earlier MR attention changed
numerical outputs; do not claim bit-identical output across the entire original
450–500 to approximately 1,047 tok/s optimization history.

Operator, state, exact tool-argument and bounded cache-restoration checks passed
the scenarios documented in [validation](VALIDATION.md). Those checks did not
cover enough autonomous, multi-step real tasks to certify agent reliability.
The later failure withdraws daily-use acceptance; it does not erase performance
results or demonstrate that a particular patch caused the failure.

## Deployment and next work

At publication of this finding, retiring the idle Qwen3.6 listener and disabling
its boot unit are authorized; the service handover is still pending. The
[handoff](../handoff.md) records the verified state after that operation. Preserve
weights and reproducible builds; do not delete them or move the frozen tag.
Repository launch defaults continue to reconstruct Stage 2 for research, not
as a recommendation to deploy it as the user's agent.

Restore the existing Qwen3.8 Q4 + DFlash2 stack with its Q8 KV and full configured
context. Verify GPU residency and an actual completion, not only router health.
Its earlier one-run-per-setting cold 64K comparison favored batch/ubatch
2048/2048 (367.33 tok/s) over 2048/1024 (354.59 tok/s); that is a starting record,
not a fresh repeated baseline or evidence for copying the MoE tuning.

Profile the dense model before choosing kernel work. Raw prefill acceleration
and stable-prefix reuse are complementary. Reuse requires exact prefix tokens,
compatible model/runtime/settings, and complete hybrid/recurrent state. Existing
checkpoint support does not prove that OpenClaw assembles a reusable stable
prefix. Measure that independently. Judge progress by time to a correct useful
action on cold and warm real requests, including tool discovery, exact arguments,
continuation and recovery; retain model capability and useful context.
