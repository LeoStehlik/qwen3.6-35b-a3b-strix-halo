# Handoff for Francis

Updated 2026-11-04 UTC after the follow-up prefill optimizations and endpoint
readiness checks at 22:07. The user's first Telegram request completed at 22:11:58.
This is a dated snapshot; verify process state before acting.

## Goal and constraints

The priority is long-context **prefill**, targeting sustained approximately
1,000 tok/s while keeping the existing Qwen3.6 model and OpenClaw capabilities.
Generation around 50–100 tok/s depending on workload/context was already useful.
Do not substitute a weaker model, reduce the tool inventory, impose an 8K output
cap, or present short-prompt throughput as a large-context result.

Do not restart the Mac mini's OpenClaw gateway or reboot that Mac. This project
runs on the Linux Strix Halo inference host. The user authorized local inference
interruptions for these experiments, and the optimized endpoint now serves
their real requests. Do not interrupt it for documentation. Do not send channel
messages or launch OpenClaw agent turns: the user supplies real prompts.

The user reviewed the numerical evidence and accepted the observed MR variance
as sufficient to continue optimization and practical testing. Exact old-kernel
answers are not the quality criterion. Preserve earlier failures as evidence;
use operator correctness, required state/output behavior and useful task results.
Do not reopen equivalence forensics without evidence of malfunction.

## Current serving state

| Item | Observed value |
|---|---|
| OpenAI-compatible base URL | `http://100.96.8.54:8082/v1` |
| Model ID | `tiercel-qwen36-a3b-fast` |
| Existing OpenClaw provider/model | `tiercelmoe/tiercel-qwen36-a3b-fast` |
| Active unit | `qwen36-prefill-test.service` |
| Unit persistence | **Transient**, under `/run/user/1000/systemd/transient/` |
| Actual binary | `$HOME/src/rocmfp4-prefill-next/build-vulkan/bin/llama-server` |
| Launch helper | `scripts/serve.py --optimized-prefill --binary <above> --host 100.96.8.54` |
| Enabled | Foundation, sparse target logits, MR H=2, tiled concat, unused draft-head skip |
| Disabled | Indirect MoE dispatch, smaller MoE tiles, all profiler/oracle/trace diagnostics |
| KV / batch / ubatch / slots | F16 / 2048 / 1024 / 1 |
| Served context | 131,072; 150K initial prompts do not fit this configuration |
| Prompt cache / checkpoints | 24,576 MiB; idle slots; 32 checkpoints every 8,192 tokens |
| Readiness | Health OK, actual completion `42`, exact tool arguments and result continuation `42` |
| Reuse check | 2,743 cached tokens; log confirms a small context checkpoint restored |

`qwen36-rocmfpx.service` remains **inactive but enabled for boot**. It is the
older persistent unit, not the unit currently serving. The transient replacement
will not automatically become a persistent boot configuration. Making the
selected configuration persistent is separate pending operational work.

`qwen38-dflash2.service` remains active. Its router at loopback 8081 had
previously autoloaded a Q4 target on CPU while GPU permissions were unavailable.
That CPU target was idle during the new timing runs. It was left alone.
Verify residency and other work before any future controlled benchmark.

Earlier in this investigation the account lacked render-node permissions after
a host reboot. Membership in `render` and `video` was added, and a user ACL on
the current render node restored access without another reboot or logout.
The older failing boot unit was stopped and the stable transient baseline
restored. It was subsequently replaced by the tested optimized trial below.
The Mac/OpenClaw was not restarted.

## Latest result

The first real Telegram request completed at 22:11:58: **50,437 prompt tokens
in 49.01 s (1,029.18 tok/s)** and **4,408 generated tokens at 70.20 tok/s**.
Draft acceptance was 94.402%. A separate 4,049-token request completed at
22:12:26, at 1,322.15 tok/s prefill and 96.30 tok/s generation. These are
user-initiated workload observations, not paired benchmarks or an assessment
of answer quality. Only aggregate timings are retained in GitHub.

MR stayed enabled in both modes. The final binary's paired 65,536-token cold
server/MTP request measured:

| Metric | MR only | MR + new changes |
|---|---:|---:|
| Prefill | 821.97 tok/s | **955.71 tok/s (+16.27%)** |
| Prefill time | 79.73 s | **68.57 s (11.16 s saved)** |
| Short generation check | 49.17 tok/s | **52.35 tok/s** |

Two earlier combined trials measured 957.56 and 952.95 tok/s. Their controls
varied from 776.01 to 833.70 tok/s; the cause is not established. Do not select
the slowest control to inflate the gain. The final pair uses the exact served
build after a workgroup-count guard was added and tested. All eight server
runs produced identical output hashes and draft counts.

Patch 0005 tiles recurrent convolution-input copies through shared memory,
coalescing large transposed reads and output writes without changing value bits.
Patch 0006 omits the MTP draft vocabulary projection when no logits or generic
embeddings are requested, retaining hidden/state computation and actual drafting.
Both can be disabled independently; source defaults are off.

139/139 CONCAT CPU-reference cases passed with the new path enabled. The 27
added cases passed with it disabled and again enabled on the final build.
The target hidden/output regression passed. The new draft regression found nine
required logit/hidden rows and saved draft KV state bit-identical across head-skip
off/on, including two physical prompt ubatches, a tail, drafting and restore.

See [step 10](docs/EXPERIMENTS.md), [validation](docs/VALIDATION.md), and
[structured evidence](results/prefill-next/summary.json). No new synthetic
benchmark was run. This does not establish sustained 1K throughput at 64K+
or broad practical quality; the user is testing their normal OpenClaw session.

## Earlier attention result and numerical evidence

| Earlier 64K track | MR off | MR on | Gain |
|---|---:|---:|---:|
| Synthetic, full warmup, target only | 740.35 | 919.33 tok/s | +24.17% |
| Cold source prompt, server with MTP | 638.97 | 817.07 tok/s | +27.87% |

These are historical paired attention measurements, two trials per mode.
Do not mix the ~741 synthetic baseline with server/MTP throughput. No 150K
performance has been established.

The old full-model equivalence gate failed (mean KL 0.60334, top-token agreement
108/128). Three captures per mode reproduced exactly. Layer tracing and router
replay support rounding differences amplified by near-tie MoE routing; replay
reduced mean KL about 41×. Sampled layer-3 attention near 64K was 36× closer to
an FP64 reference with MR. A 64K text check found PPL 5.8105 → 5.8062, within the
predeclared margin and uncertainty. This supports the user's acceptance of
variance, without proving all tasks have equivalent quality.

Both attention modes passed an earlier 65K+ tool/continuation scenario and
failed its source-comparison answer. MR's cold/restored wording changed with
MTP on, but not in the MTP-off control. That observation remains in the
[numerical investigation](docs/NUMERICAL-INVESTIGATION.md); it does not by itself
identify a serialization bug or invalidate different valid answers.
The newer small checkpoint/tool check must not be described as complete
long-context displaced-cache validation.

## Source and evidence locations

| Path under `$HOME` | Purpose |
|---|---|
| `qwen3.6-35b-a3b-strix-halo/` | This durable GitHub repository |
| `src/rocmfp4-prefill-next/` | **Currently serving** source/build, profile 0001–0006 |
| `src/rocmfp4-mtp-output/` | Stable fallback, without MR |
| `src/rocmfp4-fa-mr/` | Earlier MR port and numerical investigation |
| `src/rocmfp4-sparse/` | Pinned base, sparse checkout |
| `src/rocmfp4-prefill-experiment/` | Foundation optimizations / indirect experiment |
| `src/rocmfp4-moe-tiles/` | Rejected tile experiment |
| `src/rocmfp4-fa-profile/` | Diagnostic profiling source/build |
| `models/qwen36-rocmfpx/` | Existing weights, projector, template |
| `.cache/prefill-next.hhm3990y/` | New protocol, builds, tests, timing, deploy/check scripts and evidence |
| `.cache/fa-mr.MuLSLc/` | Earlier MR raw logs, captures and results |
| `.cache/fa-divergence.LwnTjA/` | Repeatability, layer traces, router replay and 64K quality evidence |
| `.cache/fa-profile.b5wpyH/` | Prior operator profile |
| `.cache/mtp-output.fVUPF0/` | Sparse target-logit evidence and original restore script |
| `.cache/moe-tiles.5TfjSU/` | Rejected tile evidence |
| `.cache/prefill-indirect.t1ja9L/` | Indirect dispatch evidence |
| `.cache/qwen38-ubatch-compare.fp9Hef/` | Qwen3.8 comparison and original prompt |

Preserve the source trees' uncommitted work. Some older trees are incomplete
checkouts. The new `prefill-next` profile reconstructs the served source from
pinned upstream plus six patches, with checksums. A failed local-clone attempt
was retained under the new evidence directory; official-source reconstruction
succeeded. The first build's WebUI asset step failed; the working build disables
the WebUI, matching the existing API-only runtime.

Large tensors, model files and real agent conversations stay local. GitHub
contains sanitized measurements and synthetic/public fixtures only. Do not
remove raw evidence as part of this optimization task.

## Read-only checks

```bash
systemctl --user show qwen36-prefill-test.service \
  -p ActiveState -p MainPID -p FragmentPath -p ExecStart
systemctl --user show qwen36-rocmfpx.service -p ActiveState -p UnitFileState
journalctl --user -u qwen36-prefill-test.service -n 40 --no-pager
curl -fsS http://100.96.8.54:8082/health
```

Verify the actual process binary/environment, not just a stale service file.
The generic cookbook defaults to loopback; the live private listener uses the
explicit host above. Do not launch a parallel benchmark while the user tests.

## Recovery after an intentional inference stop

The stable helper refuses to stop an active unit. To review restoring the earlier
non-MR binary/settings after an intentional stop:

```bash
cd "$HOME/qwen3.6-35b-a3b-strix-halo"
python3 ops/restore-francis-baseline.py --bind-host "$(tailscale ip -4)" --dry-run
```

Omitting `--dry-run` restores a stopped baseline, waits for health and verifies
a completion. It does not change the older boot unit or restart OpenClaw.
The current optimized trial was started on a free port with:

```bash
systemd-run --user --unit=qwen36-prefill-test python3 \
  "$HOME/qwen3.6-35b-a3b-strix-halo/scripts/serve.py" \
  --binary "$HOME/src/rocmfp4-prefill-next/build-vulkan/bin/llama-server" \
  --optimized-prefill --host 100.96.8.54
```

This is a reconstruction command, not an instruction to restart the live unit.

## Next work

1. Observe the user's Telegram/OpenClaw task and multi-turn tool/cache behavior.
   The user will assess answer quality. Local API checks are already complete.
2. Measure longer contexts if requested; 64K results do not establish 128K/150K
   throughput. Keep the served context limit explicit.
3. If more optimization is requested, profile the combined build first. Repeated
   routing and remaining attention copies are unmeasured possibilities.
4. Treat boot persistence and context expansion as separate operational changes.

Keep the failed smaller-tile, no-gain indirect-dispatch and earlier numerical
experiments in [the history](docs/EXPERIMENTS.md). Do not rerun them unchanged.
