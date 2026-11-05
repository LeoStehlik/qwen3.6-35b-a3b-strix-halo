# Handoff for Francis

Updated 2026-11-05 UTC: **daily OpenClaw acceptance withdrawn** after real-use
failures. The user authorized taking Qwen3.6 offline, retaining all artifacts,
and restoring Qwen3.8-27B dense + DFlash2. Documentation is published before the
service handover; the switch is pending at this update. The frozen performance
tag and code remain unchanged. [Findings and evidence limits](docs/OPERATIONAL-SETBACK.md).

## Goal and constraints

The priority is useful, reliable OpenClaw task completion with faster long-context
**prefill** on Qwen3.8-27B dense. Qwen3.6's roughly 1,047 tok/s performance success
did not meet the user's agent-reliability requirement. Its 50–100 tok/s generation
was useful only when it led to a completed answer.
Do not substitute a weaker model, reduce the tool inventory, impose an 8K output
cap, or present short-prompt throughput as a large-context result.

Do not restart the Mac mini's OpenClaw gateway or reboot that Mac. This project
runs on the Linux Strix Halo inference host. The user authorized local inference
interruptions for the explicitly requested model handover. Publish the setback
documentation first, then stop Qwen3.6 and restore Qwen3.8. Do not send channel
messages or launch OpenClaw agent turns: the user supplies real prompts.

The user reviewed the numerical evidence and accepted the observed MR variance
as sufficient to continue optimization and practical testing. Exact old-kernel
answers are not the quality criterion. Preserve earlier failures as evidence;
use operator correctness, required state/output behavior and useful task results.
Do not reopen equivalence forensics without evidence of malfunction.

## Pre-handover serving snapshot — 2026-11-05

This section records the state before the authorized switch, not the desired
future deployment. Do not re-enable Qwen3.6 merely because its unit is retained.

| Item | Observed value |
|---|---|
| OpenAI-compatible base URL | `http://100.96.8.54:8082/v1` |
| Model ID | `tiercel-qwen36-a3b-fast` |
| Existing OpenClaw provider/model | `tiercelmoe/tiercel-qwen36-a3b-fast` |
| Active unit | `qwen36-prefill-test.service` |
| Active process lifetime | Transient unit continues unchanged; enabled persistent boot unit now matches it |
| Actual binary | `$HOME/src/rocmfp4-prefill-stage2/build-vulkan/bin/llama-server` |
| Launch helper | `scripts/serve.py --expert-prefill --binary <above> --host 100.96.8.54` |
| Enabled | Prior optimizations + compact expert lists, ≤32-row narrow selector, fused convolution/SiLU |
| Disabled | Old max-row-grid indirect experiment, global narrow tiles, arithmetic decode, deeper K, all profiler/oracle/trace diagnostics |
| KV / batch / ubatch / slots | F16 / 2048 / 1024 / 1 |
| Served context | 131,072; 150K initial prompts do not fit this configuration |
| Prompt cache / checkpoints | 24,576 MiB; idle slots; 32 checkpoints every 8,192 tokens |
| Readiness | Health OK, actual completion `42`, exact tool arguments and result continuation `42` |
| Reuse check | 11,239 tokens reused for tool continuation; 11,310 restored after displacement; logs confirm RAM and checkpoint restore |

`qwen36-rocmfpx.service` is **inactive and enabled for boot**, now pointing to
`rocmfp4-prefill-stage2` with exactly the live argv and optimization flags.
The user explicitly authorized this promotion. Systemd syntax, argv/environment
agreement, enablement and `Linger=yes` were checked. The live PID stayed 317073
through the configuration change; no restart/reboot was performed or tested.
Do not start the persistent unit alongside the transient listener. On the next
boot it replaces the old boot configuration. If an earlier deliberate handover
is needed, stop the transient listener first, then start the persistent unit
and verify inference. Documentation work must not trigger that interruption.
[Evidence](results/combined-profile/persistence.json),
[portable systemd instructions](docs/DEPLOYMENT.md#9-persistent-systemd-user-service).

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

## Latest result and acceptance status

The pair was accepted on November 4; **daily-use acceptance was withdrawn on
November 5**. Direct operator, state, tool and cache checks passed their bounded
scenarios. Combined-build profiling is complete. Keep its results and the frozen
tag; do not repeat the timing sweep or mistake these checks for agent reliability.
See [the operational setback](docs/OPERATIONAL-SETBACK.md).

Two cold 65,536-token server/MTP runs per configuration, with all previous
accepted optimizations retained:

| Configuration | Prefill samples | Mean improvement |
|---|---:|---:|
| Previous baseline | 958.91 / 954.49 tok/s | — |
| Expert selector only | 1,012.94 / 1,014.01 tok/s | +5.93% |
| Convolution fusion only | 986.28 / 988.37 tok/s | +3.20% |
| Both | **1,045.40 / 1,048.68 tok/s** | **+9.44%** |

Mean prefill time falls 68.50→62.59 seconds, saving 5.91 seconds. Mean generation
is essentially unchanged, 52.33→52.45 tok/s. All ten uninstrumented runs have
identical output hashes and draft behavior (63 drafted / 58 accepted).
These numbers use the fixed source fixture, F16 KV, 2048/1024 batch/ubatch,
MR H=2 and MTP; they are not the historic ~741 target-only synthetic track.
Timing disables reuse/checkpoints; the live service retains both.

Selected additions, on top of the entire previous profile:

```text
GGML_VK_MMID_COMPACT=1
GGML_VK_MMID_SELECTOR=1
GGML_VK_SSM_CONV_FUSED=1
GGML_VK_MMID_FAST_TILE=0
GGML_VK_MMID_FAST_DEQUANT=0
GGML_VK_MMID_FAST_BK=32
```

The expert policy uses 64×32/128 threads for experts with ≤32 rows and the old
64×64 kernel otherwise. It keeps the existing K tile and accumulation order.
The unmeasured 33–47-row interval conservatively retains the old kernel.
Compact tile maps explicitly carry logical expert IDs/counts; an early grid.z
count bug was caught before any model run and fixed. Four exact mixed-row cases
per mode cover 1/32/48/64 rows, empty gaps, non-contiguous IDs and first/final
expert boundaries. All 334,080 value checks per mode match, including compact off.

The convolution path fuses eligible concat/history, four-tap convolution and
SiLU. It uses the elided concat allocation as safe scratch when allocator
aliasing prevents direct output, then copies after a barrier. Forty reference
cases passed off/on; full model/state tests exercise the production layout.
The two additions remain independently switchable with the launcher options
`--disable-expert-selector` and `--disable-conv-fusion`.

Seven target/MTP state cases passed with both selected changes, including
checkpoint restore, speculative verification and multiple sequences. Required
hidden/sparse-output comparisons have zero error. The live 11,243-token tool
request produced the exact function and arguments; continuation returned `42`.
After an unrelated request displaced it, 11,310 tokens restored and the same
continuation again returned `42`. The server log confirms target/draft RAM
restoration and recurrent checkpoint reuse. This is the tested 11K scenario,
not a general 64K displaced-cache quality result.

The first readiness script incorrectly expected a different log phrase after
all response/cache checks passed. Its fallback unnecessarily restored the old
listener. The assertion was corrected against the actual server source/log;
subsequent promotion and checks passed without kernel changes.

Routing-list reuse was discarded: its ceiling is below 0.4% of total prefill.
Arithmetic FP4 decode passed exhaustive all-byte checks but was slower. Deeper
K tiles passed operator checks, were slower and had no model-level validation.
A tail-only selector's single +0.4% difference was inconclusive. Do not revive
these unchanged. Compiler residency limits and logical tile fill are not
measured runtime occupancy or DRAM bandwidth.

See [full experiment](docs/EXPERT-PREFILL.md),
[validation](docs/VALIDATION.md),
[measurements](results/prefill-stage2/summary.json) and
[live readiness](results/prefill-stage2/live-readiness.json).

## First real traffic on the promoted pair

Read-only observation at 23:26:41 UTC: the user's first request processed
**73,093 tokens at 1,012.02 tok/s**, with production checkpoints enabled.
Follow-ups restored checkpoints at tokens 72,065 and 73,325 and completed.
A later full 79,548-token pass measured **993.07 tok/s**. Generation across
these completed requests was 49.26–62.73 tok/s. No server errors were observed;
more requests were active at capture. These are workload observations, not
paired speedup measurements or an answer-quality verdict. Aggregate-only
[evidence](results/prefill-stage2/live-traffic-observations.json) is retained.

## Combined-build profiling completed

[Full profile and reproducible diagnostics](docs/COMBINED-PROFILE.md).
Cold64K GPU shares are attention30.3%, experts28.6%, dense23.6%. With production
cache/checkpoints, a 2K append to 96K reuses all98,304 prefix tokens and spends
58.0% of GPU operator time in attention (1.922/3.314s; prompt wall3.615s).
A following96-token append replays4tokens; repack takes59ms of612ms wall.
These are instrumented synthetic/public-fixture measurements, not real-task
speed or a new uninstrumented benchmark.

Native RGP/SPM probes show attention memory-unit stall signals around62–68%,
versus17% on sampled dense projections. Absolute traffic counters fail sanity
checks on some probes, so **DRAM saturation and actual wave occupancy are not
established**. Next investigate attention memory access/reuse and scheduling,
then separately short-append KV preparation. Do not re-optimize experts merely
because they were the old bottleneck.

Initial RGP attempts hit a Mesa26.0.8 zero-submit fence bug. A diagnostic-only
preload shim preserves the empty fence and unique output filenames; no driver,
serving library or production environment was changed. All GPU diagnostics have
finished. The local source and raw captures remain under
`.cache/prefill-profile-next.pjSKecT3/`. Portable helpers and aggregate-only
evidence are published. No more profiling interruption is pending.

## Upstream and maintenance

[Patch-by-patch assessment](docs/UPSTREAM.md) checks mainline at
`3de52ac38818978d6da8f386eda43a856e53714a`. Best distinct mainline candidate:
extract concat+convolution+SiLU fusion. Expert selection/compact lists fit the
ROCmFP4 fork first. F16 KV contiguity is already upstream; tiled concat has an
open PR. Current MTP APIs already overlap our older sparse-output work, so
head skipping needs a current-mainline benefit demonstration. No PR was sent.

No Mesa/Vulkan driver fork is needed. Keep the pinned source+patch approach for
now; a dedicated llama.cpp source fork may help later with releases, regular
rebases or outside contributors. New installs have explicit checksum-verified
model downloads from an immutable revision. Default prepare/build/serve now
select `prefill-stage2`; `--profile stable` explicitly chooses the older fallback.

## Previous accepted step

The preceding `prefill-next` profile reached about 956 tok/s after tiled concat
and unused draft-head elimination (final paired control 821.97 tok/s). The first
real Telegram request on that build processed 50,437 tokens at 1,029.18 tok/s
and generated 4,408 tokens at 70.20 tok/s. The user reported useful answers,
sound tools and adequate reasoning through a context reported as 78.5K.
These observations concern the previous build, not acceptance of the new pair.
Its build remains available as the immediate prior fallback. Detailed evidence
is in [step 10](docs/EXPERIMENTS.md) and [prior results](results/prefill-next/summary.json).

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
The new 11K checkpoint/tool check does not erase that earlier 65K observation
or establish complete long-context displaced-cache validation.

## Source and evidence locations

| Path under `$HOME` | Purpose |
|---|---|
| `qwen3.6-35b-a3b-strix-halo/` | This durable GitHub repository |
| `src/rocmfp4-prefill-stage2/` | Preserved Stage 2 source/build, profile 0001–0007 |
| `src/rocmfp4-prefill-next/` | Previous accepted source/build, profile 0001–0006 |
| `src/rocmfp4-mtp-output/` | Stable fallback, without MR |
| `src/rocmfp4-fa-mr/` | Earlier MR port and numerical investigation |
| `src/rocmfp4-sparse/` | Pinned base, sparse checkout |
| `src/rocmfp4-prefill-experiment/` | Foundation optimizations / indirect experiment |
| `src/rocmfp4-moe-tiles/` | Rejected tile experiment |
| `src/rocmfp4-fa-profile/` | Diagnostic profiling source/build |
| `models/qwen36-rocmfpx/` | Existing weights, projector, template |
| `.cache/prefill-profile-next.pjSKecT3/` | Combined profiles, native counters, upstream review and persistence backup |
| `.cache/prefill-stage2.u4wqau5y/` | Current raw profiles, kernel/operator tests, repeats, state/endpoint checks and promotion scripts |
| `.cache/prefill-next.hhm3990y/` | Previous protocol, builds, tests, timing and endpoint evidence |
| `.cache/fa-mr.MuLSLc/` | Earlier MR raw logs, captures and results |
| `.cache/fa-divergence.LwnTjA/` | Repeatability, layer traces, router replay and 64K quality evidence |
| `.cache/fa-profile.b5wpyH/` | Prior operator profile |
| `.cache/mtp-output.fVUPF0/` | Sparse target-logit evidence and original restore script |
| `.cache/moe-tiles.5TfjSU/` | Rejected tile evidence |
| `.cache/prefill-indirect.t1ja9L/` | Indirect dispatch evidence |
| `.cache/qwen38-ubatch-compare.fp9Hef/` | Qwen3.8 comparison and original prompt |

Preserve the source trees' uncommitted work. Some older trees are incomplete
checkouts. The `prefill-stage2` profile reconstructs the served source from
pinned upstream plus seven patches, with checksums; all 30 changed-file hashes
passed reconstruction. The previous `prefill-next` profile remains available. A failed local-clone attempt
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
The historical promoted process was started before persistence was configured with:

```bash
systemd-run --user --unit=qwen36-prefill-test python3 \
  "$HOME/qwen3.6-35b-a3b-strix-halo/scripts/serve.py" \
  --binary "$HOME/src/rocmfp4-prefill-stage2/build-vulkan/bin/llama-server" \
  --expert-prefill --host 100.96.8.54
```

This records the historical transient process. Qwen3.6 is now designated for
offline research; do not run these recovery commands unless a future experiment
explicitly calls for it. Never launch both listeners on the same port.

## Next work

1. Finish the authorized handover: disable Qwen3.6 boot startup, stop its listener,
   retain weights/source/builds, and verify Qwen3.8 GPU residency and inference.
   Do not restart the Mac/OpenClaw gateway or change its routing implicitly.
2. Profile the existing dense Q4/Q8-KV/DFlash2 build before new optimization.
   The earlier cold 64K comparison was 367.33 tok/s at 2048/2048 versus 354.59
   at 2048/1024, one trial each. MoE tuning does not transfer automatically.
3. Evaluate raw prefill and exact stable-prefix reuse independently. Use cold and
   warm time to useful action, correct tool arguments and recovery as acceptance
   criteria; preserve configured context and the user's observed capability.
4. Keep this repository's tag and performance evidence frozen. Any future Qwen3.6
   diagnostic replay must distinguish runtime, sampling, model and kernel effects;
   none has been established as the cause of the observed failure.

Keep failed/rejected experiments and their conditions in the history. Publish
only aggregate real-traffic measurements, never user conversations or credentials.
