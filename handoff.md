# Handoff for Francis

Updated 2026-11-04 UTC. Read this before resuming inference work. This records
the state observed when publishing the experiment, not a live monitoring feed.

## Goal and constraints

The priority is long-context **prefill**, targeting sustained approximately
1,000 tok/s while keeping the existing Qwen3.6 model and OpenClaw capabilities.
Generation around 50–100 tok/s depending on workload/context was already useful.
Do not substitute a weaker model, reduce the tool inventory, impose an 8K output
cap, or present short-prompt throughput as a large-context result.

The user explicitly said not to restart the Mac mini's OpenClaw gateway or reboot
that Mac. This project runs on the Linux Strix Halo inference host. Earlier
authorized inference interruptions were for specific experiments; documentation
and handoff work require no interruption. Preserve running Discord/Telegram
sessions and do not send channel messages as part of this handoff.

## Current serving state

| Item | Observed value |
|---|---|
| API port / model ID | 8082 / `tiercel-qwen36-a3b-fast` |
| Network | Existing Tailscale/private address; use current provider config |
| Active unit | `qwen36-prefill-test.service` |
| Unit persistence | **Transient**, under `/run/user/1000/systemd/transient/` |
| Actual binary | `$HOME/src/rocmfp4-mtp-output/build-vulkan/bin/llama-server` |
| Stable changes | F16 KV contiguity, row lists, convolution-tail preparation, MTP sparse logits |
| Disabled | MR attention and indirect MoE dispatch |
| KV / batch / ubatch / slots | F16 / 2048 / 1024 / 1 |
| Served context | 131,072; 150K initial prompts do not fit this configuration |
| Prompt cache / checkpoints | 24,576 MiB; idle slots; 32 checkpoints every 8,192 tokens |
| Verification after testing | Health OK and actual completion returned `42` |

`qwen36-rocmfpx.service` is **inactive but enabled for boot**. It is the older
persistent unit, not the unit currently serving. The transient replacement will
not automatically become a persistent boot configuration. Do not assume a reboot
would reproduce today's binary/settings. Making the validated configuration
persistent is separate pending operational work.

`qwen38-dflash2.service` is active. Its router at loopback 8081 was left running
with model targets unloaded during experiments. Verify residency before a new
benchmark; never mix simultaneous GPU workloads and claim isolated timings.

## What just succeeded

The experimental multi-row F16 attention kernel, with H=2, achieved:

| 64K track | Baseline | Candidate | Gain |
|---|---:|---:|---:|
| Synthetic, full warmup, target only | 740.35 | 919.33 tok/s | +24.17% |
| Cold source prompt, server with MTP | 638.97 | 817.07 tok/s | +27.87% |

Server prefill saved 22.36 seconds (102.57 → 80.21). Each mode had two measured
trials. Generation in the short output check stayed near 48–49 tok/s.
Diagnostic attention compute was approximately 2.30× faster.

The ~741 baseline discussed earlier belongs to the **synthetic** track. Do not
compare it directly with the ~639 server baseline and report a regression.
The candidate does not yet establish sustained 1,000 tok/s or 150K performance.

## Why MR is not live

All 30 operator cases and the MTP regression passed. One 12K-context WikiText
sample changed perplexity by only +0.03%. But the separate 64K code-prompt test
failed its numerical-equivalence gate: mean KL 0.60334 and top-token agreement
108/128. At the first attention layer, MR was closer to an FP64 oracle than the
old kernel, so the observed divergence is not by itself proof of worse quality.
Its cause and long-context task impact still need investigation.

Do not silently relax the failed gate, promote the kernel from speed alone, or
say the candidate's production cache validation passed: that prepared script
was never executed. Read [validation](docs/VALIDATION.md).

## Source and evidence locations on the current machine

| Path under `$HOME` | Purpose |
|---|---|
| `qwen3.6-35b-a3b-strix-halo/` | This durable GitHub repository |
| `src/rocmfp4-sparse/` | Clean pinned base, sparse checkout |
| `src/rocmfp4-prefill-experiment/` | Foundation optimizations / indirect experiment |
| `src/rocmfp4-mtp-output/` | Actual validated serving source/build |
| `src/rocmfp4-moe-tiles/` | Rejected tile experiment |
| `src/rocmfp4-fa-profile/` | Diagnostic profiling source/build |
| `src/rocmfp4-fa-mr/` | Experimental MR source/build and numerical tests |
| `models/qwen36-rocmfpx/` | Existing weights, projector, template |
| `.cache/fa-mr.MuLSLc/` | Complete MR raw logs, captures, scripts and results |
| `.cache/fa-profile.b5wpyH/` | Prior operator profile |
| `.cache/mtp-output.fVUPF0/` | MTP fix evidence and original restore script |
| `.cache/moe-tiles.5TfjSU/` | Rejected tile evidence |
| `.cache/prefill-indirect.t1ja9L/` | Indirect dispatch evidence |
| `.cache/qwen38-ubatch-compare.fp9Hef/` | Qwen3.8 comparison and original prompt |

The source worktrees contain uncommitted experimental edits and are not all
complete checkouts. Preserve them. Reconstructable patches and file hashes now
live in this repository; new work should use a distinct profile/tree.

Large raw logits, hidden tensors, logs and the prose excerpt remain local.
GitHub retains sanitized results, source patches and the synthetic public-code
prompt. Do not clean the local captures until the numerical investigation is
finished or they have been deliberately archived elsewhere.

## Read-only operational checks

```bash
systemctl --user show qwen36-prefill-test.service \
  -p ActiveState -p MainPID -p FragmentPath -p ExecStart
systemctl --user show qwen36-rocmfpx.service -p ActiveState -p UnitFileState
journalctl --user -u qwen36-prefill-test.service -n 40 --no-pager
tailscale ip -4
```

Use the host reported by Tailscale with `/health` and `/v1/models` on port 8082.
Check the actual process binary, not only a stale shell script or service file.
The generic cookbook defaults to loopback; it is not evidence that the current
private-network listener moved.

## Recovery after an intentional inference stop

The old validated start script remains:
`$HOME/.cache/mtp-output.fVUPF0/start-server.sh`.
For a durable replacement that uses the same existing binary/settings:

```bash
cd "$HOME/qwen3.6-35b-a3b-strix-halo"
python3 ops/restore-francis-baseline.py --bind-host "$(tailscale ip -4)" --dry-run
```

Remove `--dry-run` only when the inference unit is intentionally stopped and
restoration is wanted. The helper refuses to stop/restart an active unit. It
starts the transient baseline, waits for health and verifies a completion. It
does not change the older enabled boot unit or restart OpenClaw. The helper was
validated in dry-run mode during documentation; it was not used to replace the
already-running service.

## Next investigation

1. Inspect 64K layer/state divergence and evaluate long-context quality on
   representative code, retrieval, source-comparison and tool tasks.
2. Keep precision changes distinct from scheduling/indexing changes. The CPU
   F32 oracle and old-kernel F32-output variant are diagnostics, not default
   runtime switches. Presence-based switches must be unset, not assigned `0`.
3. If quality warrants promotion, test MR with production cache/checkpoints,
   speculative rollback and actual multi-turn OpenClaw traffic before adoption.
4. Reprofile. Convolution-input materialization and routing reuse are possible
   next targets; no gains for them have been established here.
5. Treat persistence of the current stable unit and a larger served context as
   explicit operational changes, separate from this documentation task.

The failed smaller-tile and no-gain indirect-dispatch experiments are preserved
in [the step-by-step history](docs/EXPERIMENTS.md). Avoid rerunning them unchanged.
