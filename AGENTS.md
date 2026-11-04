# Working on this deployment and experiment record

Read `handoff.md`, `docs/EXPERIMENTS.md` and `docs/VALIDATION.md` first.
Preserve the distinction between validated serving code and experimental MR
attention. Keep synthetic target-only and server/MTP measurements separate.
Never promote a kernel solely because it is faster or silently relax a failed
numerical gate. Preserve failed experiments and their conditions.

This repository contains patches for the user's independent deployment, not an
upstream llama.cpp pull request. Retain upstream attribution and license notices.
Do not upload models, credentials, real agent conversations or tensor dumps.
Documentation work must not restart inference services or OpenClaw. Benchmark
scripts launch disposable servers; arrange workload isolation before running them.

`handoff.md` records a dated operational snapshot. Verify current process state
before changing it; a systemd unit being enabled does not establish which binary
is currently serving. Distinguish historical observations from current checks.
