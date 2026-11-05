# Retained evidence

These JSON files were exported from the local experiment archives listed in
`handoff.md`. Numeric measurements are preserved. Host-specific home paths were
replaced with `$HOME`, the private listener address with loopback, and response
text/full completion objects were omitted. Loopback strings in exported
historical records are sanitization, not a change to the live binding.

Directories correspond to the numbered experiments in `docs/EXPERIMENTS.md`.
Use each experiment's own baseline and protocol. Some historical summaries use
the word `enabled` to mean a runtime optimization was active; it does not imply
that its transient systemd unit was enabled persistently for boot.

The MR `deployment-decision.json` records a failed numerical-equivalence gate
and no promotion. Original passing/failing operator logs, large model captures,
profiling event logs and the prose text remain local. The repository contains
the synthetic public-source fixture and portable analysis scripts instead.

`repository-validation.json` describes packaging checks done when preparing
this repository. It is separate from model-quality validation and does not
claim that new GPU benchmarks or a live deployment were performed during
documentation work.

## Combined-build profiling and persistent acceptance

`combined-profile/` contains instrumented96K/continuation operator summaries,
native RGP/SPM counter aggregates, and the verified persistence configuration.
These are separate from the uninstrumented cold64K headline results. Raw driver
traffic counters are not calibrated physical bandwidth. No traces, tensors or
real conversations are published. See [analysis](../docs/COMBINED-PROFILE.md).
