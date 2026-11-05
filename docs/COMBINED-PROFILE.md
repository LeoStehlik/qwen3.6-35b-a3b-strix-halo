# Profile of the promoted combined build

Captured 2026-11-04, after the 1,045.40 / 1,048.68 tok/s pair was promoted.
**Attention is the next optimization target, particularly long-context
continuations.** No serving kernel changed during this investigation.
The user subsequently authorized this build as the persistent/default baseline.

## Where the time moved

The existing cold 64K profile sums to 61.76 seconds of target+draft GPU operator
intervals: attention 18.73 s (30.3%), expert matrix operations 17.67 s (28.6%),
dense matrix operations 14.57 s (23.6%). Attention core alone is 16.32 s.
Expert matrix work excluding its preparation fell from about 20.48 to 17.35 s.
The two measures have different boundaries; do not compare 20.48 against 17.67
as if they were the same counter. The target attention share rises from 8.8%
in the first 16K to 41.2% in the final 16K.
[Cold profiles](../results/prefill-stage2/combined-operator-profile.json).

A second diagnostic retained the **production** 24 GiB RAM cache, 32 recurrent
checkpoints and 8,192-token checkpoint interval. It seeded 98,304 tokens, appended
2,048, then appended another 96. All input comes from the public source fixture;
this is not a real OpenClaw conversation. F16 KV, MTP, 2048/1024 and one slot are
unchanged. Profiling is enabled: these are diagnostic timings, not a new paired
throughput result or a regression against the uninstrumented 64K baseline.

| Phase | Reused tokens | Evaluated tokens | Prompt wall time | GPU operator sum | Attention share of GPU |
|---|---:|---:|---:|---:|---:|
| Cold 96K | 0 | 98,304 | 113.608 s | 106.940 s | 39.8% |
| Append 2K | 98,304 | 2,048 | 3.615 s | 3.314 s | **58.0%** |
| Append 96 | 100,348 | 100 | 0.612 s | 0.443 s | **49.2%** |

The final request replays four tokens. The parser includes the small checkpoint
tails: target and draft each total exactly the API's `prompt_n`. Filtering out
all short graphs would incorrectly discard that work. Decode graphs are excluded.
For the 2K append, expert operations take 0.574 s and dense operations 0.461 s.
Attention core is 1.688 s, repack 0.118 s and mask preparation 0.116 s.
For the short append, attention core is 0.156 s and repack alone is 0.059 s.
The 0.169 s difference between its wall time and summed GPU intervals is not
attributed to checkpoints, CPU scheduling or synchronization without further
instrumentation. One-token seed decode timing is meaningless and omitted;
16-token continuation generation is not a sustained decode benchmark.
[Runs](../results/combined-profile/continuation-runs.json),
[operator breakdown](../results/combined-profile/continuation-analysis.json).

## Native counters: useful signals, bounded conclusions

The installed driver is Mesa 26.0.8-1ubuntu0.3, RADV/gfx1151. It does not expose
`VK_KHR_performance_query` here; native Mesa RGP/SPM tracing does work. Eight
synthetic single-operator probes link to the unchanged serving libraries.
They use the actual attention dimensions, major dense projection dimensions,
and one measured 64K expert-row distribution. Their data is deterministic
synthetic data, not model activations or weights. The down-projection probe
uses broadcast activations, so it is not a complete model-layout replay.

Ratios below sum their underlying numerator and denominator components before
division. They are **driver-derived block activity signals**, not fractions
of total GPU execution that are stalled. Windows include the submitted graph's
preparation; these are not instruction-level or individual-shader attribution.

| Probe | Native trace window | Memory unit busy | Memory unit stalled | LDS conflicts | L2 hit |
|---|---:|---:|---:|---:|---:|
| Attention: 1024 queries, 96K KV | 72.07 ms | 84.7% | **68.4%** | 4.4% | 89.7% |
| Attention: 96 queries, 100608 KV | 14.48 ms | 78.8% | **61.6%** | 2.6% | 52.3% |
| Expert gate: 8 rows/expert | 1.38 ms | 81.9% | 22.9% | 10.6% | 50.6% |
| Expert gate: 32 rows/expert | 1.57 ms | 90.3% | 38.6% | 9.3% | 45.7% |
| Expert gate: measured mixed rows | 2.04 ms | 89.7% | 26.5% | 9.7% | 56.7% |
| Expert down: same mixed rows | 2.09 ms | 89.9% | 21.8% | 9.8% | 62.9% |
| Dense z projection | 1.03 ms | 41.4% | 17.1% | 6.5% | 82.6% |
| Dense output projection | 1.01 ms | 46.7% | 17.1% | 6.9% | 86.7% |

Attention's larger memory-unit stall signal supports investigating memory access,
reuse and scheduling. **It does not establish external DRAM saturation.** The
reported byte counts fail physical-bandwidth sanity checks in some probes;
several write counts are also implausibly small. Their interpretation has not
been calibrated on this GPU. Raw aggregate fields remain in the
[counter evidence](../results/combined-profile/hardware-counters.json), but
absolute bandwidth claims are excluded. Actual wave occupancy and instruction
timing were not collected. Lower stalls do not prove dense GEMMs are compute-bound.
RGP tracing changes scheduling/clocks and inserts costly CPU file writes; do not
use its operator elapsed times or these windows as performance comparisons.

Three bounded diagnostic attempts stalled before the final capture succeeded.
Mesa's per-submit trace path drops zero-submit fence signaling; ggml waits for
that fence. A process-local shim converts it to one empty submission and gives
captures unique filenames. Disabling async alone and an incomplete hook did not
fix it. The working hook handles both instance and device proc lookups. This
is **diagnostic only**: no Mesa installation, Vulkan loader, inference binary or
service environment was patched. Each isolated attempt restored the endpoint.
See the [capture source and protocol](../benchmarks/diagnostics/rgp/README.md).

The counter definitions were checked against the exact Mesa 26.0.8 source:
[SPM definitions](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.0.8/src/amd/common/ac_spm.c),
[trace submission](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.0.8/src/amd/vulkan/layers/radv_sqtt_layer.c).
[Official Mesa tracing controls](https://docs.mesa3d.org/envvars.html)
describe the diagnostic environment variables.

## Next experiment and its realistic ceiling

1. Investigate attention K/V access and reuse at q≈1024 and KV≈96K. The existing
   MR kernel couples head grouping to query rows; do not blindly increase a tile
   dimension without following its indexing and resource limits.
2. Separately investigate avoiding redundant full-cache K/V preparation for short
   continuations. Keep cache invalidation, layout, mask and restore semantics exact.
3. Revisit expert or dense kernels only with evidence that a specific remaining
   regime is worth the effort. Do not infer achieved occupancy from compiler limits.

A hypothetical 1.25× faster attention core would improve the measured 2K append
by about **10.3%** overall; 1.5× would imply about **18.4%**, all else unchanged.
These are opportunity estimates, not predicted gains. The 64K core has a smaller
share: 1.5× there implies roughly 9.5% overall. Removing every repack cost would
save at most about 2% of cold64K time, versus 9.7% of this short append's wall
time (10.7% maximum throughput gain). A short-append win is not a universal
20% prefill win.

## Reproduce the continuation capture

Arrange an idle GPU and enough RAM; this launches and stops only its own
loopback server. It neither stops production services nor invokes OpenClaw.
Use a fresh output directory. Do not run concurrently with serving traffic.

```bash
python3 benchmarks/profile-continuation.py \
  --binary work/prefill-stage2/build-vulkan/bin/llama-server \
  --model-dir "$HOME/models/qwen36-rocmfpx" --output runs/continuation
python3 benchmarks/analyze-continuation.py \
  runs/continuation/server.log runs/continuation/results.json \
  --output runs/continuation/analysis.json
```

The portable parser reproduced the retained aggregate analysis exactly without
another GPU run. Raw logs/RGP captures stay local. Publication includes only
public-fixture metadata and aggregated measurements.
