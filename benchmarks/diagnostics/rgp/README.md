# Native RGP/SPM probes for the promoted build

These diagnostics were used with RADV Mesa 26.0.8 on Radeon 8060S/gfx1151.
They are not normal serving components. No driver fork is required.
Read [the conclusions and limitations](../../../docs/COMBINED-PROFILE.md) first.

`kernel-probe.cpp` links to the same built ggml libraries as the server. It runs
one warm graph and one measured graph, verifies finite output and uses deterministic
synthetic tensors. Its eight cases cover attention (1024/96 queries), two dense
FP4 projections, and balanced 8/32-row or measured mixed-row expert populations.
`expert-64k-counts.txt` is one layer's final64K gate distribution: 256 counts,
8,192 assignments, maximum293. Each synthetic token selects eight distinct experts.
It is not a replay of every layer or every activation.

The C preload shim is restricted to the diagnostic process. It redirects RGP
files to unique names and preserves empty-submit fence completion under Mesa's
per-submit tracing. Do not preload it into the production service. It assumes
the single-device capture process used here, not arbitrary multi-device applications.

Build only the diagnostics, against the already tested source/build:

```bash
src="$PWD/work/prefill-stage2"
mkdir -p runs/rgp-build
c++ -O2 -std=c++17 benchmarks/diagnostics/rgp/kernel-probe.cpp \
  -I"$src/ggml/include" -L"$src/build-vulkan/bin" \
  -Wl,-rpath,"$src/build-vulkan/bin" -lggml -lggml-base \
  -o runs/rgp-build/kernel-probe
cc -O2 -shared -fPIC benchmarks/diagnostics/rgp/rgp-capture.c \
  -ldl -o runs/rgp-build/rgp-capture.so
```

With an otherwise idle GPU, run into a new directory:

```bash
python3 benchmarks/diagnostics/rgp/run-counters.py \
  --probe runs/rgp-build/kernel-probe --shim runs/rgp-build/rgp-capture.so \
  --backend-dir "$src/build-vulkan/bin" --output runs/counter-capture
python3 benchmarks/diagnostics/rgp/analyze-rgp.py runs/counter-capture \
  --output runs/counter-summary.json
```

The runner uses the promoted flags and adds `MESA_VK_TRACE=rgp`, per-submit
tracing, cache counters and the shim. Instruction timing is disabled. Each
case has a 25-second timeout. The runner does not manage services. Arrange
workload isolation separately; do not run alongside real OpenClaw requests.
Captures can occupy substantial disk space. They remain under ignored `runs/`.

The parser reads the native derived-SPM chunk128 format from this Mesa version,
checks boundaries, finite values and timestamp order, and aggregates ratios from
component sums. It selects the longest measured submit, excluding the empty fence
captures. It is a trusted-local-artifact reader, not a hardened general RGP parser.
Inspect trace shapes before applying that selection rule to different graphs.

Raw byte counters are retained but not interpreted as calibrated external DRAM
bandwidth. Native windows and RGP operator times are not production benchmarks.
Portable path parameters are the only probe changes since the measured run;
no inference kernel or serving library changed. No new GPU run was made merely
to publish these helpers.
