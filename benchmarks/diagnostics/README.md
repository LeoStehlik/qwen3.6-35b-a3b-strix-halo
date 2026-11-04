# Numerical discrepancy diagnostics

These are standalone diagnostics for the pinned MR build. They do not alter
normal serving or stop services. Arrange available memory and an idle GPU;
correctness captures are not throughput measurements. Keep output under
`runs/` or outside Git. The trace/replay phases create many GiB of raw tensors.

After building the MR profile as described in the deployment cookbook:

```bash
bash benchmarks/diagnostics/build-trace.sh work/mr runs/trace-numerics
model="$HOME/models/qwen36-rocmfpx/Qwen3.6-35B-A3B-MTP-ROCmFP4-STRIX-embF16-headQ6.gguf"
python3 benchmarks/diagnostics/run-numerics.py --binary runs/trace-numerics \
  --model "$model" --output runs/discrepancy --phase repeat
python3 benchmarks/diagnostics/run-numerics.py --binary runs/trace-numerics \
  --model "$model" --output runs/discrepancy --phase trace
python3 benchmarks/diagnostics/run-numerics.py --binary runs/trace-numerics \
  --model "$model" --output runs/discrepancy --phase replay
```

Use Python with NumPy installed for the analyses:

```bash
python3 benchmarks/diagnostics/analyze-trace.py runs/discrepancy stages 65536
python3 benchmarks/diagnostics/analyze-router-replay.py runs/discrepancy
python3 benchmarks/diagnostics/analyze-deep.py runs/discrepancy
```

`trace-numerics.cpp` accepts `model.gguf tokens.json output-prefix`. Its diagnostic
environment controls are:

- `FA_NUMERIC_TOKENS`: multiple of 2,048, up to 65,536; fixture remains 64K.
- `FA_TRACE_KIND`: `none`, `layers`, or `stages`.
- `FA_REPLAY_ROUTERS`: prefix of a prior baseline stages capture. Pins scores,
  hence both expert choices and weights. Use only for causal diagnosis.
- `FA_CAPTURE_UBATCH`: capture layer-3 Q/K/V/mask/output at this ubatch index;
  63 selects the end of the 64K prompt. The FP64 analyzer assumes this index.

The source samples nine activation rows per ubatch and every router-score row.
Before interpreting traces, compare their terminal `.logits` and `.hidden`
hashes with uninstrumented captures. The retained investigation did this at
both 2K and 64K. The replay analyzer also asserts baseline-to-baseline replay
has identical terminal outputs.

For the synthetic source comparison, tool selection/continuation and displaced
prefix-cache check, use a fresh output directory and a free loopback port:

```bash
python3 benchmarks/diagnostics/task-check.py --output runs/task-check \
  --binary work/mr/build-vulkan/bin/llama-server \
  --model-dir "$HOME/models/qwen36-rocmfpx" --port 18082
```

This starts and stops only its own disposable servers, one mode at a time.
It uses native MTP and the production cache/checkpoint settings. Tools are
synthetic: the harness supplies a fixed tool result and executes no external
action. It records expected answers before running either mode. This single
scenario is a smoke test, not a broad agent benchmark.

The follow-up MTP-off cache control reuses that exact generated context and
question, comparing a cold answer, a restored answer and another restored repeat:

```bash
python3 benchmarks/diagnostics/cache-control.py --task-directory runs/task-check \
  --output runs/cache-control --binary work/mr/build-vulkan/bin/llama-server \
  --model-dir "$HOME/models/qwen36-rocmfpx" --port 18082
```

It changes speculative decoding to `none`, which can also change batching.
Matching answers here do not establish bit-identical restored hidden state or
locate a specific MTP bug. It records a separate follow-up protocol and preserves
the original task failures.

For the 64K perplexity comparison, obtain the exact public test text identified
in [the protocol](../../results/numerical-investigation/long-perplexity-protocol.json).
Run the MR build's `llama-perplexity` twice with the same file and model:
`-c 65536 --chunks 1 -b 2048 -ub 1024 -ctk f16 -ctv f16 -fa on -ngl 999
-dev Vulkan0 --no-mmap -t 16 --ppl-output-type 1`.
Use the normal stable optimization flags, `GGML_VK_FA_MR_H=2`, and
`GGML_VK_FA_MR=0`/`1`. Unset reference/profiling/replay switches.
Set `GGML_PPL_PROBS_OUT` to `ppl-off.probs`/`ppl-on.probs` in the output directory.
Then run `benchmarks/analyze-perplexity.py DIRECTORY` with NumPy. Require 32,767
scored positions, finite probabilities, and a successful CLI result; an empty
or too-short input is not a passing check.

See [the investigation](../../docs/NUMERICAL-INVESTIGATION.md) for what the tests
establish and what remains unresolved.
