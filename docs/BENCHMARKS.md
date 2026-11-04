# Benchmark reproduction

Arrange exclusive GPU use before running these commands. The scripts do not
stop services or OpenClaw. A Qwen3.8 router can remain alive with its model
unloaded; concurrent inference contaminates bandwidth and power measurements.

## Two separate tracks

| Property | Synthetic | Server/MTP |
|---|---|---|
| Prompt | llama-bench synthetic IDs | Included fixed source-code prompt |
| Length | 65,536 | 65,536 |
| Warmup | Full 65,536-token prompt | 2,048 tokens + 16 generated |
| Measured output | 0 | 128 |
| MTP | No | Native, max draft 5 |
| Cache/checkpoints | No server | Disabled; `cache_n == 0` required |
| KV / batch / ubatch | F16 / 2048 / 1024 | F16 / 2048 / 1024 |
| Timed repetitions | 2 per mode | 2 per mode |

Server samples run off/on then on/off. The historical MR experiment inserted
synthetic off/on runs between those two pairs; local timestamps are retained
in the archive named in `handoff.md`. The portable runner executes each track
separately. Clocks, temperature and order affect absolute values; compare modes
within the same protocol rather than mixing different experiments.

`fixtures/prompt.txt` is a synthetic review prompt containing MIT-licensed
Vulkan source. `prompt-tokens.json` pins exact IDs. The runner checks tokenizer
round-trip equality. These fixtures contain no Discord conversations or
OpenClaw sessions. Hashes are in `fixtures/manifest.json`.

## Performance

```bash
model="$HOME/models/qwen36-rocmfpx/Qwen3.6-35B-A3B-MTP-ROCmFP4-STRIX-embF16-headQ6.gguf"
python3 benchmarks/run-prefill.py --kind synthetic \
  --bin-dir work/mr/build-vulkan/bin --model "$model" --output runs/mr-synthetic
python3 benchmarks/run-prefill.py --kind server \
  --bin-dir work/mr/build-vulkan/bin --model "$model" --output runs/mr-server
```

The runner writes commands, logs and JSON, refusing to overwrite an output
directory. Server runs bind loopback port 18082 and terminate only their own
process group. Temperature 0, seed 12345 and `ignore_eos: true` keep the output
length controlled. TTFT means first nonempty streamed content, not first HTTP
byte. Same binary, `GGML_VK_FA_MR=0/1`, equal inherited optimizations.

Other comparisons use `--flag LLAMA_MTP_SPARSE_LOGITS` (stable build),
`--flag GGML_VK_MMID_INDIRECT`, or `--flag GGML_VK_MMID_SMALLN` (requires the
separate `moe-tiles` build). Synthetic tests do not exercise MTP.

## Profiling

```bash
python3 benchmarks/run-prefill.py --kind server \
  --bin-dir work/mr/build-vulkan/bin --model "$model" \
  --repetitions 1 --profile-attention --output runs/mr-profile
python3 benchmarks/analyze-profile.py \
  runs/mr-profile/GGML_VK_FA_MR-1-r1.log --output runs/mr-profile-analysis
```

The parser selects measured task 5 after tokenization/warmup, then checks 64
target and 64 draft graphs covering 65,536 tokens. Protocol changes can change
task IDs; investigate a failed assertion rather than deleting it. Profiling
adds overhead; exclude its wall time from uninstrumented throughput means.

## Operator and MTP checks

Both modes must pass the same CPU F32 reference cases without relaxed tolerance:

```bash
for mode in 0 1; do
  GGML_CPU_FA_F32_REFERENCE=1 GGML_VK_FA_MR="$mode" GGML_VK_FA_KV_CONTIG=1 \
    work/mr/build-vulkan/bin/test-backend-ops test \
    -o FLASH_ATTN_EXT -b Vulkan0 -p mr_case=1
done
GGML_VK_FA_MR=1 GGML_VK_FA_KV_CONTIG=1 LLAMA_MTP_SPARSE_LOGITS=1 \
  work/mr/build-vulkan/bin/test-mtp-hidden \
  "$model" benchmarks/fixtures/prompt-tokens.json
```

## Additional optimizations on top of MR

Use `prefill-next` to compare both new switches together, retaining MR on in
both modes. Component choices are `GGML_VK_CONCAT_TRANSPOSE` and
`LLAMA_MTP_SKIP_UNUSED_HEAD`; these also retain MR and leave the other component
disabled. `prefill-next` is a harness selector, not an environment variable.

```bash
python3 benchmarks/run-prefill.py --kind server --flag prefill-next \
  --bin-dir work/prefill-next/build-vulkan/bin --model "$model" \
  --output runs/prefill-next --repetitions 2
GGML_VK_CONCAT_TRANSPOSE=1 work/prefill-next/build-vulkan/bin/test-backend-ops \
  -b Vulkan0 -o CONCAT
mkdir -p runs/draft-head
for mode in 0 1; do
  GGML_VK_FA_MR=1 GGML_VK_FA_KV_CONTIG=1 GGML_VK_MMID_ROWLISTS=1 \
    LLAMA_MTP_SKIP_UNUSED_HEAD="$mode" \
    work/prefill-next/build-vulkan/bin/test-mtp-unused-head \
    "$model" benchmarks/fixtures/prompt-tokens.json "runs/draft-head/mode-$mode"
done
cmp runs/draft-head/mode-0.logits runs/draft-head/mode-1.logits
cmp runs/draft-head/mode-0.hidden runs/draft-head/mode-1.hidden
cmp runs/draft-head/mode-0.state runs/draft-head/mode-1.state
```

Those byte comparisons test elimination of an unused projection against the
same MR computation, not agreement between different attention algorithms.
The source profile also retains `test-mtp-hidden` for the target's sparse-output
and hidden-row contract. Headline timings use the server/MTP track; do not infer
a new synthetic target-only throughput number from them.

## Full-model numerical gate (historical equivalence diagnostic)

Analysis needs NumPy. Captures write large arrays into ignored `runs/`; do not
commit those arrays.

```bash
mkdir -p runs/numeric
for mode in 0 1; do
  label=off
  if test "$mode" = 1; then label=on; fi
  GGML_VK_FA_MR="$mode" GGML_VK_FA_MR_H=2 GGML_VK_FA_KV_CONTIG=1 \
    GGML_VK_MMID_ROWLISTS=1 LLAMA_CONV_TAIL_X=1 GGML_VK_MMID_INDIRECT=0 \
    work/mr/build-vulkan/bin/test-fa-mr-numeric \
    "$model" benchmarks/fixtures/prompt-tokens.json "runs/numeric/numeric-$label"
done
python3 benchmarks/compare-numerics.py runs/numeric
```

**The final command is expected to fail for the recorded candidate.** It writes
metrics before enforcing the original thresholds. `--report-only` reports
without enforcing a pass; it is not a waiver. For the old-kernel F32-output
oracle, use `GGML_VK_FA_F32_OUTPUT=1` with MR off, save a different prefix and
select that prefix with `--reference`.

## Paired prose perplexity

The local WikiText excerpt's hash is in
`results/multi-row-attention/prose-fixture.json`; the excerpt is not redistributed.
Use your own lawfully obtained text. Only an exact hash match reproduces the
recorded sample. The tool requires at least twice the context length of input
even with `--chunks 1`. Our sample had 31,661 tokens in the tool; context 12,288
scored 6,143 next-token positions from the latter half of its first chunk.

```bash
mkdir -p runs/perplexity
for mode in 0 1; do
  label=off
  if test "$mode" = 1; then label=on; fi
  GGML_VK_FA_MR="$mode" GGML_VK_FA_MR_H=2 GGML_VK_FA_KV_CONTIG=1 \
    GGML_VK_MMID_ROWLISTS=1 LLAMA_CONV_TAIL_X=1 GGML_VK_MMID_INDIRECT=0 \
    GGML_PPL_PROBS_OUT="$PWD/runs/perplexity/ppl-$label.probs" \
    work/mr/build-vulkan/bin/llama-perplexity -m "$model" -f /path/to/prose.txt \
    -c 12288 --chunks 1 -b 2048 -ub 1024 -ctk f16 -ctv f16 \
    -fa on -ngl 999 -dev Vulkan0 --no-mmap -t 16 --ppl-output-type 1
done
python3 benchmarks/analyze-perplexity.py runs/perplexity
```

This checks next-token accuracy on one excerpt, not agent capability. The
block-based uncertainty estimate is approximate. Broader quality claims need
more corpora and contexts. Unset all profiling/F32-reference diagnostics before
performance and perplexity comparisons, except the probability exporter above.
