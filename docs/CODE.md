# Code map and patch order

All profiles start at [rocmfp4-llama 4795079b](https://github.com/charlie12345/rocmfp4-llama/commit/4795079b04f5e0ada6e5d2e85b12bac1e27e7873).
`scripts/prepare-source.py` verifies patch SHA-256 values, applies them in order,
and compares the resulting changed-file hashes with the actual tested trees.
Build directories and binaries are not committed.

| Patch | Purpose | Principal files |
|---|---|---|
| 0001 | F16 KV contiguity, expert row lists, indirect dispatch experiment, convolution-tail preparation and backend tests | `ggml-vulkan.cpp`, transpose/row-list/matmul shaders, `delta-net-base.cpp`, backend tests |
| 0002 | Independent all-token MTP hidden-state capture and selected vocabulary outputs | speculative/server contexts, llama batch/context, `test-mtp-hidden.cpp` |
| 0003 | Attention-stage timing and operator profile records | `ggml-vulkan.cpp` |
| 0004 | MR kernel, V transpose, guarded dispatch, numerical tests and diagnostic references | Vulkan host/shaders, CPU attention, backend/model tests, perplexity exporter |
| experimental/5001 | Smaller per-expert MoE tiles; slower in this experiment | Vulkan host and routing tests |

Profiles: `stable` = 0001–0002; `profile` = 0001–0003; `mr` = 0001–0004;
`moe-tiles` = 0001–0002 plus 5001. The rejected MoE-tile patch is a sibling,
not an extra step to apply on top of MR.

## Runtime switches

| Variable | Validated serving value | Meaning |
|---|---|---|
| `GGML_VK_FA_KV_CONTIG` | `1` | Repack eligible strided F16 KV for attention |
| `GGML_VK_MMID_ROWLISTS` | `1` | Build/use per-expert row lists |
| `LLAMA_CONV_TAIL_X` | `1` | Convolution-tail graph preparation |
| `GGML_VK_MMID_INDIRECT` | `0` | Indirect dispatch; no measured gain |
| `LLAMA_MTP_SPARSE_LOGITS` | `1` | Preserve hidden rows while selecting vocabulary outputs |
| `GGML_VK_MMID_SMALLN` | `0` | Only the rejected MoE-tile profile implements this |
| `GGML_VK_FA_MR` | `0` | Experimental MR attention; opt-in only |
| `GGML_VK_FA_MR_H` | candidate `2` | Heads per MR group |
| `GGML_VK_FA_MR_LAZY`, `..._NOEB`, `..._VPOST` | candidate `1` | Ported scheduling options; all default on within MR |

`GGML_VK_FA_PROFILE`, `GGML_VK_PERF_LOGGER`, `GGML_VK_FA_MR_TRACE`,
`GGML_VK_FA_F32_OUTPUT` and `GGML_CPU_FA_F32_REFERENCE` are diagnostic
**presence-based** switches. Unset them for normal serving; assigning `0` does
not necessarily disable them. `scripts/serve.py` clears these diagnostics.

## MTP correctness detail

The new all-hidden-state path must retain original input row order independently
of sparse output ordering. It must collect rows even in physical batches with
zero requested logits, preserve verification rows, and preserve the old MTP
batch splits. These constraints are exercised by `test-mtp-hidden`.

## MR dispatch detail

The port checks F16 K/V, F32 accumulation, aligned head size 256, compatible
coopmat1/subgroup32 support on the RADV AMD path, at least 64 query rows,
KV length divisible by 64, compatible GQA/head grouping, single outer batch,
supported mask layout, stride alignment and scratch/index size bounds.
Actual Qwen3.6 runs engaged it for the ten target attention layers and one MTP
layer at q=1024, 16 query heads, two KV heads, H=2. Decode uses the existing path.

The V transpose uses 64×64 tiles with shared-memory padding and destination
stride `KV + 128`. All scratch regions are reserved before recording GPU work.
The current port retains the original V contiguity pass before V transpose;
eliminating that extra conversion remains possible follow-up work, not a
measured optimization in this repository.

## Diagnostic changes included with MR

- `GGML_CPU_FA_F32_REFERENCE` uses F32 value accumulation in the CPU oracle.
  Ordinary CPU behavior is unchanged when it is absent.
- `GGML_VK_FA_F32_OUTPUT` adds an F32-output accumulator variant of the old
  attention kernel for investigation. It is not enabled in performance runs.
- `test-fa-mr-numeric` samples 128 full-vocabulary distributions and hidden
  vectors over the fixed 64K prompt.
- `GGML_PPL_PROBS_OUT` exports per-token probabilities from `llama-perplexity`
  for paired analysis. It is a local diagnostic interface.
- `FA_MR_DUMP_INITIAL` / `FA_MR_CAPTURE_FIRST` are test-only capture controls.

See [third-party notices](../THIRD_PARTY_NOTICES.md) for kernel provenance.
