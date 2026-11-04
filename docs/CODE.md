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
| 0005 | Tiled bit-preserving copies for recurrent convolution inputs | Vulkan host, `concat_transpose.comp`, backend tests |
| 0006 | Skip unused draft vocabulary projection on zero-logit MTP calls | `qwen35moe.cpp`, `test-mtp-unused-head.cpp` |
| 0007 | Expert row-count selection, compact tile lists, fused convolution/SiLU and diagnostics/tests | Vulkan host/shaders, backend tests, `test-mmid-fp4.cpp`, `test-mmid-ownership.cpp` |
| experimental/5001 | Smaller per-expert MoE tiles; slower in this experiment | Vulkan host and routing tests |

Profiles: `stable` = 0001–0002; `profile` = 0001–0003; `mr` = 0001–0004;
`prefill-next` = 0001–0006; `prefill-stage2` = 0001–0007; `moe-tiles` = 0001–0002 plus 5001. The rejected MoE-tile patch is a sibling,
not an extra step to apply on top of MR.

## Runtime switches

| Variable | Optimized trial value | Meaning |
|---|---|---|
| `GGML_VK_FA_KV_CONTIG` | `1` | Repack eligible strided F16 KV for attention |
| `GGML_VK_MMID_ROWLISTS` | `1` | Build/use per-expert row lists |
| `LLAMA_CONV_TAIL_X` | `1` | Convolution-tail graph preparation |
| `GGML_VK_MMID_INDIRECT` | `0` | Older maximum-row-grid indirect experiment; no measured gain |
| `LLAMA_MTP_SPARSE_LOGITS` | `1` | Preserve hidden rows while selecting vocabulary outputs |
| `GGML_VK_MMID_SMALLN` | `0` | Only the rejected MoE-tile profile implements this |
| `GGML_VK_FA_MR` | `1` | MR attention, enabled by `--optimized-prefill` |
| `GGML_VK_FA_MR_H` | `2` | Heads per MR group |
| `GGML_VK_FA_MR_LAZY`, `..._NOEB`, `..._VPOST` | `1` | Ported scheduling options; all default on within MR |
| `GGML_VK_CONCAT_TRANSPOSE` | `1` | Tiled copies of eligible convolution inputs |
| `LLAMA_MTP_SKIP_UNUSED_HEAD` | `1` | Skip the draft vocabulary head when no logits are requested |
| `GGML_VK_MMID_COMPACT` | `1` | Dispatch live expert/tile pairs through compact work lists |
| `GGML_VK_MMID_SELECTOR` | `1` | Narrow 64×32 kernel for experts with ≤32 rows; old 64×64 otherwise |
| `GGML_VK_SSM_CONV_FUSED` | `1` | Fuse eligible concat, four-tap convolution and SiLU |
| `GGML_VK_MMID_FAST_TILE`, `..._FAST_DEQUANT` | `0` | Rejected global tile/arithmetic decode experiments |
| `GGML_VK_MMID_FAST_BK` | `32` | Retain the original K tile; deeper K was slower |

The launcher without opt-in profile flags retains the original stable profile.
`--optimized-prefill` selects the previous profile; `--expert-prefill` selects
the new combined profile. `--disable-expert-selector` and `--disable-conv-fusion`
control its additions independently. Source defaults for the additions are off.

`GGML_VK_FA_PROFILE`, `GGML_VK_PERF_LOGGER`, `GGML_VK_FA_MR_TRACE`,
`GGML_VK_FA_F32_OUTPUT`, `GGML_CPU_FA_F32_REFERENCE` and `GGML_VK_MMID_PROFILE` are diagnostic
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

## Convolution-input tiling and unused draft-head elimination

The generic concatenation kernel writes contiguous output while reading the
transposed activation tensor at a large stride. Patch 0005 uses a padded 32×33
shared-memory tile, with 32×8 threads, to coalesce the large activation reads
and the output writes. Values are copied as 32-bit words. The dispatch gate
requires concatenation along dimension zero, a history of at most 32 positions,
the expected transposed input strides, and bounded workgroup dimensions. Other
types/layouts/shapes use the existing kernel. Odd extents, padded views, nonzero
offsets, outer dimensions and fallback cases are tested against the CPU result.

MTP's `process()` calls advance draft state with no requested vocabulary logits,
yet the original Qwen MTP graph always projected the entire vocabulary. Patch
0006 keeps the hidden/state graph and omits only that unused projection when
`n_outputs == 0` and generic embeddings are not requested. Actual drafting still
uses the original head. The regression captures required logits and hidden
states after a two-ubatch zero-logit prefill, a short tail, ordinary draft steps
and state restoration, and compares the saved KV state across both modes.

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

## Expert selection and fused convolution

Compact work lists carry the logical expert ID independently of dispatch grid.z.
The matrix reader receives the original expert count explicitly in compact mode;
the original path keeps its original interpretation. Separate narrow and normal
dispatches own disjoint experts. K=32, the existing accumulator precision and
reduction order are retained. The policy's 33–47-row interval conservatively
uses the original kernel; measured 48+ buckets did not justify a global switch.

The convolution shader uses a padded shared-memory tile for 32 channels × 32
tokens, gathering three history rows and the transposed activation. Four-tap
convolution and SiLU consume that tile. Type, shape, stride, consumer, workgroup
and storage checks guard eligibility. If output aliases activation storage, an
eligible elided concat allocation supplies scratch, followed by a barrier and
copy. Unsafe overlap or unsupported graphs retain the original operations.

The patch also corrects stale fusion labels after an overlap veto. Per-call
expert counts and GPU timestamp queries are diagnostic only; normal serving
unsets those controls. See [full experiment](EXPERT-PREFILL.md).
