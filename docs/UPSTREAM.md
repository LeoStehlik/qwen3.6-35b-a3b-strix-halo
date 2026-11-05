# Upstream candidates and fork strategy

Assessment: 2026-11-04 UTC. This is a review of the patch series and current
upstream source/PR metadata, not a claim that any patch applies or passes on
current mainline. No PR, issue or message to maintainers was submitted.
Mainline inspected: [3de52ac38818978d6da8f386eda43a856e53714a](https://github.com/ggml-org/llama.cpp/tree/3de52ac38818978d6da8f386eda43a856e53714a).
Our serving build remains the pinned ROCmFP4 base plus patches 0001–0007.

## What is worth contributing

**Best distinct mainline candidate: fused CONCAT → SSM_CONV → SiLU**, extracted
from patch 0007. **Best immediate downstream candidate: compact expert dispatch
and row-count tile selection**, also extracted from 0007, for `rocmfp4-llama`.
The whole experiment patch stack is not an appropriate single upstream PR.

| Local work | Assessment and intended destination | Preparation required |
|---|---|---|
| 0001: F16 KV contiguity | Already represented upstream: Nathan Wilson's [#27703](https://github.com/ggml-org/llama.cpp/pull/27703) merged October 15. Do not resubmit the borrowed idea. | Compare integration details only if a distinct defect remains; retain attribution. |
| 0001: expert row lists / convolution-tail graph preparation | Potentially useful components for the ROCmFP4 fork, split by purpose. Old indirect dispatch had no measured win and should stay out. | Separate accepted work from experiments; preserve recurrent-state and rollback tests. Evaluate mainline's current graph/dispatch design before proposing a port. |
| 0002: sparse vocabulary outputs with all-token MTP hidden rows | Valuable on our pinned base, but not a fresh mainline proposal as-is. Current mainline already separates nextn hidden extraction from selected outputs. [#23198](https://github.com/ggml-org/llama.cpp/pull/23198) addressed MTP prompt-logit copying; newer APIs differ further. | Check for a remaining reproducible gap in current mainline. Offer downstream fix/tests where applicable. Do not advertise our old-base gain as mainline's gain. |
| 0003: profiling | Keep as a research helper for now. Stage counters might later be useful as a small diagnostic PR. | Integrate with existing profiling facilities; remove experiment-specific state and flags. No production speedup to submit. |
| 0004: multi-row attention | Coordinate with original author Nathan Wilson; preserve his kernel authorship. A ROCmFP4 integration is distinct from claiming the attention design. Not ready as our monolithic mainline PR. | Extract kernel/integration from reference oracles and trace tools; review latest attention interfaces, capability guards, model quality and multiple GPU/model cases. User acceptance of local numerical variance is not an upstream-wide validation result. |
| 0005: tiled CONCAT | Worth contributing evidence/test coverage, **not a duplicate PR**. [#30591](https://github.com/ggml-org/llama.cpp/pull/30591) is open, authored by 0cc4m from Nathan's proposal, with multiple GPUs measured. [#30149](https://github.com/ggml-org/llama.cpp/pull/30149) is another closed/unmerged approach. | Compare our stride/offset/outer-dimension/fallback tests with the existing PR and identify missing cases. Current master still has the generic concat shader in the inspected path. |
| 0006: skip unused draft vocabulary head | Small, understandable candidate for the ROCmFP4 fork. Mainline value is **conditional**: its Qwen MTP graph now selects output rows before the head, unlike our old base. | Reproduce any remaining zero-output head work under current nextn APIs before a mainline PR. Carry the bit-exact hidden/logit/state regression; test dense and MoE MTP and embeddings requests. |
| 0007: compact expert lists + ≤32-row narrow kernel | Strong downstream PR candidate: +5.93% whole cold64K prefill in our independent arm, with exact ownership tests. Current mainline lacks `GGML_TYPE_Q4_0_ROCMFP4_FAST` in the inspected type header. | Strip slower arithmetic-decode/K-tile experiments and profiler code. Keep logical expert count independent of dispatch geometry. Do not hardcode the measured ≤32 policy for every GPU/type without validation. A general mainline port is separate work. |
| 0007: fused concat/convolution/SiLU | Strongest distinct mainline candidate: +3.20% whole cold64K prefill in its independent arm. Existing [#22653](https://github.com/ggml-org/llama.cpp/pull/22653) fused SSM_CONV(+bias)+SiLU; our extra contribution absorbs the strided concat/history read and avoids its intermediate. | Port onto current fusion machinery, retain alias scratch/barriers and all guards, demonstrate an actual gain over current mainline, and test fallback layouts, multiple sequences and other Vulkan GPUs. Split from expert work. |
| 5001 / rejected decode and deeper-K variants | Do not propose as performance improvements. They were slower, inconclusive or lacked model-level validation. | Preserve the negative evidence here. |

Mainline's Qwen MTP graph uses `ggml_get_rows(..., inp_out_ids)` before its output
projection; `llama-context.cpp` extracts unmasked nextn rows separately. That is
why patches 0002/0006 need reassessment rather than a mechanical repost.
Mainline's recent [#30141](https://github.com/ggml-org/llama.cpp/pull/30141), merged
November 3, adds speculative MTP hidden-state save/restore. This is relevant
upstream activity; it does not by itself diagnose our earlier cache observation
or justify changing the accepted serving build during user traffic.

A separate possible **Mesa bug report/fix** is the diagnostic zero-submit fence
issue exposed by native RGP per-submit tracing. First make a minimal Vulkan
reproducer and check a current Mesa build. Our process-local capture workaround
is not a driver patch and should not be packaged into production inference.

Before actual submission, prepare each candidate as a small change against the
destination's current source and run its meaningful regression/performance
checks. Preserve attribution and clearly distinguish one-host evidence from
cross-device support. Upstream's [contribution policy](https://github.com/ggml-org/llama.cpp/blob/3de52ac38818978d6da8f386eda43a856e53714a/CONTRIBUTING.md)
requires disclosed AI assistance, human review/responsibility and human-authored
GitHub posts. This assessment does not initiate that submission workflow.

## Do we need our own forks?

**No Vulkan/Mesa driver fork is needed for this deployment.** The serving changes
are inside llama.cpp's ggml Vulkan backend and model/runtime code. They compile
to shaders and host code used with the installed RADV driver. The RGP shim is
only a research diagnostic.

We already maintain a downstream variant in substance: a pinned
`charlie12345/rocmfp4-llama` commit plus a verified patch series. The reproduction
script checks each patch and every changed file, and CI reconstructs all six
profiles. This is sufficient for others to build the exact accepted source.
It does depend on the upstream repository retaining the pinned Git objects;
a source snapshot or full fork could later remove that availability dependency.

For now, keep this deployment repository as the published source of settings,
patches, tests and measurements. Pin releases to a full cookbook commit, source
commit, patch hashes and model revision. Do not automatically rebase the working
endpoint whenever upstream changes.

A dedicated **llama.cpp source fork** becomes worthwhile when we need regularly
maintained release branches, frequent upstream merges, CI builds of the full
source, or outside contributors working directly on these kernels. Its scope
should remain explicit: a Strix Halo/ROCmFP4 deployment branch with proven
fallbacks, not a claim to replace every upstream backend. Preserve the full
upstream history and licenses, use separate development and accepted-release
branches, and selectively upstream broadly useful pieces to reduce future
maintenance. No new fork was created during this task.

The roughly doubled prefill relative to the earliest local observations makes
this specialization worth preserving. It does not measure the general health of
llama.cpp maintenance: the starting fork/version, long-context workload, model,
weight format and GPU all matter, and some of the ideas are already merged or
under review upstream. The clean repeated result we can stand behind here is
1,045.40 / 1,048.68 tok/s at cold64K under the documented conditions.
