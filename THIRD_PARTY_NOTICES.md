# Third-party code and evidence

The base is [charlie12345/rocmfp4-llama](https://github.com/charlie12345/rocmfp4-llama),
commit `4795079b04f5e0ada6e5d2e85b12bac1e27e7873`, derived from llama.cpp.
Its MIT license is preserved in [licenses/llama.cpp-MIT.txt](licenses/llama.cpp-MIT.txt).
That license also applies to the upstream source in the synthetic prompt fixture.
Original scripts/documentation use the root MIT license.

Nathan Wilson's [Strix Halo fork](https://github.com/Nathanw1014/llama.cpp)
supplied the kernel designs and changes adapted/tested here:

| Commit | Contribution |
|---|---|
| `b1a10f981f1e5131f793d4efbd2f2ca530d3f932` | F16 KV contiguity for prefill attention |
| `79097254057ace65ef0af7882c864c1542ac74a2` | Indirect expert dispatch from row counts |
| `de6896086d0c14528e1df43e4d6a0d9190e4c75a` | Per-expert tile selection experiment |
| `925cc5b774a51dc74bc5cde718e206cdfbb93627` | Multi-row coopmat1 prefill attention |
| `6f9e002163e3145f269715a6d715e2aaf93cb521` | Lazy rescale, barrier and V-load scheduling |

The MR implementation is a port of existing work plus ROCmFP4 host integration,
guards and validation. This repository does not claim invention of the kernel
or upstream endorsement. Earlier inherited changes are preserved as a source
snapshot; the list above is not a complete authorship history of llama.cpp.

The [model artifacts](https://huggingface.co/plunderstruck/Qwen3.6-35B-A3B-MTP-ROCmFP4-GGUF)
are not redistributed or relicensed. Review their publisher's terms. The local
WikiText excerpt is not redistributed; only its hash and numerical results are
retained. Large model-output tensors and real user conversations are excluded.

Substantial Codex assistance was used to port, test, package and document this
independent deployment. No upstream PR or automated message to maintainers was
created as part of this work.
