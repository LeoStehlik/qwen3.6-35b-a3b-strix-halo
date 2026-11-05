# Deployment cookbook

**Research reproduction:** daily OpenClaw acceptance was withdrawn on
2026-11-05 after [real-use failures](OPERATIONAL-SETBACK.md). The frozen Stage 2
code and launch defaults remain available for experiments. These instructions
are not a recommendation to restore it as the user's daily agent.

These steps prepare files and optionally start a foreground process. They do
not install/enable a systemd unit unless you explicitly follow the persistent
setup section. They do not stop an existing server, change OpenClaw, or switch
the active provider. For the existing Francis host, read `handoff.md`
before starting another listener on port 8082.

## 1. Prerequisites and source

The measured host uses Ubuntu 26.04.1, Ryzen AI Max+ 395, Radeon 8060S
`gfx1151`, RADV Vulkan, and roughly 115 GiB visible system memory on a nominal
128 GB machine. Build tools include a C/C++ compiler, CMake, Ninja, Vulkan
development headers/loader and `glslc`; scripts use Python 3.11 or newer.
Numerical analysis additionally needs NumPy. This is a hardware-specific
community fork, not a generic CPU or upstream llama.cpp build.

On Ubuntu, the corresponding build/runtime packages are:

```bash
sudo apt install build-essential cmake ninja-build git python3 \
  libvulkan-dev glslc libcurl4-openssl-dev mesa-vulkan-drivers vulkan-tools
vulkaninfo --summary
```

Confirm that RADV sees the Radeon GPU under the account that will serve the
model. The measured driver was Mesa26.0.8; that is a recorded environment, not
a requirement to downgrade a working driver. Allow disk space for about22GB of
model artifacts plus the source/build. The tested machine has128GB RAM; smaller
memory configurations and other GPUs have not been certified by these results.

```bash
python3 scripts/prepare-source.py --profile prefill-stage2
bash scripts/build.sh work/prefill-stage2
```

The source script uses the full commit in `source.json`; the branch name is
only provenance. It refuses to overwrite an existing destination.
`--destination PATH` selects a separate tree; `--upstream PATH` accepts a
complete local Git clone. A partial clone with missing objects may still need
its original remote. Release flags explicitly include `-O3 -DNDEBUG`.
The build includes regression executables but does not run GPU tests or start
inference. Do not build during timed GPU work on this shared-power machine.

## 2. Exact model artifacts

Expected directory: `$HOME/models/qwen36-rocmfpx`.

| File | Purpose |
|---|---|
| `Qwen3.6-35B-A3B-MTP-ROCmFP4-STRIX-embF16-headQ6.gguf` | Target and native MTP |
| `mmproj-F32.gguf` | Vision projector |
| `chat_template.jinja` | Existing chat/tool/reasoning template |

Source: [plunderstruck's repository](https://huggingface.co/plunderstruck/Qwen3.6-35B-A3B-MTP-ROCmFP4-GGUF).
Expected hashes in `models.json` come from the preceding deployment record.
The immutable public revision is
`fe537448c6eb259e6591fea8b9ed320a64ef4856`. Remote LFS hashes and template
bytes were checked against our manifest; existing local weights were not
redownloaded. For a new installation, this explicit command fetches approximately
22 GB and checks every SHA-256 before publishing each completed file:

```bash
python3 scripts/download-models.py "$HOME/models/qwen36-rocmfpx"
```

It reuses matching files, refuses to overwrite mismatched files and retains a
`.part` after failure. Inspect/remove that incomplete file before retrying;
resumable transfer is not implemented. Setup and serving never download models
implicitly. To verify an existing directory:

```bash
python3 scripts/verify-models.py "$HOME/models/qwen36-rocmfpx"
```

This reads approximately 22 GB; schedule it outside measurements. It fails on
missing/different artifacts rather than silently selecting a new revision.

## 3. Review and launch

```bash
python3 scripts/serve.py --dry-run
python3 scripts/serve.py
```

Use `--binary PATH` for an already-built server, `--model-dir PATH` for another
artifact location, and `--host HOST --port PORT` for the intended binding.
Default is loopback. Remote clients should use the existing private-network
address or an SSH tunnel. This launcher configures no authentication layer;
choose the binding/access controls accordingly.

| Setting | Value |
|---|---|
| API model | `tiercel-qwen36-a3b-fast` |
| Served context / slots | 131,072 / 1 |
| Batch / ubatch | 2048 / 1024 |
| Target / draft KV | F16 / F16 |
| Flash attention | On |
| MTP | Native, GPU, max draft 5, split probability 0.10 |
| Temperature | 0.3 default; client settings may override |
| Vision | F32 projector, minimum 1,024 image tokens |
| Host prompt-cache budget | 24,576 MiB, idle-slot caching |
| Checkpoints | 32, every 8,192 tokens |

An 8,192-token checkpoint interval is not an output limit. Benchmark output
lengths of 16/128 are test settings, not serving limits. The served context must
accommodate prompt, tools, images and output; it does not fit a 150K initial
prompt. Increasing context is separate work requiring validation.

## 4. Verify readiness with inference

```bash
curl -fsS http://127.0.0.1:8082/health
curl -fsS http://127.0.0.1:8082/v1/models
curl -fsS http://127.0.0.1:8082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"tiercel-qwen36-a3b-fast","messages":[{"role":"user","content":"What is 17 + 25? Reply with only the number."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

Expected answer: `42`. A process existing or `/health` responding alone does
not prove inference works. Arithmetic is only a smoke check, not a capability
evaluation or a tool/vision/cache acceptance test.

## 5. Experimental MR build

```bash
python3 scripts/prepare-source.py --profile mr
bash scripts/build.sh work/mr
python3 scripts/serve.py --experimental-mr --port 18082 --dry-run
```

The last command only prints configuration. Read `VALIDATION.md` before
actually running it. Compilation does not enable MR; the environment switch is
required. The serving default is now `prefill-stage2`; `--profile stable` selects the
previous fallback explicitly. Unset profiler and diagnostic oracle
variables during serving and timed throughput runs.

## 6. Additional prefill optimizations

The `prefill-next` profile adds tiled convolution-input copies and removal of
unused MTP draft vocabulary projections on top of MR:

```bash
python3 scripts/prepare-source.py --profile prefill-next
bash scripts/build.sh work/prefill-next
python3 scripts/serve.py --optimized-prefill --port 18082 --dry-run
```

After reviewing the printed configuration, omit `--dry-run` to start it on a
free port. Use the actual private-network host and port when configuring a
client. `--optimized-prefill` enables MR and both new optimizations together;
the original stable launcher remains available with `--profile stable`. Builds
disable the embedded WebUI, matching the existing API-only inference binaries.

## 7. Selected expert and convolution kernels

The current endpoint uses `prefill-stage2`, adding row-count-based expert tile
selection and fused convolution on top of `prefill-next`:

```bash
python3 scripts/prepare-source.py --profile prefill-stage2
bash scripts/build.sh work/prefill-stage2
python3 scripts/serve.py --expert-prefill --port 18082 --dry-run
```

The last command only prints configuration. Omit `--dry-run` only on an intended
free port. The selected pair is now the default; `--expert-prefill` remains an alias. Use
`--disable-expert-selector` or `--disable-conv-fusion` to isolate components.
The old stable and `--optimized-prefill` launches remain available. See
[measured gains and validation](EXPERT-PREFILL.md).

## 8. Cache and integration

Keep the existing OpenClaw provider/model identity and stable prefixes.
`cache_prompt: true` enables the tested reuse. Faster cold prefill does not
remove the value of prefix reuse and checkpoints.

This repository does not edit the Mac's OpenClaw configuration, restart its
gateway, reduce tools, or change the model. Test complete tool loops,
reasoning/stream handling, vision and cache restoration before deployment
changes. Timing runs disable caches/checkpoints for measurement isolation;
live readiness checks retain the production settings. The user will perform the
real OpenClaw prompt test; no synthetic OpenClaw agent turn is part of this
optimization run.

## 9. Persistent systemd user service

This optional section is for a Linux host with systemd. First verify a foreground
launch and stop it before starting a second process on the same port. The renderer
prints an exact binary/configuration unit; it does not install, enable or start it.
Choose the interface deliberately; loopback is the default.

```bash
mkdir -p "$HOME/.config/systemd/user"
python3 scripts/render-service.py --host 127.0.0.1 > /tmp/qwen36-rocmfpx.service
systemd-analyze --user verify /tmp/qwen36-rocmfpx.service
# Review/backup an existing unit before replacing it.
install -m 0644 /tmp/qwen36-rocmfpx.service \
  "$HOME/.config/systemd/user/qwen36-rocmfpx.service"
systemctl --user daemon-reload
systemctl --user enable qwen36-rocmfpx.service
sudo loginctl enable-linger "$USER"
# Only after the intended port/GPU is available:
systemctl --user start qwen36-rocmfpx.service
```

Use `--binary /absolute/path/to/llama-server` and `--model-dir PATH` for existing
artifacts. The unit records direct absolute paths and exact flags, clears
profiling/preload overrides, and restarts after failure. Check GPU device access
for the user service; group membership changes need a refreshed user session or
boot. On the measured Ubuntu host, the account belongs to `render` and `video`.
For a private Tailscale listener, supply its address with `--host`; startup retry
allows that address to appear after the user manager starts.

```bash
systemctl --user show qwen36-rocmfpx.service \
  -p ActiveState -p MainPID -p FragmentPath -p ExecStart
systemctl --user is-enabled qwen36-rocmfpx.service
loginctl show-user "$USER" -p Linger
journalctl --user -u qwen36-rocmfpx.service -n 30 --no-pager
```

Then repeat the health **and inference** checks above. This procedure does not
configure OpenClaw. Reproduction needs the pinned source and all seven patches;
a stock upstream binary is not equivalent. No Mesa/Vulkan driver fork is used.

On Francis specifically, the enabled `qwen36-rocmfpx.service` was updated to
match the promoted process at 23:54 UTC without restarting it. The live unit is
still `qwen36-prefill-test.service` until an intentional handover or boot; do
not start the persistent unit alongside it. Its argv and selected flags match
the live process, systemd validation passed and lingering was already enabled.
This verifies configuration, not an actual reboot. See [handoff](../handoff.md).
