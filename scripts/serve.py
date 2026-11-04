#!/usr/bin/env python3
"""Run the pinned Qwen3.6 configuration in the foreground; no service mutations."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen3.6-35B-A3B-MTP-ROCmFP4-STRIX-embF16-headQ6.gguf'

def command(args):
    optimized = getattr(args, 'optimized_prefill', False)
    profile = 'prefill-next' if optimized else ('mr' if args.experimental_mr else 'stable')
    binary = args.binary or ROOT / 'work' / profile / 'build-vulkan/bin/llama-server'
    cmd = [str(binary), '--model', str(args.model_dir / MODEL),
           '--mmproj', str(args.model_dir / 'mmproj-F32.gguf'),
           '--chat-template-file', str(args.model_dir / 'chat_template.jinja'),
           '--alias', 'tiercel-qwen36-a3b-fast', '--host', args.host, '--port', str(args.port),
           '--n-gpu-layers', '999', '--device', 'Vulkan0', '--ctx-size', '131072',
           '--cache-type-k', 'f16', '--cache-type-v', 'f16', '--flash-attn', 'on',
           '--batch-size', '2048', '--ubatch-size', '1024', '--parallel', '1', '--no-mmap',
           '--spec-type', 'draft-mtp', '--spec-draft-device', 'Vulkan0', '--spec-draft-ngl', 'all',
           '--spec-draft-type-k', 'f16', '--spec-draft-type-v', 'f16',
           '--spec-draft-n-max', '5', '--spec-draft-p-split', '0.10',
           '--temp', '0.3', '--jinja', '--reasoning-format', 'deepseek', '--image-min-tokens', '1024',
           '--cache-ram', '24576', '--cache-idle-slots', '--ctx-checkpoints', '32',
           '--checkpoint-every-n-tokens', '8192', '--metrics']
    env = os.environ.copy()
    # Presence enables diagnostic flags; setting them to "0" is insufficient.
    for key in ('GGML_VK_FA_PROFILE', 'GGML_VK_PERF_LOGGER', 'GGML_VK_PERF_LOGGER_CONCURRENT',
                'GGML_VK_FA_F32_OUTPUT', 'GGML_CPU_FA_F32_REFERENCE', 'GGML_VK_FA_MR_TRACE'):
        env.pop(key, None)
    overrides = dict(GGML_VK_FA_KV_CONTIG='1', GGML_VK_MMID_ROWLISTS='1', LLAMA_CONV_TAIL_X='1',
                     GGML_VK_MMID_INDIRECT='0', LLAMA_MTP_SPARSE_LOGITS='1',
                     GGML_VK_MMID_SMALLN='0', GGML_VK_FA_MR='1' if args.experimental_mr or optimized else '0',
                     GGML_VK_CONCAT_TRANSPOSE='1' if optimized else '0',
                     LLAMA_MTP_SKIP_UNUSED_HEAD='1' if optimized else '0',
                     GGML_VK_FA_MR_H='2', GGML_VK_FA_MR_LAZY='1', GGML_VK_FA_MR_NOEB='1', GGML_VK_FA_MR_VPOST='1')
    env.update(overrides)
    return cmd, env, overrides

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--model-dir', type=Path, default=Path.home() / 'models/qwen36-rocmfpx')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8082)
    parser.add_argument('--experimental-mr', action='store_true', help='Explicitly enable unpromoted MR attention; see docs/VALIDATION.md')
    parser.add_argument('--optimized-prefill', action='store_true', help='Enable MR plus tiled convolution-input copies and unused draft-head elimination; see docs/EXPERIMENTS.md')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cmd, env, overrides = command(args)
    if args.dry_run:
        print(json.dumps({'argv': cmd, 'environment_overrides': overrides}, indent=2))
        return
    for path in (Path(cmd[0]), args.model_dir / MODEL, args.model_dir / 'mmproj-F32.gguf', args.model_dir / 'chat_template.jinja'):
        if not path.is_file():
            parser.error(f'Missing artifact: {path}')
    os.execve(cmd[0], cmd, env)

if __name__ == '__main__':
    main()
