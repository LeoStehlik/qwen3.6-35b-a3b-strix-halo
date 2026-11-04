#!/usr/bin/env python3
"""Run controlled local prefill comparisons without stopping existing services.

Arrange exclusive GPU use first. Server runs listen only on loopback and own
only their spawned process group. No systemctl commands are issued.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ('GGML_VK_FA_MR', 'LLAMA_MTP_SPARSE_LOGITS', 'GGML_VK_MMID_SMALLN', 'GGML_VK_MMID_INDIRECT',
         'GGML_VK_CONCAT_TRANSPOSE', 'LLAMA_MTP_SKIP_UNUSED_HEAD', 'prefill-next', 'prefill-stage2', 'GGML_VK_MMID_COMPACT', 'GGML_VK_MMID_SELECTOR', 'GGML_VK_SSM_CONV_FUSED')

def environment(flag, mode):
    env = os.environ.copy()
    for key in ('GGML_VK_FA_PROFILE', 'GGML_VK_PERF_LOGGER', 'GGML_VK_PERF_LOGGER_CONCURRENT',
                'GGML_VK_FA_F32_OUTPUT', 'GGML_CPU_FA_F32_REFERENCE', 'GGML_VK_MMID_PROFILE'):
        env.pop(key, None)
    env.update(GGML_VK_FA_KV_CONTIG='1', GGML_VK_MMID_ROWLISTS='1', LLAMA_CONV_TAIL_X='1',
               GGML_VK_MMID_INDIRECT='0', LLAMA_MTP_SPARSE_LOGITS='1', GGML_VK_MMID_SMALLN='0',
               GGML_VK_FA_MR='0', GGML_VK_FA_MR_H='2', GGML_VK_FA_MR_TRACE='1',
               GGML_VK_CONCAT_TRANSPOSE='0', LLAMA_MTP_SKIP_UNUSED_HEAD='0',
               GGML_VK_FA_MR_LAZY='1', GGML_VK_FA_MR_NOEB='1', GGML_VK_FA_MR_VPOST='1')
    env.update(GGML_VK_MMID_COMPACT='0', GGML_VK_MMID_SELECTOR='0', GGML_VK_SSM_CONV_FUSED='0',
               GGML_VK_MMID_FAST_TILE='0', GGML_VK_MMID_FAST_DEQUANT='0', GGML_VK_MMID_FAST_BK='32')
    if flag in ('GGML_VK_CONCAT_TRANSPOSE', 'LLAMA_MTP_SKIP_UNUSED_HEAD', 'prefill-next', 'prefill-stage2'):
        env['GGML_VK_FA_MR'] = '1'
    if flag in ('GGML_VK_MMID_COMPACT', 'GGML_VK_MMID_SELECTOR', 'GGML_VK_SSM_CONV_FUSED'):
        env.update(GGML_VK_FA_MR='1', GGML_VK_CONCAT_TRANSPOSE='1', LLAMA_MTP_SKIP_UNUSED_HEAD='1')
        if flag == 'GGML_VK_MMID_SELECTOR':
            env['GGML_VK_MMID_COMPACT'] = '1'
        env[flag] = str(mode)
    elif flag == 'prefill-stage2':
        env.update(GGML_VK_CONCAT_TRANSPOSE='1', LLAMA_MTP_SKIP_UNUSED_HEAD='1',
                   GGML_VK_MMID_COMPACT=str(mode), GGML_VK_MMID_SELECTOR=str(mode), GGML_VK_SSM_CONV_FUSED=str(mode))
    elif flag == 'prefill-next':
        env.update(GGML_VK_CONCAT_TRANSPOSE=str(mode), LLAMA_MTP_SKIP_UNUSED_HEAD=str(mode))
    else:
        env[flag] = str(mode)
    return env

def post(base, path, payload):
    request = urllib.request.Request(base + path, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=1200) as response:
        return json.load(response)

def server(args, mode, repeat):
    base = f'http://127.0.0.1:{args.port}'
    # Refuse a busy port before starting so requests cannot reach another server.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', args.port))
    label = f'{args.flag}-{mode}-r{repeat}'
    cmd = [str(args.bin_dir / 'llama-server'), '--model', str(args.model),
           '--cache-type-k', 'f16', '--cache-type-v', 'f16', '--spec-type', 'draft-mtp',
           '--spec-draft-device', 'Vulkan0', '--spec-draft-ngl', 'all',
           '--spec-draft-type-k', 'f16', '--spec-draft-type-v', 'f16',
           '--spec-draft-n-max', '5', '--spec-draft-p-split', '0.10',
           '--host', '127.0.0.1', '--port', str(args.port), '--alias', 'comparison',
           '--ctx-size', '131072', '--parallel', '1', '--batch-size', '2048', '--ubatch-size', '1024',
           '--threads', '16', '--threads-batch', '16', '--n-gpu-layers', '999', '--device', 'Vulkan0',
           '--flash-attn', 'on', '--no-mmap', '--cache-ram', '0', '--ctx-checkpoints', '0', '--jinja', '--metrics']
    tokens = json.loads((ROOT / 'benchmarks/fixtures/prompt-tokens.json').read_text())
    prompt = (ROOT / 'benchmarks/fixtures/prompt.txt').read_text()
    assert len(tokens) == 65536
    save = lambda name, data: (args.output / name).write_text(json.dumps(data, indent=2) + '\n')
    env = environment(args.flag, mode)
    if args.profile_attention:
        env.update(GGML_VK_FA_PROFILE='1', GGML_VK_PERF_LOGGER_FREQUENCY='1000000')
    if getattr(args, 'profile_experts', False):
        env['GGML_VK_MMID_PROFILE'] = '1'
    save(label + '-command.json', {'argv': cmd, 'explicit_environment': {k: v for k, v in env.items() if k.startswith(('GGML_', 'LLAMA_'))}})
    with (args.output / (label + '.log')).open('w') as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            ready = False
            for _ in range(180):
                if proc.poll() is not None:
                    raise RuntimeError(f'Server exited: {proc.returncode}; see {log.name}')
                try:
                    with urllib.request.urlopen(base + '/health', timeout=2) as response:
                        ready = json.load(response).get('status') == 'ok'
                    if ready:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            if not ready:
                raise RuntimeError('Server readiness timeout')
            tokenize = lambda text: post(base, '/tokenize', {'content': text, 'add_special': False, 'parse_special': True})['tokens']
            assert tokenize(prompt) == tokens, 'Fixture tokenizer mismatch'
            options = {'temperature': 0, 'seed': 12345, 'cache_prompt': False, 'ignore_eos': True}
            warm = tokenize('Warmup: explain matrix multiplication. ' * 1024)[:2048]
            post(base, '/completion', {**options, 'prompt': warm, 'n_predict': 16})
            request = urllib.request.Request(base + '/completion',
                json.dumps({**options, 'prompt': tokens, 'n_predict': 128, 'stream': True}).encode(),
                {'Content-Type': 'application/json'})
            started = time.monotonic()
            first = None
            final = None
            content = ''
            with urllib.request.urlopen(request, timeout=1200) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith('data: ') or line == 'data: [DONE]':
                        continue
                    item = json.loads(line[6:])
                    if first is None and item.get('content'):
                        first = time.monotonic() - started
                    content += item.get('content', '')
                    if item.get('stop'):
                        final = item
            assert final is not None
            timings = final['timings']
            assert timings['prompt_n'] == 65536 and timings.get('cache_n', 0) == 0
            assert timings['predicted_n'] == 128
            result = {'flag': args.flag, 'mode': mode, 'repeat': repeat, 'timings': timings,
                      'first_output_seconds': first, 'wall_seconds': time.monotonic() - started,
                      'output_sha256': hashlib.sha256(content.encode()).hexdigest()}
            save(label + '-result.json', result)
            print(json.dumps(result), flush=True)
            return result
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('server', 'synthetic'), required=True)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--flag', choices=FLAGS, default='GGML_VK_FA_MR')
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--port', type=int, default=18082)
    parser.add_argument('--profile-attention', action='store_true', help='Diagnostic server run; adds profiling overhead')
    parser.add_argument('--profile-experts', action='store_true', help='Diagnostic expert stages, row counts and selector timings; implies --profile-attention')
    args = parser.parse_args()
    if args.profile_experts:
        args.profile_attention = True
    if args.repetitions < 1:
        parser.error('repetitions must be positive')
    if args.profile_attention and args.kind != 'server':
        parser.error('--profile-attention requires --kind server')
    args.bin_dir = args.bin_dir.resolve()
    args.model = args.model.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    if args.kind == 'server':
        for repeat in range(1, args.repetitions + 1):
            for mode in ((0, 1) if repeat % 2 else (1, 0)):
                results.append(server(args, mode, repeat))
                (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    else:
        for mode in (0, 1):
            cmd = [str(args.bin_dir / 'llama-bench'), '-m', str(args.model), '-p', '65536', '-n', '0',
                   '-b', '2048', '-ub', '1024', '-ctk', 'f16', '-ctv', 'f16', '-fa', 'on',
                   '-ngl', '999', '-dev', 'Vulkan0', '-mmp', '0', '-t', '16',
                   '-r', str(args.repetitions), '-o', 'json', '--progress']
            with (args.output / f'mode-{mode}.json').open('w') as output, (args.output / f'mode-{mode}.log').open('w') as log:
                subprocess.run(cmd, env=environment(args.flag, mode), stdout=output, stderr=log, check=True)
            results.append({'flag': args.flag, 'mode': mode, 'argv': cmd,
                            'benchmark': json.loads((args.output / f'mode-{mode}.json').read_text())})
            (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')

if __name__ == '__main__':
    main()
