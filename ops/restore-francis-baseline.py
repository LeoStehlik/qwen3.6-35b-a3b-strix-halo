#!/usr/bin/env python3
"""Restore the existing validated Francis binary after an intentional stop.

This does not stop/restart a running unit or install persistent boot services.
"""
import argparse
import json
from pathlib import Path
import subprocess
import socket
import sys
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bind-host', required=True, help='Existing private-network address, e.g. the result of tailscale ip -4')
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
binary = Path.home() / 'src/rocmfp4-mtp-output/build-vulkan/bin/llama-server'
cmd = ['systemd-run', '--user', '--unit=qwen36-prefill-test', sys.executable,
       str(root / 'scripts/serve.py'), '--binary', str(binary), '--host', args.bind_host]
if args.dry_run:
    print(json.dumps({'argv': cmd, 'profile': 'validated MTP; MR disabled'}, indent=2))
    raise SystemExit(0)
if not binary.is_file():
    raise SystemExit(f'Missing validated binary: {binary}')
state = subprocess.run(['systemctl', '--user', 'is-active', 'qwen36-prefill-test.service'], capture_output=True, text=True)
if state.stdout.strip() in ('active', 'activating', 'reloading'):
    raise SystemExit('The unit is already running; no action taken.')
with socket.socket() as probe:
    probe.bind((args.bind_host, 8082))
subprocess.run(cmd, check=True)
base = f'http://{args.bind_host}:8082'
for _ in range(90):
    status = subprocess.check_output(['systemctl', '--user', 'show', 'qwen36-prefill-test.service', '-p', 'ActiveState', '--value'], text=True).strip()
    if status in ('failed', 'inactive', 'deactivating'):
        raise SystemExit('The requested unit stopped; inspect its journal. No readiness claim made.')
    try:
        with urllib.request.urlopen(base + '/health', timeout=2) as response:
            if json.load(response).get('status') == 'ok':
                break
    except (OSError, ValueError):
        pass
    time.sleep(1)
else:
    raise SystemExit('Health timeout; inspect journalctl --user -u qwen36-prefill-test.service')
pid = int(subprocess.check_output(['systemctl', '--user', 'show', 'qwen36-prefill-test.service', '-p', 'MainPID', '--value'], text=True))
if pid <= 0 or (Path('/proc') / str(pid) / 'exe').resolve() != binary.resolve():
    raise SystemExit('Running unit does not use the expected validated binary; no readiness claim made.')
body = {'model': 'tiercel-qwen36-a3b-fast', 'messages': [{'role': 'user', 'content': 'What is 17 + 25? Reply with only the number.'}],
        'max_tokens': 16, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}
request = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=60) as response:
    result = json.load(response)
answer = result['choices'][0]['message']['content'].strip()
if answer != '42':
    raise SystemExit(f'Completion check failed: {answer!r}; unit remains running for inspection')
print('Validated baseline restored; health OK and completion returned 42. MR remains disabled.')
