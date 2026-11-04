#!/usr/bin/env python3
"""Capture/replay the fixed 64K fixture with the standalone trace binary.

No serving-unit changes. These are correctness diagnostics, not timings.
Raw traces are large (several GiB per mode); retain them outside Git.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary',type=Path,required=True)
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--phase',choices=('repeat','trace','replay'),required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[2]
tokens=root/'benchmarks/fixtures/prompt-tokens.json'
args.output.mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
for key in list(env):
    if key.startswith('FA_') or key in ('GGML_VK_FA_PROFILE','GGML_VK_PERF_LOGGER',
        'GGML_VK_PERF_LOGGER_CONCURRENT','GGML_VK_FA_F32_OUTPUT','GGML_CPU_FA_F32_REFERENCE','GGML_VK_FA_MR_TRACE'):
        env.pop(key)
env.update(GGML_VK_FA_KV_CONTIG='1',GGML_VK_MMID_ROWLISTS='1',LLAMA_CONV_TAIL_X='1',
           GGML_VK_MMID_INDIRECT='0',GGML_VK_MMID_SMALLN='0',LLAMA_MTP_SPARSE_LOGITS='1',
           GGML_VK_FA_MR_H='2',GGML_VK_FA_MR_LAZY='1',GGML_VK_FA_MR_NOEB='1',GGML_VK_FA_MR_VPOST='1',
           FA_NUMERIC_TOKENS='65536',FA_TRACE_KIND='none' if args.phase=='repeat' else 'stages')
if args.phase=='replay':
    prefix=args.output.resolve()/'stages-65536-mode-0'
    for suffix in ('.trace','.trace.jsonl'):
        if not Path(str(prefix)+suffix).is_file(): parser.error('Run the trace phase first')
    env.update(FA_REPLAY_ROUTERS=str(prefix),FA_CAPTURE_UBATCH='63')
runs=[]
for repeat in range(1,3 if args.phase=='repeat' else 2):
    for mode in (0,1):
        label=(f'mode-{mode}-repeat-{repeat}' if args.phase=='repeat' else
               f'stages-65536-mode-{mode}' if args.phase=='trace' else f'router-replay-65536-mode-{mode}')
        prefix=args.output.resolve()/label
        if any(args.output.glob(label+'.*')): parser.error(f'Refusing to overwrite {prefix}')
        print(time.strftime('%FT%TZ',time.gmtime()),'START',label,flush=True)
        with Path(str(prefix)+'.log').open('w') as log:
            subprocess.run([str(args.binary.resolve()),str(args.model.resolve()),str(tokens),str(prefix)],
                           env={**env,'GGML_VK_FA_MR':str(mode)},stdout=log,stderr=subprocess.STDOUT,check=True)
        hashes={}
        for suffix in ('.logits','.hidden'):
            with Path(str(prefix)+suffix).open('rb') as data:
                hashes[suffix[1:]]=hashlib.file_digest(data,'sha256').hexdigest()
        runs.append({'label':label,'sha256':hashes})
        (args.output/(args.phase+'-hashes.json')).write_text(json.dumps(runs,indent=2)+'\n')
        print(time.strftime('%FT%TZ',time.gmtime()),'DONE',label,flush=True)
