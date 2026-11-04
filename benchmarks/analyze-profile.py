#!/usr/bin/env python3
"""Extract measured-request GPU records and separate target/draft prefill."""
from collections import Counter
import json
from pathlib import Path
import re
import argparse

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('log',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
OUT=args.output
OUT.mkdir(parents=True,exist_ok=True)
graphs = {}
phase = -1
for line in args.log.read_text().splitlines():
    launch = re.search(r'launch_slot_.*task (\d+) \| processing task', line)
    if launch:
        phase = int(launch[1])
    if phase != 5 or not line.startswith(('VKP_OP\t', 'VKP_FA\t')):
        continue
    f = line.split('\t')
    key = (f[1], int(f[2]))
    graph = graphs.setdefault(key, {'context':f[1], 'graph':int(f[2]), 'ops':[], 'fa':[]})
    if f[0] == 'VKP_OP':
        assert len(f) == 10, f
        graph['ops'].append({'name':f[3], 'op':f[4], 'ns':int(f[5]), 'shape':list(map(int,f[6:10]))})
    else:
        assert len(f) == 18, f
        names = ['q_tokens','kv_tokens','br','bc','split_k','repack','bytes',
                 'repack_ns','mask_ns','attention_ns','q_heads','kv_heads','hsk','hsv']
        entry = {'name':f[3], **dict(zip(names,map(int,f[4:])))}
        graph['fa'].append(entry)

prefill = [g for g in graphs.values() if g['fa'] and all(a['q_tokens'] >= 256 for a in g['fa'])]
target = [g for g in prefill if len(g['fa']) == 10]
draft = [g for g in prefill if len(g['fa']) == 1]
unknown = [g for g in prefill if len(g['fa']) not in (1,10)]
assert len(target) == 64 and len(draft) == 64 and not unknown
assert sum(g['fa'][0]['q_tokens'] for g in target) == 65536
assert sum(g['fa'][0]['q_tokens'] for g in draft) == 65536

def summarize(group):
    fa = [a for g in group for a in g['fa']]
    ops = [a for g in group for a in g['ops']]
    by_op, by_name = Counter(), Counter()
    for a in ops:
        by_op[a['op']] += a['ns']/1e9
        name = re.sub(r'-\d+$', '', a['name'])
        by_name[name] += a['ns']/1e9
    stages = {stage+'_seconds':sum(a[stage+'_ns'] for a in fa)/1e9
              for stage in ('repack','mask','attention')}
    return {'graphs':len(group), 'attention_calls':len(fa),
            'q_tokens_per_layer':sum(g['fa'][0]['q_tokens'] for g in group),
            'GPU_operator_seconds':sum(a['ns'] for a in ops)/1e9,
            'attention_operator_seconds':sum(a['ns'] for a in ops if a['op']=='FLASH_ATTN_EXT')/1e9,
            **stages, 'repack_payload_GiB':sum(a['bytes'] for a in fa)/2**30,
            'by_operator_seconds':dict(by_op.most_common()),
            'by_node_name_seconds':dict(by_name.most_common()),
            'attention_shapes':sorted({(a['q_tokens'],a['br'],a['bc'],a['split_k'],a['q_heads'],a['kv_heads'],a['hsk'],a['hsv']) for a in fa})}

summary = {'target':summarize(target), 'draft':summarize(draft), 'unknown_prefill_graphs':len(unknown),
           'target_context_bands':[]}
for end in (16384,32768,49152,65536):
    selected = [g for g in target if end-16384 < g['fa'][0]['kv_tokens'] <= end]
    summary['target_context_bands'].append({'start':end-16384, 'end':end, **summarize(selected)})
(OUT/'mr-profile-records.json').write_text(json.dumps(list(graphs.values()))+'\n')
(OUT/'mr-profile-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
