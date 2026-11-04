#!/usr/bin/env python3
"""Summarize measured 64K expert counts/stages; bucket counts are not occupancy counters."""
import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    entries, phase = {}, -1
    fields = ['m', 'k', 'tokens', 'experts', 'selected', 'bm', 'bn', 'quantize_y',
              'dequant_x', 'count_ns', 'stage_ns', 'lists_ns', 'matmul_ns']
    for line in args.log.read_text().splitlines():
        match = re.search(r'launch_slot_.*task (\d+) \| processing task', line)
        if match:
            phase = int(match[1])
        if phase != 5:
            continue
        f = line.split('\t')
        if f[0] == 'VKP_MMID':
            key = tuple(f[1:4])
            entries[key] = dict(zip(fields, map(int, f[5:])))
        elif f[0] == 'VKP_MMID_ROWS' and tuple(f[1:4]) in entries:
            entries[tuple(f[1:4])]['rows'] = list(map(int, f[4:]))
        elif f[0] == 'VKP_MMID_BUCKET' and tuple(f[1:4]) in entries:
            entries[tuple(f[1:4])]['large_ns'] = int(f[4])
            entries[tuple(f[1:4])]['small_ns'] = int(f[5])
    selected = [e for e in entries.values() if e['tokens'] == 1024 and 'rows' in e]
    if len(selected) != 7872:
        raise RuntimeError(f'Expected 7,872 expert calls for the pinned 64K Qwen fixture; found {len(selected)}')
    for e in selected:
        assert len(e['rows']) == e['experts']
        assert sum(e['rows']) == e['tokens'] * e['selected']
    buckets = []
    for lo, hi in [(0, 0), (1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256), (257, 1024)]:
        counts = [n for e in selected for n in e['rows'] if lo <= n <= hi]
        buckets.append(dict(range=[lo, hi], expert_instances=len(counts), rows=sum(counts),
                            tiles64=sum((n+63)//64 for n in counts),
                            tiles32=sum((n+31)//32 for n in counts)))
    rows = sum(b['rows'] for b in buckets)
    summary = dict(calls=len(selected), rows=rows, buckets=buckets,
                   fill64=rows/(64*sum(b['tiles64'] for b in buckets)),
                   fill32=rows/(32*sum(b['tiles32'] for b in buckets)),
                   stages_seconds={name: sum(e[name+'_ns'] for e in selected)/1e9
                                   for name in ('count', 'stage', 'lists', 'matmul')},
                   selector_dispatch_seconds={name: sum(e.get(name+'_ns', 0) for e in selected)/1e9
                                              for name in ('large', 'small')},
                   notes='Target and draft included. Warmup/decode excluded. Row buckets are work counts, not per-bucket elapsed time or hardware occupancy. Selector times include dispatch overhead and are instrumented.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
