from pathlib import Path
import json, sys
import numpy as np

root = Path(sys.argv[1]).resolve()
def load(mode):
    prefix = root / f'router-replay-65536-mode-{mode}.deep'
    meta = json.loads(prefix.with_suffix('.deep.json').read_text())
    arrays = {}
    for name, m in meta.items():
        raw = Path(str(prefix) + '.' + name).read_bytes()
        arrays[name] = np.ndarray(tuple(m['ne']), dtype={'f32':np.float32,'f16':np.float16}[m['type']], buffer=raw, strides=tuple(m['nb']))
    return arrays
def nrmse(ref, got):
    ref = np.asarray(ref, dtype=np.float64); got = np.asarray(got, dtype=np.float64)
    return float(np.linalg.norm(ref-got)/np.linalg.norm(ref))
base = load(0); candidate = load(1)
same = {k: bool(np.array_equal(base[k],candidate[k])) for k in ('q','k','v','m')}
assert all(same.values()), same
records = []; references = []; baseline = []; experimental = []
for kv_head in range(2):
    k = base['k'][:,:,kv_head,0].astype(np.float64)
    v = base['v'][:,:,kv_head,0].astype(np.float64)
    for row in (0,31,255,767,1023):
        for head in (0,1,7,8,15):
            if head//8 != kv_head: continue
            q = (base['q'][:,row,head,0]/16).astype(np.float16).astype(np.float64)
            scores = q@k + base['m'][:,row,0,0].astype(np.float64)
            p = np.exp(scores-scores.max()); p /= p.sum()
            ref = v@p; old = base['out'][:,head,row,0]; new = candidate['out'][:,head,row,0]
            references.append(ref); baseline.append(old); experimental.append(new)
            records.append({'row':row,'absolute_token':64512+row,'head':head,'baseline_NRMSE':nrmse(ref,old),'candidate_NRMSE':nrmse(ref,new)})
summary = {'scope':'First full-attention layer (layer 3), final 1024-query ubatch of 65536 tokens; 25 selected query/head pairs. Independent FP64 softmax/value calculation using actual scaled-F16 Q and stored F16 K/V. Router replay is active but preceding layers and captured inputs are bit-identical.',
           'inputs_identical':same,'selected_query_head_pairs':len(records),
           'baseline_NRMSE_vs_FP64_oracle':nrmse(references,baseline),
           'candidate_NRMSE_vs_FP64_oracle':nrmse(references,experimental),
           'records':records}
(root/'deep-attention-comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
