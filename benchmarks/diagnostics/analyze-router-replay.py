from pathlib import Path
import json, sys
import numpy as np

root=Path(sys.argv[1]).resolve()
def nrmse(ref,got):
    return float(np.linalg.norm(ref-got)/np.linalg.norm(ref))
ref=np.fromfile(root/'mode-0-repeat-1.logits',np.float32).astype(float).reshape(-1,248320)
hidden=np.fromfile(root/'mode-0-repeat-1.hidden',np.float32).astype(float).reshape(-1,2048)
def logsoftmax(x):
    y=x-x.max(axis=1,keepdims=True)
    return y-np.log(np.exp(y).sum(axis=1,keepdims=True))
logp=logsoftmax(ref); p=np.exp(logp)
results={}
for label in ('mode-1-repeat-1','router-replay-65536-mode-0','router-replay-65536-mode-1'):
    got=np.fromfile(root/(label+'.logits'),np.float32).astype(float).reshape(ref.shape)
    h=np.fromfile(root/(label+'.hidden'),np.float32).astype(float).reshape(hidden.shape)
    kl=np.sum(p*(logp-logsoftmax(got)),axis=1)
    results[label]={'logits_identical':bool(np.array_equal(ref,got)), 'hidden_identical':bool(np.array_equal(hidden,h)),
        'hidden_NRMSE':nrmse(hidden,h),'logit_NRMSE':nrmse(ref,got),
        'mean_KL':float(kl.mean()),'max_KL':float(kl.max()),'per_row_KL':kl.tolist(),
        'greedy_agreement_count':int((ref.argmax(axis=1)==got.argmax(axis=1)).sum()),'sampled_rows':len(ref)}
summary={'scope':'Diagnostic replay pins all MoE router scores to the baseline, hence both expert choices and weights. Not a serving change.', 'results':results}
assert results['router-replay-65536-mode-0']['logits_identical'] and results['router-replay-65536-mode-0']['hidden_identical']
(root/'router-replay-64k-comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:{a:b for a,b in v.items() if a!='per_row_KL'} for k,v in results.items()},indent=2))
