#!/usr/bin/env python3
from pathlib import Path
import json
import sys
import argparse
import numpy as np

parser=argparse.ArgumentParser(description='Compare saved model distributions; fail the original equivalence gate unless --report-only is explicit.')
parser.add_argument('directory', type=Path)
parser.add_argument('--reference', default='numeric-off')
parser.add_argument('--output', default='numeric-comparison.json')
parser.add_argument('--report-only', action='store_true')
args=parser.parse_args()
out=args.directory
reference=args.reference
output=args.output
a=json.loads((out/(reference+'.json')).read_text())
b=json.loads((out/'numeric-on.json').read_text())
assert a == b
vocab=a['vocab']; rows=len(a['positions'])
ref=np.memmap(out/(reference+'.logits'),dtype=np.float32,mode='r',shape=(rows,vocab))
got=np.memmap(out/'numeric-on.logits',dtype=np.float32,mode='r',shape=(rows,vocab))
kl=[]; nrmse=[]; top1=[]; maxabs=[]
for x,y in zip(ref,got):
 x=x.astype(np.float64); y=y.astype(np.float64)
 assert np.isfinite(x).all() and np.isfinite(y).all()
 nrmse.append(float(np.sqrt(np.sum((x-y)**2)/np.sum(x*x))))
 maxabs.append(float(np.max(np.abs(x-y))))
 top1.append(bool(x.argmax()==y.argmax()))
 lx=x-x.max(); ly=y-y.max()
 lx-=np.log(np.exp(lx).sum()); ly-=np.log(np.exp(ly).sum())
 kl.append(float(np.sum(np.exp(lx)*(lx-ly))))
hx=np.fromfile(out/(reference+'.hidden'),dtype=np.float32).astype(np.float64)
hy=np.fromfile(out/'numeric-on.hidden',dtype=np.float32).astype(np.float64)
assert np.isfinite(hx).all() and np.isfinite(hy).all()
herror=float(np.sqrt(np.sum((hx-hy)**2)/np.sum(hx*hx)))
summary={'reference':reference,'sampled_positions':rows,'context_tokens':65536,'positions':a['positions'],
 'mean_KL_ref_to_candidate':float(np.mean(kl)),'max_KL':max(kl),
 'mean_logit_NRMSE':float(np.mean(nrmse)),'max_logit_NRMSE':max(nrmse),
 'max_logit_absolute_difference':max(maxabs),'greedy_agreement_count':sum(top1),
 'greedy_agreement_fraction':float(np.mean(top1)),'hidden_NRMSE':herror,
 'per_position_KL':kl,'per_position_logit_NRMSE':nrmse,'per_position_greedy_match':top1,
 'scope':'128 sampled next-token distributions on one fixed source-code prompt, not a general capability or perplexity evaluation.'}
(out/output).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if not k.startswith('per_position') and k!='positions'},indent=2))
if not args.report_only:
 assert np.mean(kl)<0.001 and max(nrmse)<0.02 and np.mean(top1)>=0.95 and herror<0.02, 'Model numerical differences require investigation'
