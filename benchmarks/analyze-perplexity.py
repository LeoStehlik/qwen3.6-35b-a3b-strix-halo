from pathlib import Path
import json
import argparse
import numpy as np

parser=argparse.ArgumentParser(description='Compare paired per-token probabilities from the diagnostic perplexity exporter.')
parser.add_argument('directory',type=Path)
args=parser.parse_args()
out=args.directory
a=np.fromfile(out/'ppl-off.probs',dtype=np.float32).astype(float)
b=np.fromfile(out/'ppl-on.probs',dtype=np.float32).astype(float)
assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
assert np.array_equal(a>0,b>0), 'Scored positions differ or a probability underflowed'
mask=a>0; a=a[mask];b=b[mask]
loss_a=-np.log(a);loss_b=-np.log(b);delta=loss_b-loss_a
blocks=np.array([np.mean(delta[i:i+128]) for i in range(0,len(delta),128)])
se=float(blocks.std(ddof=1)/np.sqrt(len(blocks)))
mean_delta=float(delta.mean())
summary={'scored_tokens':len(a),'baseline_perplexity':float(np.exp(loss_a.mean())),
 'candidate_perplexity':float(np.exp(loss_b.mean())),
 'perplexity_change_percent':float((np.exp(mean_delta)-1)*100),
 'paired_mean_NLL_change':mean_delta,'approximate_128_token_block_standard_error':se,
 'approximate_95_percent_NLL_interval':[mean_delta-1.96*se,mean_delta+1.96*se],
 'scope':'One local WikiText excerpt. Measures language-model next-token accuracy, not a general tool-use/capability evaluation.',
 'within_one_percent_perplexity_margin':bool(mean_delta+1.96*se < np.log(1.01))}
(out/'perplexity-comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
