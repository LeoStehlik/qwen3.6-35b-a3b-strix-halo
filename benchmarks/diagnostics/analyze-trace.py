from pathlib import Path
import json,sys
import numpy as np
out=Path(sys.argv[1]).resolve()
kind=sys.argv[2] if len(sys.argv)>2 else 'stages';count=int(sys.argv[3]) if len(sys.argv)>3 else 65536
prefixes=[out/f'{kind}-{count}-mode-{m}' for m in (0,1)]
records=[[json.loads(l) for l in Path(str(p)+'.trace.jsonl').read_text().splitlines()] for p in prefixes]
raw=[np.memmap(str(p)+'.trace',dtype='f4',mode='r') for p in prefixes]
assert records[0]==records[1]
entries=[]
for rec in records[0]:
 arr=[np.array(a[rec['offset']//4:rec['offset']//4+rec['floats']],dtype=float).reshape(len(rec['rows']),-1) for a in raw]
 x,y=arr;diff=x-y
 nr=np.sqrt(np.sum(diff*diff,axis=1)/np.maximum(np.sum(x*x,axis=1),1e-30))
 entry={'name':rec['name'],'ubatch':rec['occurrence'],'rows':rec['rows'],'NRMSE':float(np.linalg.norm(diff)/max(np.linalg.norm(x),1e-30)),'max_row_NRMSE':float(nr.max()),'per_row_NRMSE':nr.tolist(),'maxabs':float(np.max(np.abs(diff))),'identical':bool(np.array_equal(x,y))}
 if rec['name'].startswith('ffn_moe_logits-'):
  ids=[np.argsort(-z,axis=1,kind='stable')[:,:8] for z in arr]
  sets_equal=np.all(np.sort(ids[0],axis=1)==np.sort(ids[1],axis=1),axis=1)
  entry['expert_sets_differ']=int(np.count_nonzero(~sets_equal)); entry['expert_sets_differ_rows']=np.flatnonzero(~sets_equal).tolist()
  order=np.sort(-x,axis=1);entry['baseline_rank8_rank9_margin_min']=float(np.min(order[:,8]-order[:,7]))
 entries.append(entry)
res={'kind':kind,'tokens':count,'entries':entries}
(out/f'{kind}-{count}-comparison.json').write_text(json.dumps(res,indent=2)+'\n')
for e in entries:
 if e['ubatch']==0:print(e['name'],'NRMSE',round(e['NRMSE'],7),'maxrow',round(e['max_row_NRMSE'],7),'router changes',e.get('expert_sets_differ',''))
# Verify whether instrumentation itself changed terminal outputs.
for m,p in enumerate(prefixes):
 old=np.fromfile(out/f'mode-{m}-repeat-1.logits',dtype='f4')[:count//2048*4*248320].astype(float)
 new=np.fromfile(str(p)+'.logits',dtype='f4').astype(float)
 print('instrumentation mode',m,'terminal identical',bool(np.array_equal(old,new)),'NRMSE',float(np.linalg.norm(old-new)/np.linalg.norm(old)))
