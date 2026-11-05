#!/usr/bin/env python3
"""Separate prompt GPU records, including checkpoint replay tails, from decoding."""
from pathlib import Path
from collections import Counter
import json,re,argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('log',type=Path)
parser.add_argument('results',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
results=json.loads(args.results.read_text());by_graph={};task=-1
for line in args.log.read_text().splitlines():
 launch=re.search(r'launch_slot_.*task (\d+) \| processing task',line)
 if launch:task=int(launch[1])
 if not line.startswith(('VKP_OP\t','VKP_FA\t')):continue
 f=line.split('\t');key=(task,f[1],int(f[2]));g=by_graph.setdefault(key,{'task':task,'ops':[],'fa':[]})
 if f[0]=='VKP_OP':
  assert len(f)==10
  g['ops'].append({'name':f[3],'op':f[4],'ns':int(f[5]),'shape':list(map(int,f[6:10]))})
 else:
  assert len(f)==18
  names=['q','kv','br','bc','split','repack','bytes','repack_ns','mask_ns','attention_ns','q_heads','kv_heads','hsk','hsv']
  g['fa'].append({'name':f[3],**dict(zip(names,map(int,f[4:])))})
graphs=list(by_graph.values());tasks=list(dict.fromkeys(g['task'] for g in graphs if g['task']>=0));summary=[]
assert len(results)==len(tasks)==4
for result,task in zip(results,tasks):
 selected=[];target=0;draft=0;expected=result['timings']['prompt_n']
 for g in [x for x in graphs if x['task']==task]:
  ops=[o for o in g['ops'] if o['op']=='FLASH_ATTN_EXT'];assert len(ops) in (1,10)
  q=ops[0]['shape'][2]
  if len(ops)==10:
   if target>=expected:break
   target+=q
  else:
   if draft>=expected:break
   draft+=q
  selected.append(g)
  if target==expected and draft==expected:break
 assert (target,draft)==(expected,expected),(task,target,draft,expected)
 ops=Counter();names=Counter();shapes=Counter();fa=[]
 for g in selected:
  for op in g['ops']:
   ops[op['op']]+=op['ns']/1e9;names[re.sub(r'-\d+$','',op['name'])]+=op['ns']/1e9
   if op['op']=='MUL_MAT':shapes[(re.sub(r'-\d+$','',op['name']),tuple(op['shape']))]+=op['ns']/1e9
  fa+=g['fa']
 v={'label':result['label'],'task':task,'target_tokens':target,'draft_tokens':draft,'graphs':len(selected),'GPU_seconds':sum(ops.values()),'prompt_wall_seconds':result['timings']['prompt_ms']/1000,'by_operator_seconds':dict(ops.most_common()),'by_node_name_seconds':dict(names.most_common(25)),'attention_stages_seconds':{k:sum(a[k+'_ns'] for a in fa)/1e9 for k in ['repack','mask','attention']},'attention_shapes':sorted({(a['q'],a['kv'],a['br'],a['bc'],a['split']) for a in fa}),'dense_shapes':[{'node':k[0],'output_shape':k[1],'seconds':val} for k,val in shapes.most_common(12)]}
 summary.append(v)
 print(result['label'],'GPU',round(v['GPU_seconds'],3),'prompt_s',v['prompt_wall_seconds'],'attention',round(ops['FLASH_ATTN_EXT'],3),'share',round(100*ops['FLASH_ATTN_EXT']/sum(ops.values()),1))
args.output.write_text(json.dumps(summary,indent=2)+'\n')
