#!/usr/bin/env python3
"""Run eight synthetic native-counter probes. Requires an otherwise idle GPU."""
from pathlib import Path
import argparse,os,json,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
import serve
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--probe',type=Path,required=True)
parser.add_argument('--shim',type=Path,required=True)
parser.add_argument('--backend-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--dry-run',action='store_true',help='Print commands without creating files or using the GPU')
args=parser.parse_args()
out=args.output.resolve()
if not args.dry_run:out.mkdir(parents=True,exist_ok=False)
settings=serve.select_default(serve.create_parser().parse_args([]))
_,base,_=serve.command(settings)
base.update(MESA_VK_TRACE='rgp',MESA_VK_TRACE_PER_SUBMIT='1',RADV_THREAD_TRACE_INSTRUCTION_TIMING='0',RADV_THREAD_TRACE_CACHE_COUNTERS='1',LD_PRELOAD=str(args.shim.resolve()),GGML_BACKEND_DIR=str(args.backend_dir.resolve()),GGML_VK_FA_PROFILE='1',GGML_VK_MMID_PROFILE='1',GGML_VK_PERF_LOGGER_FREQUENCY='1000000')
cases=[('attention-1024-96k',['attention','1024','98304']),('attention-96-98k',['attention','96','100608']),('expert-gate-8',['expert','512','2048','8']),('expert-gate-32',['expert','512','2048','32']),('expert-gate-mixed',['expert','512','2048',str(Path(__file__).resolve().parent/'expert-64k-counts.txt')]),('expert-down-mixed',['expert','2048','512',str(Path(__file__).resolve().parent/'expert-64k-counts.txt')]),('dense-z',['dense','4096','2048','1024']),('dense-out',['dense','2048','4096','1024'])]
results=[]
for name,probe_args in cases:
 d=out/name;env=base.copy();env['RGP_OUTPUT_DIR']=str(d)
 command=[str(args.probe.resolve())]+probe_args
 if args.dry_run:
  print(json.dumps({'name':name,'argv':command,'output':str(d)}));continue
 d.mkdir()
 print('START',name,flush=True)
 with (d/'probe.log').open('w') as log:
  p=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=25)
 result={'name':name,'arguments':probe_args,'returncode':p.returncode,'captures':[f.name for f in sorted(d.glob('*.rgp'))]}
 results.append(result);(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(result),flush=True)
 if p.returncode!=0:raise RuntimeError('Native capture failed; see probe.log')
 if not result['captures']:raise RuntimeError('No native captures generated')
