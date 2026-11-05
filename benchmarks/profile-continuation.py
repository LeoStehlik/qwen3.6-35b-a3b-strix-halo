#!/usr/bin/env python3
"""Profile cold 96K and cached continuations on a disposable loopback server."""
from pathlib import Path
import json,os,subprocess,time,urllib.request,signal,hashlib,socket
import argparse,sys
repo=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo/'scripts'))
import serve
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary',type=Path,required=True)
parser.add_argument('--model-dir',type=Path,default=Path.home()/'models/qwen36-rocmfpx')
parser.add_argument('--port',type=int,default=18082)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
settings=serve.select_default(serve.create_parser().parse_args(['--binary',str(args.binary.resolve()),'--model-dir',str(args.model_dir.resolve()),'--port',str(args.port)]))
cmd,env,_=serve.command(settings)
base=f'http://127.0.0.1:{args.port}'
with socket.socket() as s:s.bind(('127.0.0.1',args.port))
env.update(GGML_VK_FA_PROFILE='1',GGML_VK_MMID_PROFILE='1',GGML_VK_PERF_LOGGER_FREQUENCY='1000000')
def save(name,obj):(out/name).write_text(json.dumps(obj,indent=2)+'\n')
def post(path,body):
 req=urllib.request.Request(base+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=900) as f:return json.load(f)
save('command.json',{'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith(('GGML_','LLAMA_'))}})
with (out/'server.log').open('w') as log:
 p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  for _ in range(120):
   if p.poll() is not None:raise RuntimeError('Profile server exited')
   try:
    with urllib.request.urlopen(base+'/health',timeout=2) as f:ready=json.load(f)['status']=='ok'
    if ready:break
   except Exception:pass
   time.sleep(1)
  else:raise RuntimeError('Profile health timeout')
  fixture=json.loads((repo/'benchmarks/fixtures/prompt-tokens.json').read_text())
  assert len(fixture)==65536
  seed=fixture+fixture[:32768]
  prefix=seed+fixture[32768:34816]
  cases=[('warmup',fixture[:2048],False,16),('seed-96k',seed,False,1),('append-2048',prefix,True,16),('append-96',prefix+fixture[34816:34912],True,16)]
  results=[]
  for label,tokens,cache,n in cases:
   print(time.strftime('%FT%TZ',time.gmtime()),'START',label,len(tokens),flush=True)
   start=time.monotonic();r=post('/completion',{'prompt':tokens,'n_predict':n,'temperature':0,'seed':12345,'cache_prompt':cache,'ignore_eos':True})
   timings=r['timings']
   if cache:assert timings.get('cache_n',0)>96000,timings
   result={'label':label,'input_tokens':len(tokens),'timings':timings,'wall_seconds':time.monotonic()-start,'output_sha256':hashlib.sha256(r['content'].encode()).hexdigest()}
   results.append(result);save('results.json',results);print(json.dumps(result),flush=True)
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=45)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
