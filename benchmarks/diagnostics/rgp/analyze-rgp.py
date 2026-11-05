#!/usr/bin/env python3
"""Analyze trusted, local Mesa 26.0.8 native RGP/SPM captures (not a general RGP reader)."""
from pathlib import Path
import struct,json,math,statistics,argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("captures",type=Path)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()

def parse(path):
 data=path.read_bytes();u=lambda fmt,off:struct.unpack_from('<'+fmt,data,off)
 assert u('I',0)[0]==0x50303042
 pos=u('i',16)[0];asic={};result=None;chunks=[]
 while pos<len(data):
  cid,minor,major,size,padding=u('IHHii',pos);typ=cid&255
  assert size>=16 and pos+size<=len(data),(path,pos,size)
  chunks.append({'type':typ,'size':size})
  if typ==0:
   asic={'shader_clock_hz':u('Q',pos+24)[0],'memory_clock_hz':u('Q',pos+32)[0],'gpu_timestamp_frequency':u('Q',pos+424)[0],'gpu_name':data[pos+152:pos+408].split(b'\0')[0].decode()}
  if typ==128:
   offset,flags,n,ng,nc,ncomp,interval=u('7I',pos+16)
   at=pos+offset;ts=u(str(n)+'Q',at);at+=n*8
   groups=[];counters=[];components=[]
   for _ in range(ng):
    sz,off,nl,dl,num=u('5I',at);name=data[at+off:at+off+nl].decode();groups.append(name);at+=sz
   for _ in range(nc):
    sz,off,nl,dl,num,usage=u('5IB',at)
    counters.append({'name':data[at+off:at+off+nl].decode(),'description':data[at+off+nl:at+off+nl+dl].decode(),'usage':usage,'component_ids':list(u(str(num)+'I',at+off+nl+dl))});at+=sz
   for _ in range(ncomp):
    sz,off,nl,dl,usage=u('5I',at)
    components.append({'name':data[at+off:at+off+nl].decode(),'usage':usage});at+=sz
   for entry in counters+components:
    values=u(str(n)+'d',at);at+=8*n
    assert all(math.isfinite(x) for x in values)
    entry['sum']=sum(values);entry['mean']=statistics.mean(values) if values else None;entry['max']=max(values,default=None)
   assert at==pos+size,(at,pos+size)
   assert not ts or all(a<=b for a,b in zip(ts,ts[1:])), 'timestamp ordering'
   result={'samples':n,'sampling_interval_shader_cycles':interval,'timestamp_span':ts[-1]-ts[0] if n>1 else 0,'counters':counters,'components':components}
  pos+=size
 assert pos==len(data)
 if result:
  result.update(file=path.name,asic=asic,trace_seconds=result['timestamp_span']/asic['gpu_timestamp_frequency'] if asic.get('gpu_timestamp_frequency') else None)
  for c in result['counters']:
   vals=[result['components'][i]['sum'] for i in c['component_ids']]
   if 'cache hit' in c['name'].lower() and len(vals)==3 and vals[0]:c['weighted_percent']=100*vals[1]/vals[0]
   if c['name'] in ['LDS Bank Conflict','Memory unity busy','Memory unit stalled'] and len(vals)==2 and vals[0]:c['weighted_percent']=100*vals[1]/vals[0]
  result['chunks']=chunks
 return result

all_results=[]
for directory in sorted(args.captures.iterdir()):
 if not directory.is_dir():continue
 captures=[parse(p) for p in sorted(directory.glob('measured-*.rgp'))]
 captures=[c for c in captures if c]
 if not captures:continue
 # The actual graph submit is the longest measured trace. Other submits are empty fence operations.
 best=max(captures,key=lambda c:c['trace_seconds'])
 selection={'name':directory.name,'selected':best,'other_trace_durations':[c['trace_seconds'] for c in captures if c is not best]}
 all_results.append(selection)
 print(directory.name,'trace_ms',round(best['trace_seconds']*1000,3),'samples',best['samples'],'clock',best['asic']['shader_clock_hz'])
 for c in best['counters']:
  if c['name'] in ['L0 cache hit','L1 cache hit','L2 cache hit','Instruction cache hit','Scalar cache hit','LDS Bank Conflict','Fetch size','Write size','Local video memory bytes','Memory unity busy','Memory unit stalled']:
   print(' ',c['name'],round(c.get('weighted_percent',c['sum']),3))
args.output.write_text(json.dumps(all_results,indent=2)+'\n')
