"""Refuse latency measurements while another process uses a GPU."""
import os,subprocess

def gpu_snapshot():
 result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
 rows=[]
 for line in result.stdout.splitlines():
  if not line.strip():continue
  pid,memory=line.split(',',1);rows.append({'pid':int(pid),'used_memory_mib':memory.strip()})
 return rows

def require_idle_gpu():
 rows=gpu_snapshot();others=[r for r in rows if r['pid']!=os.getpid()]
 if others:raise RuntimeError(f'External GPU processes invalidate timing: {others}')
 return rows
