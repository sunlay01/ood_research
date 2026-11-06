import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from model import make,Model
P=Path(__file__).resolve().parent
summ=json.loads((P/'summary.json').read_text())
out=[]
for seed in [21,22,23]:
 s=next(x for x in summ if x['seed']==seed and x['arm']=='native')
 qT=np.array([s['snapshots']['0'][k] for k in ['a','d0','b','h']])
 m=Model(make(seed,512,1))
 def fast(t,q):
  r,g,H=m.rgh(q);return -2*(r[0]-r[1])*(g[0]-g[1])
 def finish(t,q):return abs(m.gap(q))-1e-10
 finish.terminal=True;finish.direction=-1
 f=solve_ivp(fast,[0,20000],qT,events=finish,method='DOP853',rtol=1e-10,atol=1e-12)
 assert f.success and abs(m.gap(f.y[:,-1]))<2e-10
 qf=f.y[:,-1]
 def slow(t,q):
  r,g,H=m.rgh(q);n=g[0]-g[1];gr=g.sum(0)
  return -gr+n*(gr@n)/(n@n)
 v0=slow(0,qf)
 slowf=solve_ivp(slow,[0,100],qf,method='DOP853',rtol=1e-10,atol=1e-12)
 assert slowf.success
 qe=slowf.y[:,-1]
 _,gr,_=m.rgh(qT);n=gr[0]-gr[1];r,g,H=m.rgh(qT)
 vpen=-2*1e4*(r[0]-r[1])*n
 def meas(q):
  v=q[2]+q[3]*m.Z[:,0];u=q[0]*m.D+q[1]
  S=np.mean(v*v);U=np.mean(u*u)
  return dict(q=q.tolist(),S=float(S),rho=float(S/(S+U)),gap=float(m.gap(q)),
   H=float(q[3]**2*np.var(m.Z[:,0])))
 row=dict(seed=seed,warm=meas(qT),penalty_only=meas(qf),hard_final=meas(qe),
  initial_h_velocity_penalty=float(vpen[3]),
  initial_h_velocity_risk=float(-gr.sum(0)[3]),
  h_velocity_after_fast=float(v0[3]),
  source_risk_norm_at_warm=float(np.linalg.norm(gr.sum(0))))
 out.append(row);print(json.dumps(row),flush=True)
(P/'penalty_only.json').write_text(json.dumps(out,indent=2))

