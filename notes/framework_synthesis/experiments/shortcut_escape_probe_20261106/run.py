import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
from pathlib import Path
import json,csv
import numpy as np
from scipy.integrate import solve_ivp
from model import make,Model

P=Path(__file__).resolve().parent
times=np.unique(np.r_[0,np.geomspace(1e-6,.1,70),np.linspace(.1,10,200),np.linspace(10,100,180),[.01,.1,1,3,10,30,100]])
summary=[]; trajectories={}

def dynamics(m,lam,active,constrained=False):
 def fun(t,q):
  r,g,H=m.rgh(q); n=g[0]-g[1]; gr=g.sum(0)
  if constrained:return -gr+n*(gr@n)/(n@n)
  return -(gr+2*lam*(r[0]-r[1])*n)*active
 def jac(t,q):
  return -m.obj(q,lam)[2]*active[:,None]
 return fun,jac

def measure(m,q,lam,active,constrained=False):
 r,g,H=m.rgh(q); d=float(r[0]-r[1]); n=g[0]-g[1]; gr=g.sum(0)
 nn=n@n; tangent=-gr+n*(gr@n)/nn
 risk=-gr*active;pen=-2*lam*d*n*active
 vel=tangent if constrained else risk+pen
 normal=vel-tangent
 z=m.Z[:,0];u=q[0]*m.D+q[1];v=q[2]+q[3]*z
 S=float(np.mean(v*v));U=float(np.mean(u*u));E=S+U
 gs=np.array([0,0,2*np.mean(v),2*np.mean(v*z)])
 gu=np.array([2*np.mean(u*m.D),2*np.mean(u),0,0])
 gp=(U*gs-S*gu)/E**2
 hp=float(q[3]**2*np.var(z))
 rhopop=(q[2]**2+q[3]**2)/(q@q)
 bq=q.copy();bq[3]=0;hq=q.copy();hq[2]=0
 db=m.gap(bq);dh=m.gap(hq)
 pb=.5*(db+d-dh);ph=.5*(dh+d-db)
 cancel=1-abs(d)/(abs(pb)+abs(ph)) if abs(pb)+abs(ph)>1e-15 else 0.
 Y=2*m.y-1
 gamma=np.logaddexp(0,-(Y*u-v))-np.logaddexp(0,-(Y*u+v))
 if m.balanced:
  fd=np.full(m.n,.1)
 else:
  F=(m.C!=Y).astype(float);fd=F[0]-F[1]
 signed=fd*gamma
 indcancel=1-abs(np.mean(signed))/np.mean(abs(signed)) if np.mean(abs(signed))>1e-15 else 0.
 return dict(a=q[0],d0=q[1],b=q[2],h=q[3],Rmean=float(r.mean()),Delta=d,
  J=float(r.sum()+lam*d*d),penalty=lam*d*d,
  S=S,U=U,rho=S/E,rho_pop=rhopop,hetero_share=hp/max(S,1e-300),hetero_energy=hp,
  shapley_b=pb,shapley_h=ph,cancellation=cancel,individual_cancellation=indcancel,
  rho_dot=float(gp@vel),rho_dot_risk=float(gp@risk),rho_dot_penalty=float(gp@pen),
  rho_dot_tangent=float(gp@tangent),rho_dot_normal=float(gp@normal),
  S_dot=float(gs@vel),U_dot=float(gu@vel),
  Rmean_dot=float(gr@vel/2),J_dot=float((gr+2*lam*d*n)@vel),
  tangent_norm=float(np.linalg.norm(tangent)),normal_norm=float(np.linalg.norm(normal)))

def run(m,start,seed,arm,lam=1e4,active=None,constrained=False):
 active=np.ones(4) if active is None else active
 fn,jac=dynamics(m,lam,active,constrained)
 opts=dict(method="DOP853",rtol=1e-10,atol=1e-12) if constrained else dict(method="Radau",jac=jac,rtol=1e-9,atol=1e-11)
 sol=solve_ivp(fn,[0,100],start,t_eval=times,**opts)
 assert sol.success,sol.message
 rows=[dict(time=float(t),**measure(m,q,lam,active,constrained)) for t,q in zip(sol.t,sol.y.T)]
 key=f"{seed}_{arm}_lam{lam:g}";trajectories[key]=rows
 arr=lambda field:np.array([r[field] for r in rows])
 rho=arr('rho'); rr=arr('Rmean'); rd=arr('Rmean_dot'); dd=arr('Delta')
 plats=[]
 for threshold in [.01,.001,.0001]:
  good=(abs(rd)/rr<=threshold)&(abs(dd)<=.001)
  stable=np.logical_and.accumulate(good[::-1])[::-1]
  ix=np.flatnonzero(stable)
  if len(ix):
   i=int(ix[0])
   plats.append(dict(threshold=threshold,start=float(times[i]),end=100.,
    rho_start=float(rho[i]),rho_end=float(rho[-1]),net_rho_gain=float(rho[-1]-rho[i]),
    max_rho_gain=float(rho[i:].max()-rho[i]),R_relative_change=float((rr[-1]-rr[i])/rr[i]),
    fraction_positive_rhodot=float(np.mean(arr('rho_dot')[i:]>1e-8))))
 snaps={str(t):rows[int(np.argmin(abs(times-t)))] for t in [0,.01,.1,1,3,10,30,100]}
 s=dict(seed=seed,arm=arm,lam=lam,plateaus=plats,snapshots=snaps,
  max_decomposition_error=float(np.max(abs(arr('rho_dot')-arr('rho_dot_tangent')-arr('rho_dot_normal')))))
 assert s['max_decomposition_error']<1e-8
 if constrained:assert np.max(abs(dd))<1e-7
 summary.append(s)
 print(json.dumps(dict(seed=seed,arm=arm,lam=lam,rho_final=float(rho[-1]),
  plateau=[dict(threshold=p['threshold'],start=p['start'],gain=p['net_rho_gain']) for p in plats])),flush=True)

for seed in range(21,33):
 data=make(seed,512,1);m=Model(data)
 warm=solve_ivp(lambda t,q:-m.obj(q,0)[1],[0,100],
  np.random.default_rng(1000).normal(0,.1,4),
  jac=lambda t,q:-m.obj(q,0)[2],method='Radau',rtol=1e-9,atol=1e-11)
 assert warm.success
 qT=warm.y[:,-1]
 run(m,qT,seed,'native')
 if seed in [21,22,23]:
  ps=np.array([m.y[m.D==s].mean() for s in [-1,1]]);us=np.log(ps/(1-ps))
  qg=np.array([(us[1]-us[0])/2,(us[1]+us[0])/2,0,0])
  run(m,qg,seed,'gray_reset')
  qh=qT.copy();qh[3]=0
  run(m,qh,seed,'no_heterogeneity',active=np.array([1,1,1,0]))
  run(Model(data,balanced=True),qT,seed,'averaged_color')
  run(m,qg,seed,'hard_constraint_reset',lam=0,constrained=True)
 if seed==22:
  for lam in [1e3,1e5]:run(m,qT,seed,'native_lambda',lam=lam)
(P/'summary.json').write_text(json.dumps(summary,indent=2))
for key,rows in trajectories.items():
 with (P/(key+'.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

