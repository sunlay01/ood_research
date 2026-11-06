import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from model import *
P=Path(__file__).resolve().parent;records=[];saved={}
for seed in [21,22,23]:
 m=Model(make(seed,512,1));target=m.fit_erm().x
 for ini in range(3):
  q0=np.random.default_rng(1000+ini).normal(0,.1,4);times=np.r_[0,np.geomspace(1e-4,100,180)];warm=solve_ivp(lambda t,q:-m.obj(q,0)[1],[0,100],q0,jac=lambda t,q:-m.obj(q,0)[2],method='Radau',t_eval=times,rtol=1e-9,atol=1e-11);assert warm.success
  qT=warm.y[:,-1];tt=np.r_[0,np.geomspace(1e-7,2000,260)];flow=solve_ivp(lambda t,q:-m.obj(q,10000)[1],[0,2000],qT,jac=lambda t,q:-m.obj(q,10000)[2],method='Radau',t_eval=tt,rtol=1e-9,atol=1e-11);assert flow.success
  last=flow.y[:,-1];stats=m.stats(last);stats.update(seed=seed,init=ini,q_initial=q0.tolist(),warm_error=float(np.linalg.norm(qT-target)),S_initial=m.stats(q0)['S'],S_erm=m.stats(qT)['S'],gradient_norm=float(np.linalg.norm(m.obj(last,10000)[1])));records.append(stats)
  key=f'{seed}_{ini}';saved[key+'_warm_t']=warm.t;saved[key+'_warm_q']=warm.y;saved[key+'_penalty_t']=flow.t;saved[key+'_penalty_q']=flow.y
  print(seed,ini,stats['S_initial'],stats['S_erm'],stats['S'],stats['warm_error'],flush=True)
(P/'reproduced_full_flow.json').write_text(json.dumps(records,indent=2));np.savez_compressed(P/'reproduced_full_flow.npz',**saved)
