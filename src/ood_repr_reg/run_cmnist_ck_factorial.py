"""Local C/K factorial response intervention on ColoredMNIST.

At a common step-300 state, estimate risk gradient B, penalty gradient C and
penalty curvature K in a small random parameter subspace.  Four local steps are
then applied and functional responses measured. This is a local intervention
pilot; it is not a claim that C/K are independently realizable algorithms.
"""
from pathlib import Path
import copy,json
import numpy as np,pandas as pd,torch
from .task3_aopi_multimethod_mechanism_survey.algorithms.registry import get_algorithm
from .task3_aopi_multimethod_mechanism_survey.functional_banks import bank_logits,build_functional_banks
from .task3_aopi_multimethod_mechanism_survey.smooth_world5 import build_smooth_world5
from .task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from .task3_cmnist_cpu_minimal.data import build_task3_data,scheduled_source_batches
from .task3_cmnist_cpu_minimal.model import build_model_from_config
from .task3_aopi_multimethod_mechanism_survey.method_trainer import train_survey_method
ROOT=Path(__file__).resolve().parents[2];CFG_PATH=ROOT/'configs/task3_aopi_multimethod_mechanism_survey.json';OUT=ROOT/'round3_redesign/method_agnostic_mechanism/ck_factorial';METHODS=('IRMv1','VREX','FISHR');SEEDS=(10,11,12);STEP=300;KDIM=8;EPS=0.001;RHO=0.001;SMOKE_SEEDS=(10,)

def flatten_params(m): return torch.cat([p.detach().reshape(-1) for p in m.parameters()])
def set_flat(m,x):
 o=0
 with torch.no_grad():
  for p in m.parameters(): n=p.numel();p.copy_(x[o:o+n].reshape_as(p));o+=n
def feat(m,b):
 z=bank_logits(m,b);return torch.cat((z['source'],z['counterfactual_red'],z['counterfactual_green'],z['clean_task'])).detach().double()
def run_one(method,seed):
 raw=json.loads(CFG_PATH.read_text());raw['methods']=[method];raw['candidate_methods']=[method];cfg=copy.deepcopy(validate_config(raw)); cfg['training']['steps']=STEP+1; cfg['training']['checkpoint_steps']=[STEP-1];
 data=build_task3_data(cfg,seed,data_root=ROOT/'data',download=bool(cfg['execution']['download_mnist']));world=build_smooth_world5(cfg,seed,data_root=ROOT/'data',download=False);banks=build_functional_banks(world,source_size_per_environment=int(cfg['banks']['source_bank_size_per_environment']),counterfactual_size=int(cfg['banks']['counterfactual_bank_size']))
 torch.manual_seed(seed);m=build_model_from_config(cfg);res=train_survey_method(model=m,source_envs=data.source_envs,batch_schedule=data.batch_schedule,method=method,config=cfg,seed=seed,initial_parameter_hash='');m.load_state_dict(res.checkpoint_state_dicts[STEP-1]);m.train();alg=get_algorithm(method,cfg);batches=scheduled_source_batches(data.source_envs,data.batch_schedule,STEP,cfg['device']); parts=alg.objective(m,batches,step=STEP); risk=parts.risk + float(cfg['training']['l2_regularizer_weight'])*sum(p.square().sum() for p in m.parameters()); pen=parts.penalty; params=[p for p in m.parameters()]; gr=torch.autograd.grad(risk,params,create_graph=True);gp=torch.autograd.grad(pen,params,create_graph=True);gB=torch.cat([x.reshape(-1) for x in gr]);gC=torch.cat([x.reshape(-1) for x in gp]);
 vecs=[]
 for j in range(KDIM):
  v=torch.randn_like(gB);v=v/(v.norm()+1e-12); vecs.append(v)
 V=torch.stack(vecs); B=V@gB; C=V@gC; Hc=torch.zeros((KDIM,KDIM));
 for j in range(KDIM):
  hv=torch.autograd.grad((gp[0]*vecs[j][:gp[0].numel()].reshape_as(gp[0])).sum() if len(gp)==1 else sum((a*b.reshape_as(a)).sum() for a,b in zip(gp,vecs[j].split([p.numel() for p in params]))),params,retain_graph=True)
  flat=torch.cat([a.reshape(-1) for a in hv]);Hc[:,j]=V@flat
 lam=float(parts.applied_penalty_weight)
 # risk Hessian projected via HVP
 Hr=torch.zeros((KDIM,KDIM))
 for j in range(KDIM):
  splits=vecs[j].split([p.numel() for p in params]);dot=sum((a*b.reshape_as(a)).sum() for a,b in zip(gr,splits));hv=torch.autograd.grad(dot,params,retain_graph=True);Hr[:,j]=V@torch.cat([a.reshape(-1) for a in hv])
 H=(Hr+Hr.T)/2;K=(Hc+Hc.T)/2;eye=1e-4*torch.eye(KDIM);f0=feat(m,banks);rows=[];responses={}; raw_ops={}
 for c_on in (0,1):
  for k_on in (0,1):
   M=H+k_on*lam*K+eye;rhs=B+c_on*lam*C; eig=torch.linalg.eigvalsh(M); cond=float((eig.abs().max()/(eig.abs().min()+1e-12))); d=-torch.linalg.solve(M,rhs)
   raw_ops[f'{c_on}{k_on}']=(M,rhs,d,eig,cond)
 maxnorm=max(float((V.T@q[2]).norm()) for q in raw_ops.values());
 for radius in (RHO, RHO/2.0, RHO/4.0):
  alpha=radius/max(maxnorm,1e-12); responses={}
  for key,(M,rhs,d,eig,cond) in raw_ops.items():
   raw_delta=V.T@d; theta=flatten_params(m); theta2=theta + alpha*raw_delta;branch=copy.deepcopy(m);set_flat(branch,theta2);r=feat(branch,banks)-f0;responses[key]=r;rows.append({'method':method,'seed':seed,'checkpoint':STEP,'radius':radius,'cell':key,'delta_norm':float((alpha*raw_delta).norm()),'raw_direction_norm':float(raw_delta.norm()),'common_alpha':alpha,'functional_response_norm':float(r.norm()),'source_response_norm':float(r[:len(banks.source)].norm()),'counterfactual_response_norm':float(r[len(banks.source):len(banks.source)+len(banks.counterfactual_red)+len(banks.counterfactual_green)].norm()),'clean_response_norm':float(r[-len(banks.clean_task):].norm()),'projected_condition_number':cond,'min_eigenvalue':float(eig.min()),'negative_eigenvalue_count':int((eig<0).sum()),'pd_gate':bool(eig.min()>1e-6),'linear_solve_residual':float((M@d+rhs).norm())})
  dC=responses['10']-responses['00'];dK=responses['01']-responses['00'];dI=responses['11']-responses['10']-responses['01']+responses['00'];
  for name,x in [('delta_C',dC),('delta_K',dK),('delta_interaction',dI)]: rows.append({'method':method,'seed':seed,'checkpoint':STEP,'radius':radius,'cell':name,'delta_norm':np.nan,'functional_response_norm':float(x.norm()),'source_response_norm':float(x[:len(banks.source)].norm()),'counterfactual_response_norm':float(x[len(banks.source):len(banks.source)+len(banks.counterfactual_red)+len(banks.counterfactual_green)].norm()),'clean_response_norm':float(x[-len(banks.clean_task):].norm())})
 return rows

def main():
 OUT.mkdir(parents=True,exist_ok=True);allr=[]
 for m in METHODS:
  for s in SMOKE_SEEDS: allr.extend(run_one(m,s))
 df=pd.DataFrame(allr);df.to_csv(OUT/'factorial_response_summary.csv',index=False);json.dump({'dataset':'ColoredMNIST','methods':list(METHODS),'seeds':list(SMOKE_SEEDS),'checkpoint':STEP,'subspace_dimension':KDIM,'epsilon':EPS,'trust_region_radius':RHO,'cells':['00','10','01','11'],'local_only_h1':True,'target_used':False,'interpretation':'C/K local operator pilot; no claim of independently realizable algorithms'},open(OUT/'provenance.json','w'),indent=2);df.to_csv(OUT/'report.csv',index=False)
 print(df[df.cell.isin(['delta_C','delta_K','delta_interaction'])].to_string(index=False))
if __name__=='__main__':main()
