"""Local C/K factorial response intervention on ColoredMNIST.

At a common step-300 state, estimate risk gradient B, penalty gradient C and
penalty curvature K in a small random parameter subspace.  Four local steps are
then applied and functional responses measured. This is a local intervention
pilot; it is not a claim that C/K are independently realizable algorithms.
"""
from pathlib import Path
import copy,json
from dataclasses import replace
import numpy as np,pandas as pd,torch
from torch import nn
from .task3_aopi_multimethod_mechanism_survey.algorithms.registry import get_algorithm
from .task3_aopi_multimethod_mechanism_survey.functional_banks import bank_logits,build_functional_banks
from .task3_aopi_multimethod_mechanism_survey.smooth_world5 import build_smooth_world5
from .task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from .task3_cmnist_cpu_minimal.data import build_task3_data,scheduled_source_batches
from .task3_cmnist_cpu_minimal.model import build_model_from_config
from .task3_aopi_multimethod_mechanism_survey.method_trainer import train_survey_method
ROOT=Path(__file__).resolve().parents[2];CFG_PATH=ROOT/'configs/task3_aopi_multimethod_mechanism_survey.json';OUT=ROOT/'round3_redesign/method_agnostic_mechanism/ck_factorial';METHODS=('IRMv1','VREX','FISHR');SEEDS=(10,11,12);STEP=300;KDIM=8;EPS=0.001;RHO=0.001;SMOKE_SEEDS=(10,)
LINEARITY_TOL=0.25;COND_MAX=1e6;PD_THRESH=1e-6;RESIDUAL_TOL=1e-6
MODEL_VARIANT='minimal'

class WideColoredMNISTMLP(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(392,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU());self.head=nn.Linear(256,1)
  for q in self.modules():
   if isinstance(q,nn.Linear): nn.init.xavier_uniform_(q.weight);nn.init.zeros_(q.bias)
 def encode(self,x): return self.encoder(x.reshape(x.shape[0],-1))
 def forward(self,x): return self.head(self.encode(x))

def build_factorial_model(cfg): return WideColoredMNISTMLP() if MODEL_VARIANT=='wide256x3' else build_model_from_config(cfg)

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
 torch.manual_seed(seed);m=build_factorial_model(cfg);res=train_survey_method(model=m,source_envs=data.source_envs,batch_schedule=data.batch_schedule,method=method,config=cfg,seed=seed,initial_parameter_hash='');m.load_state_dict(res.checkpoint_state_dicts[STEP-1]);m=m.double();m.train();banks=replace(banks,**{k:getattr(banks,k).double() if getattr(banks,k).is_floating_point() else getattr(banks,k) for k in ('source','counterfactual_red','counterfactual_green','clean_task')});alg=get_algorithm(method,cfg);batches=scheduled_source_batches(data.source_envs,data.batch_schedule,STEP,cfg['device']);batches=tuple((x.double(),y) for x,y in batches); parts=alg.objective(m,batches,step=STEP); risk=parts.risk + float(cfg['training']['l2_regularizer_weight'])*sum(p.square().sum() for p in m.parameters()); pen=parts.penalty; params=[p for p in m.parameters()]; gr=torch.autograd.grad(risk,params,create_graph=True);gp=torch.autograd.grad(pen,params,create_graph=True);gB=torch.cat([x.reshape(-1) for x in gr]);gC=torch.cat([x.reshape(-1) for x in gp]);
 Q,_=torch.linalg.qr(torch.randn(gB.numel(),KDIM,dtype=torch.float64),mode='reduced'); V=Q.T; vecs=[V[j] for j in range(KDIM)]; B=V@gB; C=V@gC; Hc=torch.zeros((KDIM,KDIM),dtype=torch.float64);
 for j in range(KDIM):
  hv=torch.autograd.grad((gp[0]*vecs[j][:gp[0].numel()].reshape_as(gp[0])).sum() if len(gp)==1 else sum((a*b.reshape_as(a)).sum() for a,b in zip(gp,vecs[j].split([p.numel() for p in params]))),params,retain_graph=True)
  flat=torch.cat([a.reshape(-1) for a in hv]);Hc[:,j]=V@flat
 lam=float(parts.applied_penalty_weight)
 # risk Hessian projected via HVP
 Hr=torch.zeros((KDIM,KDIM),dtype=torch.float64)
 for j in range(KDIM):
  splits=vecs[j].split([p.numel() for p in params]);dot=sum((a*b.reshape_as(a)).sum() for a,b in zip(gr,splits));hv=torch.autograd.grad(dot,params,retain_graph=True);Hr[:,j]=V@torch.cat([a.reshape(-1) for a in hv])
 H=(Hr+Hr.T)/2;K=(Hc+Hc.T)/2;eye=1e-4*torch.eye(KDIM);f0=feat(m,banks);rows=[];responses={}; raw_ops={}
 for c_on in (0,1):
  for k_on in (0,1):
   M=H+k_on*lam*K+eye;rhs=B+c_on*lam*C; eig=torch.linalg.eigvalsh(M); cond=float((eig.abs().max()/(eig.abs().min()+1e-12))); d=-torch.linalg.solve(M,rhs)
   raw_ops[f'{c_on}{k_on}']=(M,rhs,d,eig,cond)
 maxnorm=max(float((V.T@q[2]).norm().detach()) for q in raw_ops.values());
 radii=(RHO, RHO/2.0, RHO/4.0); response_by_radius={}; base_meta={}
 for radius in radii:
  alpha=radius/max(maxnorm,1e-12); responses={}
  for key,(M,rhs,d,eig,cond) in raw_ops.items():
   raw_delta=V.T@d; theta=flatten_params(m); theta2=theta + alpha*raw_delta;branch=copy.deepcopy(m);set_flat(branch,theta2);r=feat(branch,banks)-f0;responses[key]=r
   base_meta[(radius,key)]={'method':method,'seed':seed,'checkpoint':STEP,'radius':radius,'cell':key,'delta_norm':float((alpha*raw_delta).norm()),'raw_direction_norm':float(raw_delta.norm()),'common_alpha':alpha,'functional_response_norm':float(r.norm()),'source_response_norm':float(r[:len(banks.source)].norm()),'counterfactual_response_norm':float(r[len(banks.source):len(banks.source)+len(banks.counterfactual_red)+len(banks.counterfactual_green)].norm()),'clean_response_norm':float(r[-len(banks.clean_task):].norm()),'projected_condition_number':cond,'min_eigenvalue':float(eig.min()),'negative_eigenvalue_count':int((eig<0).sum()),'pd_gate':bool(eig.min()>PD_THRESH),'linear_solve_residual':float((M@d+rhs).norm())}
  response_by_radius[radius]=responses
 # Radius-scaling errors are attached to each base cell.  They compare vectors,
 # rather than only norms, and therefore detect directional nonlinearity too.
 for key in raw_ops:
  r1,r2,r4=response_by_radius[RHO][key],response_by_radius[RHO/2.0][key],response_by_radius[RHO/4.0][key]
  e2=float((r2-0.5*r1).norm()/max(0.5*float(r1.norm()),1e-12));e4=float((r4-0.25*r1).norm()/max(0.25*float(r1.norm()),1e-12))
  linearity=bool(np.isfinite(e2) and np.isfinite(e4) and e2<=LINEARITY_TOL and e4<=LINEARITY_TOL)
  for radius in radii:
   meta=base_meta[(radius,key)];meta.update({'radius_error_half':e2,'radius_error_quarter':e4,'linearity_gate':linearity,'resolution_gate':bool(meta['pd_gate'] and meta['projected_condition_number']<=COND_MAX and meta['linear_solve_residual']<=RESIDUAL_TOL and np.isfinite(meta['functional_response_norm']))});rows.append(meta)
 for radius in radii:
  responses=response_by_radius[radius]
  dC=responses['10']-responses['00'];dK=responses['01']-responses['00'];dI=responses['11']-responses['10']-responses['01']+responses['00'];
  validity={k: bool(base_meta[(radius,k)]['resolution_gate'] and base_meta[(radius,k)]['linearity_gate']) for k in raw_ops}
  derived_valid=bool(all(validity.values()))
  for name,x,required in [('delta_C',dC,('10','00')),('delta_K',dK,('01','00')),('delta_interaction',dI,('11','10','01','00'))]:
   def cosine(a,b): return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
   rows.append({'method':method,'seed':seed,'checkpoint':STEP,'radius':radius,'cell':name,'delta_norm':np.nan,'functional_response_norm':float(x.norm()),'source_response_norm':float(x[:len(banks.source)].norm()),'counterfactual_response_norm':float(x[len(banks.source):len(banks.source)+len(banks.counterfactual_red)+len(banks.counterfactual_green)].norm()),'clean_response_norm':float(x[-len(banks.clean_task):].norm()),'cosine_to_R00':cosine(x,responses['00']),'cosine_to_C':cosine(x,dC),'cosine_to_K':cosine(x,dK),'pd_gate':bool(all(base_meta[(radius,k)]['pd_gate'] for k in required)),'linearity_gate':bool(all(base_meta[(radius,k)]['linearity_gate'] for k in required)),'resolution_gate':bool(all(base_meta[(radius,k)]['resolution_gate'] for k in required)),'derived_valid':bool(all(validity[k] for k in required)),'required_cells':','.join(required)})
 return rows

def main():
 OUT.mkdir(parents=True,exist_ok=True);allr=[]
 for m in METHODS:
  for s in SMOKE_SEEDS: allr.extend(run_one(m,s))
 df=pd.DataFrame(allr);df.to_csv(OUT/'factorial_response_summary.csv',index=False);json.dump({'dataset':'ColoredMNIST','methods':list(METHODS),'seeds':list(SMOKE_SEEDS),'checkpoint':STEP,'subspace_dimension':KDIM,'epsilon':EPS,'trust_region_radius':RHO,'radius_grid':[RHO,RHO/2.0,RHO/4.0],'cells':['00','10','01','11'],'local_only_h1':True,'target_used':False,'orthonormal_subspace':True,'dtype':'float64','common_alpha':True,'linearity_tolerance':LINEARITY_TOL,'condition_threshold':COND_MAX,'pd_threshold':PD_THRESH,'residual_threshold':RESIDUAL_TOL,'derived_contrast_validity_propagated':True,'interpretation':'C/K local operator pilot; no claim of independently realizable algorithms'},open(OUT/'provenance.json','w'),indent=2);df.to_csv(OUT/'report.csv',index=False)
 print(df[df.cell.isin(['delta_C','delta_K','delta_interaction'])].to_string(index=False))
if __name__=='__main__':main()
