"""A/O versus full semantic-response geometry for held-out finite continuations."""
from pathlib import Path
import copy,json,hashlib,subprocess
import numpy as np,pandas as pd,torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from .task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from .task3_aopi_multimethod_mechanism_survey.smooth_world5 import build_smooth_world5
from .task3_aopi_multimethod_mechanism_survey.functional_banks import build_functional_banks,bank_logits
from .task3_aopi_multimethod_mechanism_survey.full_response import _run_path
from .task3_aopi_multimethod_mechanism_survey.task_response import task_geometry
from .task3_aopi_multimethod_mechanism_survey.source_observation import observation_geometry
from .task3_aopi_multimethod_mechanism_survey.method_trainer import train_survey_method
from .task3_cmnist_cpu_minimal.data import build_task3_data
from .task3_cmnist_cpu_minimal.model import build_model_from_config
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'round3_redesign/semantic_mechanism_bridge';CFG=ROOT/'configs/task3_aopi_multimethod_mechanism_survey.json';METHODS=('ERM','CORAL','IRMv1','VREX','FISHR');SEEDS=(10,11,12,13,14);STEP=300;H=20;EPS=0.01;ALPHA=0.1;BASIS=(0,1,3);NAMES=('source_env0_color','source_env1_color','source_label_noise')
def run():
 raw=json.loads(CFG.read_text());raw['methods']=list(METHODS);raw['candidate_methods']=list(METHODS);raw['seeds']=list(SEEDS);raw['training']['checkpoint_steps']=[STEP];cfg=validate_config(raw);cfg=copy.deepcopy(cfg);rows=[];feature_rows=[];arrays={}
 for seed in SEEDS:
  data=build_task3_data(cfg,seed,data_root=ROOT/'data',download=bool(cfg['execution']['download_mnist']));world=build_smooth_world5(cfg,seed,data_root=ROOT/'data',download=False);banks=build_functional_banks(world,source_size_per_environment=int(cfg['banks']['source_bank_size_per_environment']),counterfactual_size=int(cfg['banks']['counterfactual_bank_size']))
  gen=torch.Generator().manual_seed(seed+271828);schedule=(torch.randint(len(world.source[0].digits),(H,int(cfg['training']['batch_size_per_environment'])),generator=gen),torch.randint(len(world.source[1].digits),(H,int(cfg['training']['batch_size_per_environment'])),generator=gen));zero=torch.zeros(5,dtype=torch.double)
  for method in METHODS:
   torch.manual_seed(seed);m=build_model_from_config(cfg);res=train_survey_method(model=m,source_envs=data.source_envs,batch_schedule=data.batch_schedule,method=method,config=cfg,seed=seed,initial_parameter_hash='');
   if not res.finite: raise RuntimeError(res.invalid_reason)
   state=copy.deepcopy(m.state_dict()); ck=build_model_from_config(cfg);ck.load_state_dict(res.checkpoint_state_dicts[STEP]);ck.eval();tg=task_geometry(ck,world,pool_size=int(cfg['banks']['geometry_pool_size_per_pool']));og=observation_geometry(ck,world,pool_size=int(cfg['banks']['geometry_pool_size_per_pool']));control=_run_path(res,world,banks,zero,method,cfg,schedule,(0,H))[H][3];
   rs=[]
   for basis,name in zip(BASIS,NAMES):
    d=torch.zeros(5,dtype=torch.double);d[basis]=1.;plus=_run_path(res,world,banks,EPS*d,method,cfg,schedule,(0,H));minus=_run_path(res,world,banks,-EPS*d,method,cfg,schedule,(0,H));r0={k:((plus[H][3][k]-minus[H][3][k])/(2*EPS)).detach() for k in control};flat=torch.cat([r0[k] for k in ('source','counterfactual_red','counterfactual_green','clean_task')]);rs.append(flat);arrays[f'{method}__{seed}__{name}']=flat.numpy();
   R=torch.stack([x.reshape(-1) for x in rs],dim=0);G=R@R.T;A=tg.A[:,BASIS];O=og.O[:,BASIS];
   # Independent outcome: finite-shift continuation response relative to zero-shift continuation.
   for j,(basis,name) in enumerate(zip(BASIS,NAMES)):
    d=torch.zeros(5,dtype=torch.double);d[basis]=1.;shift=_run_path(res,world,banks,ALPHA*d,method,cfg,schedule,(0,H))[H][3];out=torch.cat([(shift[k]-control[k]).detach() for k in ('source','counterfactual_red','counterfactual_green','clean_task')]);
    rows.append({'method':method,'seed':seed,'semantic_direction':name,'horizon':H,'finite_alpha':ALPHA,'outcome_norm':float(out.norm()),'outcome_source_norm':float(out[:len(banks.source)].norm()),'outcome_counterfactual_norm':float(out[len(banks.source):len(banks.source)+len(banks.counterfactual_red)+len(banks.counterfactual_green)].norm()),'outcome_clean_norm':float(out[-len(banks.clean_task):].norm()),'A_norm':float(A[:,j].norm()),'O_norm':float(O[:,j].norm()),'R_norm':float(R[j].norm()),'A_rank':tg.A.shape[1],'O_rank':og.rank})
    for q in range(3): rows[-1][f'G_{j}_{q}']=float(G[j,q])
   # compact predictors per method/seed: A/O norms + Gram entries and response norms
   for j,name in enumerate(NAMES): feature_rows.append({'method':method,'seed':seed,'semantic_direction':name,'A_norm':float(A[:,j].norm()),'O_norm':float(O[:,j].norm()),'R_norm':float(R[j].norm()),'G_diag':float(G[j,j]),'G_cross_01':float(G[0,1]),'G_cross_03':float(G[0,2]),'G_cross_13':float(G[1,2]),'R_source_norm':float(R[j,:len(banks.source)].norm()),'R_counterfactual_norm':float(R[j,len(banks.source):len(banks.source)+len(banks.counterfactual_red)+len(banks.counterfactual_green)].norm()),'R_clean_norm':float(R[j,-len(banks.clean_task):].norm())})
 np.savez_compressed(OUT/'aopi_semantic_response_vectors.npz',**arrays);pd.DataFrame(rows).to_csv(OUT/'aopi_semantic_finite_outcomes.csv',index=False);pd.DataFrame(feature_rows).to_csv(OUT/'aopi_semantic_predictor_features.csv',index=False)
 out=pd.DataFrame(rows);feat=pd.DataFrame(feature_rows);audit=[]
 for direction in NAMES:
  z=out[out.semantic_direction==direction].merge(feat,on=['method','seed','semantic_direction'],suffixes=('','_feat'));y=z.outcome_norm.values
  for label,cols in [('A_only',['A_norm']),('AO',['A_norm','O_norm']),('AO_R',['A_norm','O_norm','R_norm','G_diag','G_cross_01','G_cross_03','G_cross_13','R_source_norm','R_counterfactual_norm','R_clean_norm'])]:
   pred=np.zeros(len(z));
   for method in z.method.unique():
    tr=z.method!=method;te=~tr;model=Ridge(alpha=1e-8).fit(z.loc[tr,cols],y[tr]);pred[te]=model.predict(z.loc[te,cols])
   audit.append({'semantic_direction':direction,'model':label,'n':len(z),'methods':z.method.nunique(),'lo_method_rmse':float(np.sqrt(mean_squared_error(y,pred))),'outcome_sd':float(y.std())})
 pd.DataFrame(audit).to_csv(OUT/'aopi_semantic_incremental_predictive_audit.csv',index=False)
 report=['# A/O-conditioned semantic predictive audit','',f'Five methods and five seeds; checkpoint {STEP}, finite shift alpha={ALPHA}, continuation horizon {H}. Predictors are source-defined A/O norms or full response geometry (norms, Gram entries and bank allocation). Outcome is the independently rolled finite-shift continuation displacement, not an algebraic multiple of the local response derivative.','', 'Leave-one-method-out results:']
 report += [pd.DataFrame(audit).to_string(index=False), '', 'Interpretation: the full response model is only evidence beyond A/O if its leave-one-method-out RMSE improves over AO and the improvement is not caused by the outcome construction. BIRM/LoRA-BIRM remain a separate static head-only geometry layer.']
 (OUT/'aopi_semantic_predictive_report.md').write_text('\n'.join(report));prov={'methods':list(METHODS),'seeds':list(SEEDS),'checkpoint':STEP,'horizon':H,'finite_alpha':ALPHA,'predictors':['A_norm','O_norm','full_response_norms','response_Gram','bank_allocation'],'outcome':'finite-shift continuation displacement relative to zero-shift continuation','target_used_for_training':False,'target_used_for_feature_selection':False,'birm_boundary':'separate static head-only artifact','status':'PREDICTIVE_AUDIT'};(OUT/'aopi_semantic_predictive_provenance.json').write_text(json.dumps(prov,indent=2));print(pd.DataFrame(audit).to_string(index=False))
if __name__=='__main__':run()
