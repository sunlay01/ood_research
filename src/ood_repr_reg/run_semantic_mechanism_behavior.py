"""Test whether local semantic response geometry predicts mechanism-specific finite behavior.

The response is a model-level finite difference to the data-generating semantic
coordinate; outcomes are held-out finite shifts of that same coordinate. This
keeps BIRM/LoRA-BIRM comparable without pretending their head-only continuation
state is identical to the survey runner's optimizer state.
"""
from pathlib import Path
import csv, glob, importlib.util, json, subprocess
import numpy as np, pandas as pd, torch
import sys
from .task3_aopi_multimethod_mechanism_survey.smooth_world5 import build_smooth_world5, subset_pool, environment_parameters, outcome_weight
from .task3_aopi_multimethod_mechanism_survey.functional_banks import build_functional_banks
from .task3_cmnist_cpu_minimal.data import build_task3_data, colorize_downsampled
from .task3_cmnist_cpu_minimal.model import build_model_from_config
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'round3_redesign/semantic_mechanism_bridge'; CFG=ROOT/'configs/task3_aopi_multimethod_mechanism_survey.json'; SEEDS=(10,11,12,13,14); METHODS=('ERM','CORAL','IRMv1','VREX','FISHR','Full-BIRM (Official)','LoRA-BIRM (Official)'); EPS=0.01; ALPHAS=(0.02,0.05,0.1); DIRECTION_NAMES=('source_color_common','source_color_contrast','source_label_noise'); DIRECTION_VECS=(np.array([1,1,0,0,0])/np.sqrt(2),np.array([1,-1,0,0,0])/np.sqrt(2),np.array([0,0,0,1,0]))

def _official(path):
 spec=importlib.util.spec_from_file_location('dense_birm','/Users/sunlay/Desktop/lora_birm/cmnist/run_cmnist_official_no_torchvision.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m); return m
sys.path.insert(0,'/Users/sunlay/Desktop/lora_birm/cmnist'); DENSE=_official(None)
LO_SPEC=importlib.util.spec_from_file_location('lora_birm','/Users/sunlay/Desktop/lora_birm/cmnist/run_lora_birm_mlp_svd.py');LORA=importlib.util.module_from_spec(LO_SPEC);LO_SPEC.loader.exec_module(LORA)

def load_model(method,seed):
 if method=='Full-BIRM (Official)':
  p=ROOT/f'round3_redesign/birm_cmnist_checkpoint_rerun/birm/cmnist_birm_official_checkpoints/seed_{seed}_full_birm_official_step_501.pt'; x=torch.load(p,map_location='cpu',weights_only=False);m=DENSE.StandardMLP();m.load_state_dict(x['model_state']);return m.eval()
 if method=='LoRA-BIRM (Official)':
  p=ROOT/f'round3_redesign/birm_cmnist_checkpoint_rerun/lora-birm/cmnist_lora-birm_official_checkpoints/seed_{seed}_lora_birm_step_501.pt';x=torch.load(p,map_location='cpu',weights_only=False);c=x['model_config'];m=LORA.LoRABIRMMLP(int(c['hidden_dim']),int(c['rank']),float(c['lora_alpha']),bool(c['train_base']),bool(c['train_bias']));m.load_state_dict(x['model_state']);return m.eval()
 p=ROOT/f'round3_redesign/task3_aopi_multimethod_mechanism_survey/results/checkpoints/seed_{seed}_{method}.pt';x=torch.load(p,map_location='cpu',weights_only=False);m=build_model_from_config(json.loads(CFG.read_text()));m.load_state_dict(x['state_dict']);return m.eval()

def logits(m,x):
 with torch.no_grad(): return m(x, sample=False).reshape(-1) if 'LoRA' in type(m).__name__ else m(x).reshape(-1)
def shifted_bank(world,delta,size=512):
 pool=subset_pool(world.evaluation[0],size); rows=[]
 for env in (0,1):
  p,q=environment_parameters(torch.tensor(delta,dtype=torch.double),environment=env,evaluation=False)
  for lf in (0,1):
   for cf in (0,1):
    label=(pool.digits<5).double(); label=(label-float(lf)).abs(); color=(label-float(cf)).abs(); rows.append((_colored(pool,color),label,outcome_weight(p,q,lf,cf)))
 return rows

def _colored(pool,color): return colorize_downsampled(pool.images,color.float(),image_subsample=2,normalize_pixels=True)
def expected(m,world,delta):
 rows=shifted_bank(world,delta); acc=loss=0.;
 for x,y,w in rows:
  z=logits(m,x); ww=float(w.detach()); acc += ww*float(((z>0)==y.reshape(-1)).double().mean()); loss += ww*float(torch.nn.functional.binary_cross_entropy_with_logits(z,y.reshape(-1)))
 return acc/2.,loss/2.
def run():
 raw=json.loads(CFG.read_text());cfg=raw; rows=[]; perf=pd.read_csv(ROOT/'round3_redesign/task3_aopi_multimethod_mechanism_survey/results/method_performance.csv')
 for seed in SEEDS:
  world=build_smooth_world5(cfg,seed,data_root=ROOT/'data',download=False)
  for method in METHODS:
   m=load_model(method,seed); base=np.zeros(5); base_acc,base_loss=expected(m,world,base)
   for name,vec in zip(DIRECTION_NAMES,DIRECTION_VECS):
    plus=expected(m,world,EPS*vec);minus=expected(m,world,-EPS*vec); local=np.array([(plus[0]-minus[0])/(2*EPS),(plus[1]-minus[1])/(2*EPS)])
    for a in ALPHAS:
     acc,loss=expected(m,world,a*vec); rows.append({'method':method,'seed':seed,'semantic_direction':name,'alpha':a,'local_response_acc_slope':local[0],'local_response_loss_slope':local[1],'finite_accuracy':acc,'finite_loss':loss,'accuracy_change':acc-base_acc,'loss_change':loss-base_loss,'base_accuracy':base_acc,'base_loss':base_loss,'target_posthoc':float(perf[(perf.seed==seed)&(perf.method==method)].target_acc.iloc[0]) if method in set(perf.method) else np.nan,'source_only_geometry':True})
 out=pd.DataFrame(rows);out.to_csv(OUT/'mechanism_specific_finite_behavior.csv',index=False)
 agg=out.groupby(['method','semantic_direction','alpha'])[['accuracy_change','loss_change','finite_accuracy']].agg(['mean','std']).reset_index();agg.to_csv(OUT/'mechanism_specific_finite_behavior_summary.csv',index=False)
 report=['# Mechanism-specific finite behavior audit','','Local response is estimated with epsilon=0.01 on fixed CMNIST semantic coordinates. Outcomes are evaluated at held-out finite shifts alpha in {0.1,0.2,0.3}; target accuracy is post-hoc metadata only.','', 'The response geometry is evaluated per semantic mechanism, rather than against one pooled target-accuracy scalar.']
 for name in DIRECTION_NAMES:
  q=agg[(agg.semantic_direction==name)&(agg.alpha==0.3)];report += [f'\n## {name}\n',q[['method','accuracy_change','loss_change']].round(4).to_string(index=False)]
 report += ['\n## Verdict\n','This first behavior audit reports a mechanism-specific state representation test. BIRM and LoRA-BIRM are included using their official final checkpoints. A positive mechanism claim requires cross-method prediction that beats method identity and a held-out intervention; this artifact does not grant that claim automatically.']
 (OUT/'mechanism_specific_finite_behavior_report.md').write_text('\n'.join(report))
 prov={'methods':list(METHODS),'seeds':list(SEEDS),'directions':list(DIRECTION_NAMES),'epsilon':EPS,'alphas':list(ALPHAS),'target_used_for':'posthoc metadata only','birm_interface':'official final checkpoint; no optimizer continuation state','geometry_definition':'fixed-model finite difference under semantic data-generating perturbation','verdict':'PENDING_CROSS_METHOD_PREDICTIVE_AUDIT'};(OUT/'behavior_provenance.json').write_text(json.dumps(prov,indent=2));print('rows',len(out))
if __name__=='__main__':run()
