"""Leakage-free lagged response-descriptor discovery pilot.

This is intentionally *not* called causal mechanism discovery.  Features at
step t are available before update t+1 (one-step lag); target is never read.
A group sparse multi-task model ranks whole descriptors, with fold-local scaling.
"""
from pathlib import Path
import numpy as np,pandas as pd,json
from sklearn.linear_model import MultiTaskLasso
BASE=Path('/Users/sunlay/Desktop/ood-representation-regularization/round3_redesign')
DIRS={'VREX':'vrex_trajectory_cmnist_5seed','IRMv1':'irm_trajectory_cmnist_5seed','FISHR':'fishr_trajectory_cmnist_5seed'}
ATOMS=['parameter_update_norm','functional_update_norm','source_functional_update_norm','counterfactual_functional_update_norm','clean_functional_update_norm','counterfactual_source_ratio','partition_entropy','update_cosine_prev','functional_update_cosine_prev']
def run(alpha=0.001):
 out=BASE/'method_agnostic_mechanism';out.mkdir(exist_ok=True); rows=[];selected=[]
 for meth,dd in DIRS.items():
  u=pd.read_csv(BASE/dd/'trajectory_updates.csv'); y=np.load(BASE/dd/'functional_update_vectors.npy');
  # strict temporal lag: descriptors at t predict response at t+1, per seed
  X=[];Y=[];S=[]
  for s in sorted(u.seed.unique()):
   ix=np.where(u.seed.to_numpy()==s)[0]; X.append(u.iloc[ix][ATOMS].fillna(0).to_numpy(float)[:-1]);Y.append(y[ix][1:]);S += [s]*(len(ix)-1)
  X=np.vstack(X);Y=np.vstack(Y);S=np.asarray(S)
  for held in sorted(set(S)):
   tr=S!=held;te=~tr; mu=X[tr].mean(0);sd=X[tr].std(0)+1e-8; xt=(X-mu)/sd
   model=MultiTaskLasso(alpha=alpha,max_iter=300).fit(xt[tr],Y[tr]);pred=model.predict(xt[te]);base=np.sum((Y[te]-Y[tr].mean(0))**2);err=np.sum((Y[te]-pred)**2)
   norms=np.linalg.norm(model.coef_,axis=0); order=np.argsort(norms)[::-1]
   rows.append({'method':meth,'heldout_seed':int(held),'heldout_vector_r2':float(1-err/(base+1e-12)),'train_rows':int(tr.sum()),'test_rows':int(te.sum()),'alpha':alpha})
   for rank,j in enumerate(order):
    if norms[j]>1e-10:selected.append({'method':meth,'heldout_seed':int(held),'atom':ATOMS[j],'group_norm':float(norms[j]),'rank':rank})
 pd.DataFrame(rows).to_csv(out/'mechanism_atom_discovery_cv.csv',index=False);pd.DataFrame(selected).to_csv(out/'mechanism_atom_selected_atoms.csv',index=False)
 (out/'mechanism_atom_discovery_report.md').write_text('# Leakage-free lagged response-descriptor pilot\n\nAt step t, descriptors are lagged to predict the next functional update at t+1. Scaling is fitted within each held-out-seed training fold. MultiTaskLasso imposes group sparsity across the full response vector. This remains a descriptor-predictability experiment, not causal mechanism identification; matched interventions are required.\n\n'+ '\n'.join(f'- {m}: held-out vector R2={g.heldout_vector_r2.mean():.4f}±{g.heldout_vector_r2.std():.4f}.' for m,g in pd.DataFrame(rows).groupby('method'))+'\n')
 json.dump({'dataset':'ColoredMNIST','response':'functional_update_t_plus_1','features':'descriptor_at_t','no_same_step_response':True,'fold_local_standardization':True,'model':'MultiTaskLasso_group_sparse','target_used':False,'status':'descriptor_predictability_not_mechanism','next_step':'pre_update_state_atoms_and_held_out_interventions'},open(out/'mechanism_atom_discovery_provenance.json','w'),indent=2)
 print(pd.DataFrame(rows).groupby('method').heldout_vector_r2.agg(['mean','std']).to_string())
if __name__=='__main__':run()
