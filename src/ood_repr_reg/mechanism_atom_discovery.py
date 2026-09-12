"""Data-first mechanism-atom discovery pilot for CMNIST.

Builds a predeclared neutral atom library from source-side state summaries and
uses blocked (seed-held-out) sparse multivariate regression to predict the next
high-dimensional functional update. This is discovery, not causal identification;
causal status requires matched fork interventions.
"""
from pathlib import Path
import numpy as np,pandas as pd,json
from sklearn.linear_model import Ridge
BASE=Path('/Users/sunlay/Desktop/ood-representation-regularization/round3_redesign')
DIRS={'VREX':'vrex_trajectory_cmnist_5seed','IRMv1':'irm_trajectory_cmnist_5seed','FISHR':'fishr_trajectory_cmnist_5seed'}
ATOMS=['parameter_update_norm','functional_update_norm','source_functional_update_norm','counterfactual_functional_update_norm','clean_functional_update_norm','counterfactual_source_ratio','partition_entropy','update_cosine_prev','functional_update_cosine_prev']
def run():
 out=BASE/'method_agnostic_mechanism';out.mkdir(exist_ok=True); rows=[]; selected=[]
 for meth,dd in DIRS.items():
  d=BASE/dd; u=pd.read_csv(d/'trajectory_updates.csv'); y=np.load(d/'functional_update_vectors.npy')
  X=u[ATOMS].fillna(0).to_numpy(float); X=(X-X.mean(0))/(X.std(0)+1e-8)
  seeds=sorted(u.seed.unique())
  for held in seeds:
   tr=u.seed.to_numpy()!=held; te=~tr
   model=Ridge(alpha=10.0).fit(X[tr],y[tr]); pred=model.predict(X[te]);
   base=np.sum((y[te]-y[tr].mean(0))**2); err=np.sum((y[te]-pred)**2)
   r2=1-err/(base+1e-12); norms=np.linalg.norm(model.coef_,axis=0); order=np.argsort(norms)[::-1]
   rows.append({'method':meth,'heldout_seed':int(held),'heldout_vector_r2':float(r2),'train_rows':int(tr.sum()),'test_rows':int(te.sum())})
   for j in order[:5]: selected.append({'method':meth,'heldout_seed':int(held),'atom':ATOMS[j],'coefficient_group_norm':float(norms[j]),'rank':int(np.where(order==j)[0][0])})
 pd.DataFrame(rows).to_csv(out/'mechanism_atom_discovery_cv.csv',index=False);pd.DataFrame(selected).to_csv(out/'mechanism_atom_selected_atoms.csv',index=False)
 report=out/'mechanism_atom_discovery_report.md';
 with report.open('w') as f:
  f.write('# Data-first mechanism atom discovery pilot\n\n')
  f.write('A neutral atom library was fixed before reading target outcomes. Atoms are source-side checkpoint summaries; the response is the 2048-dimensional next functional update. Ridge/SINDy-style group ranking is evaluated by held-out seed.\n\n')
  for m,g in pd.DataFrame(rows).groupby('method'): f.write(f"- {m}: held-out vector R2 mean={g.heldout_vector_r2.mean():.4f}, std={g.heldout_vector_r2.std():.4f}.\n")
  f.write('\nThis pilot tests generative predictability of observed response, not causality. Selected atoms are method-conditioned and do not establish forcing/filtering. The next decisive step is held-out intervention prediction followed by matched fork knockout/swap.\n')
 json.dump({'dataset':'ColoredMNIST','response':'next_functional_update_vector','atom_library':ATOMS,'cv':'leave_one_seed_out','target_used':False,'causal_identification':False,'next_step':'held_out_intervention_and_matched_fork'},open(out/'mechanism_atom_discovery_provenance.json','w'),indent=2)
 print(pd.DataFrame(rows).groupby('method').heldout_vector_r2.agg(['mean','std']).to_string())
if __name__=='__main__':run()
