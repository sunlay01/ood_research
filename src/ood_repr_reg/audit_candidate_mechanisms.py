from pathlib import Path
import pandas as pd,numpy as np,json
BASE=Path('/Users/sunlay/Desktop/ood-representation-regularization/round3_redesign')
DIRS={'VREX':'vrex_trajectory_microscope','IRMv1':'irm_trajectory_cmnist','FISHR':'fishr_trajectory_cmnist'}
T={'VREX':{11:.4815,12:.6178,13:.4847},'IRMv1':{11:.6631,12:.6725,13:.6894},'FISHR':{11:.5765,12:.5576,13:.5199}}
W={'early':(1,100),'middle':(101,300),'late':(301,500)}
rows=[]
for meth,dd in DIRS.items():
 d=BASE/dd;u=pd.read_csv(d/'trajectory_updates.csv');v=np.load(d/'functional_update_vectors.npy')
 for s in sorted(u.seed.unique()):
  ix=np.where(u.seed.to_numpy()==s)[0]; uu=u.iloc[ix].reset_index(drop=True); vv=v[ix]
  for wn,(lo,hi) in W.items():
   m=(uu.step>=lo)&(uu.step<=hi); x=vv[m.to_numpy()]; n=np.linalg.norm(x,axis=1); path=n.sum(); net=np.linalg.norm(x.sum(0));
   cos=(np.sum(x[1:]*x[:-1],1)/(np.linalg.norm(x[1:],axis=1)*np.linalg.norm(x[:-1],axis=1)+1e-12)) if len(x)>1 else []
   # source-observable transfer proxy: clean movement / source movement, and counterfactual/source ratio
   rows.append(dict(method=meth,seed=s,window=wn,target_acc=T[meth][s],path_length=path,net_displacement=net,cancellation_efficiency=net/(path+1e-12),mean_adjacent_cosine=np.mean(cos),negative_cosine_fraction=np.mean(np.array(cos)<0),clean_to_source=float(uu.loc[m,'clean_functional_update_norm'].sum()/(uu.loc[m,'source_functional_update_norm'].sum()+1e-12)),cf_source_ratio=float(uu.loc[m,'counterfactual_source_ratio'].mean()),entropy=float(uu.loc[m,'partition_entropy'].mean())))
f=pd.DataFrame(rows);out=BASE/'method_agnostic_mechanism';f.to_csv(out/'candidate_mechanism_audit.csv',index=False)
# rank agreement and counterexamples by method/window
res=[]
for (meth,w),g in f.groupby(['method','window']):
 for metric,ori in [('cancellation_efficiency',1),('mean_adjacent_cosine',1),('clean_to_source',1),('cf_source_ratio',-1)]:
  pairs=correct=0; ce=[]
  z=g.reset_index(drop=True)
  for i in range(len(z)):
   for j in range(i+1,len(z)):
    mr=np.sign(ori*(z.loc[i,metric]-z.loc[j,metric])); tr=np.sign(z.loc[i,'target_acc']-z.loc[j,'target_acc'])
    if mr and tr:
     pairs+=1;correct+=mr==tr
     if mr!=tr: ce.append((int(z.loc[i,'seed']),int(z.loc[j,'seed'])))
  res.append(dict(method=meth,window=w,hypothesis=metric,pairwise_rank_agreement=correct/pairs if pairs else np.nan,pairs=pairs,counterexamples=';'.join(f'{a}-{b}' for a,b in ce)))
# cross-method late association and within method qualification
for metric,ori in [('cancellation_efficiency',1),('mean_adjacent_cosine',1),('clean_to_source',1),('cf_source_ratio',-1)]:
 g=f[f.window=='late'];res.append(dict(method='ALL',window='late',hypothesis=metric,pairwise_rank_agreement=g[[metric,'target_acc']].corr().iloc[0,1],pairs=len(g),counterexamples='cross-method pooled correlation'))
r=pd.DataFrame(res);r.to_csv(out/'candidate_mechanism_audit_summary.csv',index=False)
with open(out/'candidate_mechanism_audit_report.md','w') as h:
 h.write('# CMNIST candidate mechanism audit\n\n')
 h.write('All features are source-side; target is merged only for external ranking audit.\n\n')
 for _,x in r.iterrows(): h.write(f"- {x.method} / {x.window} / {x.hypothesis}: rank agreement={x.pairwise_rank_agreement:.3f}, pairs={x.pairs}, counterexamples={x.counterexamples or 'none'}.\n")
 h.write('\n## Decision\n\nNo candidate passes a method-independent mechanism test. A candidate must hold within methods and across methods; these statistics show method-conditioned behavior and counterexamples. Observable transfer is only a proxy because valid dynamic common-space A_t/O_t pullback is unavailable in these artifacts.\n')
json.dump({'dataset':'ColoredMNIST','target_used_only_external_audit':True,'methods':list(DIRS),'candidate_metrics':['cancellation_efficiency','mean_adjacent_cosine','clean_to_source','cf_source_ratio'],'decision':'numeric_geometry_family'},open(out/'candidate_mechanism_audit_provenance.json','w'),indent=2)
print(r.to_string(index=False))
