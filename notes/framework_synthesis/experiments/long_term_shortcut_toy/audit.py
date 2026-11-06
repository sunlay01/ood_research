"""Check the fixed-data assumptions and quantitative stationary-point exclusion.
The theorem is analytic; floating-point tests are not interval certificates.
Run beside model.py and reference_full_flow.json. Requires numpy and scipy.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from model import make, Model

root=Path(__file__).resolve().parent
reference=json.loads((root/'reference_full_flow.json').read_text())
rows=[]
for seed in [21,22,23]:
    D,y,Z,C=make(seed,512,1); z=Z[:,0]; n=len(y); m=Model((D,y,Z,C))
    F=(C!=(2*y-1)).astype(int)
    ks=[int((F[0]-F[1])[(D==s)&(y==t)].sum())
        for s in [-1,1] for t in [0,1]]
    overlaps=[]
    for s in [-1,1]:
        for c in [-1,1]:
            groups=[np.concatenate([z[(D==s)&(C[e]==c)&(y==t)]
                                    for e in [0,1]]) for t in [0,1]]
            assert all(len(g)>0 for g in groups)
            overlaps.append(float(min(g.max() for g in groups)-max(g.min() for g in groups)))
    ps=np.array([y[D==s].mean() for s in [-1,1]])
    ug=np.log(ps/(1-ps))
    qg=np.array([(ug[1]-ug[0])/2,(ug[1]+ug[0])/2,0,0])
    _,gr,_=m.rgh(qg)
    A=gr.sum(0)[2:]; B=(gr[0]-gr[1])[2:]
    tangent=A-B*(A@B)/(B@B); rho=float(np.linalg.norm(tangent))
    G=np.column_stack([np.ones(n),z]); G=G.T@G/n
    gmin=float(np.linalg.eigvalsh(G).min())
    kap=2*(min(ks[:2])+min(ks[2:]))/n
    assert min(overlaps)>0 and min(ks)>0 and rho>0 and gmin>0
    pmin=float(np.minimum(ps,1-ps).min()); M=4/kap
    L=.5*(3+np.mean(z*z))
    Hu=((1+M)/4)/((pmin/2)*(1-pmin/2))
    Hq=np.sqrt(Hu*Hu+1/gmin)
    thresholds=[kap,2*pmin/(1+M),rho/(L*(1+M)*Hq)]
    tstar=min(thresholds)
    explicit=gmin/(1+z*z).max()*tstar*tstar
    assert explicit>0

    # Check exact Delta_b(u,0), uniform bound and shape block estimates
    # at broad finite shape parameters, independent of fitted endpoints.
    rng=np.random.default_rng(seed+20000)
    max_identity=0.; max_bound_violation=0.
    for j in range(100):
        q=rng.normal(size=4)*np.array([10,10,.1,.1])
        qzero=q.copy();qzero[2:]=0
        _,g,_=m.rgh(q);_,gz,_=m.rgh(qzero)
        t=np.max(np.abs(q[2]+q[3]*z))
        us=np.array([q[1]-q[0],q[1]+q[0]])
        formula=2/n*sum(ks[2*k]*expit(us[k])+ks[2*k+1]*(1-expit(us[k]))
                         for k in [0,1])
        max_identity=max(max_identity,abs((gz[0]-gz[1])[2]-formula))
        assert formula>=kap-1e-12
        max_bound_violation=max(max_bound_violation,
            abs((g[0]-g[1])[2]-formula)-t/2)
        for k,s in enumerate([-1,1]):
            # partial_u_s = (partial_d + s partial_a)/2
            R_us=((g[0]+g[1])[1]+s*(g[0]+g[1])[0])/2
            D_us=((g[0]-g[1])[1]+s*(g[0]-g[1])[0])/2
            frac=np.mean(D==s)
            max_bound_violation=max(max_bound_violation,
                abs(R_us-2*frac*(expit(us[k])-ps[k]))-frac*t/2,
                abs(D_us)-frac*t/2)
    assert max_identity<1e-12 and max_bound_violation<1e-12

    endpoint_checks=[]
    for record in reference:
        if record['seed']!=seed: continue
        q=np.array(record['q'])
        _,grad,H=m.obj(q,10000)
        S=m.stats(q)['S']
        assert np.linalg.norm(grad)<1e-8
        assert S>=explicit
        assert abs(S-record['S'])<1e-12
        endpoint_checks.append({'init':record['init'],'S':S,
            'gradient_norm_recomputed':float(np.linalg.norm(grad)),
            'hessian_min_recomputed':float(np.linalg.eigvalsh(H).min())})
    rows.append({'seed':seed,'k_st':ks,'minimum_overlap_width':min(overlaps),
        'p_s':ps.tolist(),'kappa':kap,'noncollinearity_rho':rho,
        'G_min':gmin,'explicit_stationary_S_lower_bound':float(explicit),
        'thresholds_t':list(map(float,thresholds)),
        'identity_error':float(max_identity),
        'max_positive_inequality_violation':float(max(0,max_bound_violation)),
        'reference_endpoints_recomputed':endpoint_checks})
result={'status':'PASS','scope':'data conditions, proof inequalities, and recomputed derivatives at prior endpoints; no new training','rows':rows}
(root/'audit_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

