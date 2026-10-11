from __future__ import annotations

import argparse, copy, json, sys
from pathlib import Path
import torch
from torch import autograd
from torch.nn import functional as F

REPO_SRC = Path('/Users/sunlay/Documents/Codex/2026-09-13/zu/repo/src')
sys.path.insert(0, str(REPO_SRC)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from ood_repr_reg.task3_baseline_fidelity.official_irm import OfficialIRMMLP, weight_norm_squared
from run_cmnist_source_only_response import read_mnist_source, make_source, flatten, unflatten

def pvec(m): return torch.cat([p.detach().reshape(-1) for p in m.parameters()])
def virtual(m,opt,g):
    c=copy.deepcopy(m); ps=tuple(c.parameters()); o=torch.optim.Adam(ps,lr=opt.param_groups[0]['lr']); o.load_state_dict(copy.deepcopy(opt.state_dict())); before=pvec(c); o.zero_grad(set_to_none=True)
    for p,v in zip(ps,unflatten(g,ps)): p.grad=v.detach().clone()
    o.step(); c.eval(); return c,pvec(c)-before
def risks(m,envs):
    with torch.no_grad(): return torch.stack([F.binary_cross_entropy_with_logits(m(e.images).reshape(-1),e.labels.reshape(-1)) for e in envs])
def responses(m,envs,graph=False):
    out=[]
    for e in envs:
        z=m(e.images).reshape(-1); y=e.labels.reshape(-1); s=torch.ones((),requires_grad=True)
        out.append(autograd.grad(F.binary_cross_entropy_with_logits(z*s,y),s,create_graph=graph)[0])
    return torch.stack(out)
def objective(m,envs,a,ps):
    rs=risks(m,envs); risk=rs.mean(); rr=responses(m,envs,True); pen=rr.square().mean(); obj=(risk+a.l2*weight_norm_squared(m)+a.penalty_weight*pen)/a.penalty_weight
    return risk.detach(),rr.detach(),flatten(autograd.grad(obj,ps,allow_unused=True),ps).detach()
def diff_gradient(m,envs,ps):
    rr=responses(m,envs,True); d=(rr[0]-rr[1]).square(); return flatten(autograd.grad(torch.log1p(d),ps,allow_unused=True),ps).detach(),d.detach()
def run(a):
    torch.set_num_threads(a.threads); torch.use_deterministic_algorithms(True)
    im,dig,_,_=read_mnist_source(a.data_root,50000); envs,_=make_source(im,dig,a.seed,0)
    path=a.checkpoint_dir/'trajectory'/f'seed{a.seed}__arm-irm_vanilla__step{a.start_step:05d}.pt'; payload=torch.load(path,map_location='cpu',weights_only=False)
    m=OfficialIRMMLP(int(payload['hidden_dim'])); m.load_state_dict(payload['model_state']); m.eval(); opt=torch.optim.Adam(m.parameters(),lr=a.lr); opt.load_state_dict(payload['optimizer_state']); ps=tuple(m.parameters()); rows=[]
    for step in range(a.start_step,a.end_step):
        risk,rr,g0=objective(m,envs,a,ps); gd,diff=diff_gradient(m,envs,ps); base,bdelta=virtual(m,opt,g0); br=risks(base,envs); be=responses(base,envs); bexp=(be-be.mean()).square().mean(); gn=g0.norm().clamp_min(1e-30); dn=gd.norm().clamp_min(1e-30); cand=[]; states=[]
        for alpha in a.alphas:
            g=g0+float(alpha)*gn*gd/dn; cm,delta=virtual(m,opt,g); cr=risks(cm,envs); ce=responses(cm,envs); cexp=(ce-ce.mean()).square().mean(); corr=float((delta-bdelta).norm()/bdelta.norm().clamp_min(1e-30)); risk_ok=bool(torch.all(cr<=br+a.risk_tol)); exposure_ok=bool(cexp<=bexp+a.exposure_tol); budget_ok=corr<=a.max_correction; cand.append({'alpha':float(alpha),'diff_after':float(((ce[0]-ce[1])**2).detach()),'risk_ok':risk_ok,'exposure_ok':exposure_ok,'budget_ok':budget_ok,'feasible':risk_ok and exposure_ok and budget_ok,'correction_ratio':corr,'responses_after':ce.detach().tolist()}); states.append((g,delta))
        ok=[i for i,x in enumerate(cand) if x['feasible']]; fallback=not ok; i=min(ok,key=lambda j:cand[j]['diff_after']) if ok else min(range(len(cand)),key=lambda j:abs(cand[j]['alpha'])); g,delta=states[i]; before=pvec(m); opt.zero_grad(set_to_none=True)
        for p,v in zip(ps,unflatten(g,ps)): p.grad=v.detach().clone()
        opt.step(); actual=pvec(m)-before; rows.append({'from_step':step,'to_step':step+1,'diff_before':float(diff),'diff_after':float(((responses(m,envs)[0]-responses(m,envs)[1])**2).detach()),'selected':cand[i],'candidates':cand,'fallback':fallback,'update_error':float((actual-delta).norm()/delta.norm().clamp_min(1e-30))})
        if (step+1-a.start_step)%a.progress==0 or step+1==a.end_step: print(f'step={step+1} alpha={cand[i]["alpha"]:g} diff={float(diff):.3g}->{cand[i]["diff_after"]:.3g} fallback={fallback}',flush=True)
    a.out.mkdir(parents=True,exist_ok=True); cp=a.out/f'diff_control_step{a.end_step:05d}.pt'; torch.save({'model_state':m.state_dict(),'optimizer_state':opt.state_dict(),'step':a.end_step,'hidden_dim':int(payload['hidden_dim'])},cp); (a.out/'source_only_run.json').write_text(json.dumps({'experiment':'cmnist_source_response_difference_control','protocol':{'source_only':True,'target_data_loaded':False,'start_step':a.start_step,'end_step':a.end_step,'controlled':'(r_env0-r_env1)^2'},'checkpoint':str(cp),'steps':rows},indent=2)+'\n')
if __name__=='__main__':
    z=argparse.ArgumentParser(); z.add_argument('--data-root',type=Path,default=Path('/Users/sunlay/Desktop/data')); z.add_argument('--checkpoint-dir',type=Path,required=True); z.add_argument('--out',type=Path,required=True); z.add_argument('--seed',type=int,required=True); z.add_argument('--start-step',type=int,default=100); z.add_argument('--end-step',type=int,default=500); z.add_argument('--alphas',type=float,nargs='+',default=[-1,-.5,-.25,-.1,-.05,0,.05,.1,.25,.5,1]); z.add_argument('--lr',type=float,default=1e-3); z.add_argument('--l2',type=float,default=1e-3); z.add_argument('--penalty-weight',type=float,default=10000.); z.add_argument('--risk-tol',type=float,default=1e-4); z.add_argument('--exposure-tol',type=float,default=0.); z.add_argument('--max-correction',type=float,default=.25); z.add_argument('--threads',type=int,default=4); z.add_argument('--progress',type=int,default=100); run(z.parse_args())
