"""Deterministic checks for the environment-response shortcut skeleton.
Run: python environment_response_shortcut_checks.py
Requires numpy and scipy. No training or downloaded data.
"""
import json
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm

ell = lambda x: np.logaddexp(0.0, -x)
eta, a, h, eps, tau = .25, np.log(3.), .4, .15, .1

def integral(fun, points=()):
    edges = [-np.inf] + sorted(set(float(p) for p in points)) + [np.inf]
    return sum(quad(lambda z: fun(z)*norm.pdf(z), lo, hi,
                    epsabs=2e-11, epsrel=2e-10)[0]
               for lo, hi in zip(edges[:-1], edges[1:]))

def conditional_gamma(z, q):
    aa, dd, bb, hh = q
    v = bb + hh*z
    return sum(.5*w*(ell(D*N*(aa*D+dd)-v)-ell(D*N*(aa*D+dd)+v))
               for D in [-1,1] for N,w in [(1,1-eta),(-1,eta)])

q0 = np.array([a, 0., 0., h])
beta0 = integral(lambda z: conditional_gamma(z,q0), [0])
beta1 = integral(lambda z: np.sign(z)*conditional_gamma(z,q0), [0])

def risk(t):
    def at_z(z):
        p=eps+t*np.sign(z); v=h*z
        return sum(w*((1-p)*ell(A+v)+p*ell(A-v))
                   for A,w in [(a,1-eta),(-a,eta)])
    return integral(at_z,[0])

increment=risk(tau)-risk(0)
lower=2*expit(-a)*abs(h)*np.sqrt(2/np.pi)
assert abs(beta0)<1e-10
assert abs(increment-tau*beta1)<1e-10
assert beta1>=lower-1e-10

def clean_direct(t, hh):
    def at_z(z):
        p=eps+t*np.sign(z)
        return sum(.5*wN*wF*(D*(a*D+D*N*(1-2*F)*hh*z)<0)
                   for D in [-1,1] for N,wN in [(1,1-eta),(-1,eta)]
                   for F,wF in [(0,1-p),(1,p)])
    return integral(at_z,[-a/abs(hh),0,a/abs(hh)])

Aflip=2*norm.cdf(-a/abs(h))
classification_errors=[]
for hh in [h,-h]:
    for t in [-tau,0,tau]:
        expected=Aflip*(.5+(1-2*eta)*t*np.sign(hh))
        classification_errors.append(abs(clean_direct(t,hh)-expected))
assert max(classification_errors)<1e-10

rng=np.random.default_rng(21); n=512
D=rng.choice([-1,1],n)
Y=D*np.where(rng.random(n)<eta,-1,1)
z=rng.normal(size=n)
F=np.stack([rng.random(n)<p for p in [.2,.1]]).astype(float)
df=F[0]-F[1]
q=np.array([1.16,-.02,.04,-.16])

def sample_gamma(q):
    aa,dd,bb,hh=q; u=aa*D+dd;v=bb+hh*z
    return ell(Y*u-v)-ell(Y*u+v)

def delta(q): return np.mean(df*sample_gamma(q))
u=q[0]*D+q[1];v=q[2]+q[3]*z
direct=np.mean(ell(Y[None,:]*(u[None,:]+Y[None,:]*(1-2*F)*v[None,:])),axis=1)
paired=np.mean(ell(Y*u+v))+np.mean(F*sample_gamma(q),axis=1)
pair_error=float(np.max(np.abs(direct-paired)))
assert pair_error<1e-12

A=Y*u
gamma_v=expit(v-A)+expit(-A-v)
db=np.mean(df*gamma_v); dh=np.mean(df*gamma_v*z)
assert abs(db)>1e-6
direction=np.array([0.,0.,-dh/db,1.])
step=1e-5
tangent_fd=(delta(q+step*direction)-delta(q-step*direction))/(2*step)
assert abs(tangent_fd)<1e-8

lip_max=0.
for bb,dd in [(.03,-.02),(-.1,.15),(.2,-.3)]:
    q=np.array([a,dd,bb,h])
    b1=integral(lambda z:np.sign(z)*conditional_gamma(z,q),[0])
    excess=abs(b1-beta1)-(2*abs(bb)+abs(dd))
    lip_max=max(lip_max,excess)
    assert excess<1e-10

print(json.dumps({
    "status":"PASS",
    "empirical_pairing_max_error":pair_error,
    "compensation_tangent_fd":float(tangent_fd),
    "beta0":beta0,"beta1":beta1,"beta1_lower_bound":lower,
    "target_bce_increment":increment,
    "color_sensitive_mass":Aflip,
    "clean_source_error":.5*Aflip,
    "clean_target_error":(.5+tau*(1-2*eta))*Aflip,
    "classification_identity_max_error":max(classification_errors),
    "lipschitz_bound_max_positive_violation":lip_max
},indent=2))
