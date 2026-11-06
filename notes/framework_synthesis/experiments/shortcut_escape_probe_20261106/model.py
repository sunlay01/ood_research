import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

def make(seed=21,n=512,k=16):
 rng=np.random.default_rng(seed);D=rng.choice([-1.,1.],n);Y=D*rng.choice([-1.,1.],n,p=[.25,.75]);Z=np.random.default_rng(seed+10000).normal(size=(n,64))[:,:k];C=np.array([Y*rng.choice([-1.,1.],n,p=[ep,1-ep]) for ep in [.2,.1]])
 return D,(Y+1)/2,Z,C
class Model:
 def __init__(self,data,balanced=False):
  self.D,self.y,self.Z,self.C=data;self.n=len(self.y);self.k=self.Z.shape[1];self.balanced=balanced
  def features(c):return np.column_stack([self.D,np.ones(self.n),c,c[:,None]*self.Z])
  if balanced:
   self.X=np.concatenate([features(np.ones(self.n)),features(-np.ones(self.n))]);self.yy=np.tile(self.y,2)
   self.W=np.array([np.r_[np.where(self.y==1,1-e,e),np.where(self.y==1,e,1-e)]/self.n for e in [.2,.1]])
  else:
   self.X=np.concatenate([features(c) for c in self.C]);self.yy=np.tile(self.y,2);self.W=np.zeros((2,2*self.n));self.W[0,:self.n]=1/self.n;self.W[1,self.n:]=1/self.n
 def rgh(self,q):
  f=self.X@q;p=expit(f);l=np.logaddexp(0,f)-self.yy*f;r=self.W@l;g=(self.W*(p-self.yy))@self.X;h=np.array([self.X.T@((w*p*(1-p))[:,None]*self.X) for w in self.W]);return r,g,h
 def obj(self,q,lam=0):
  r,g,h=self.rgh(q);d=r[0]-r[1];v=g[0]-g[1]
  return r.sum()+lam*d*d,g.sum(0)+2*lam*d*v,h.sum(0)+2*lam*(np.outer(v,v)+d*(h[0]-h[1]))
 def risk(self,q):r,g,h=self.rgh(q);return r.sum(),g.sum(0)
 def gap(self,q):r,g,h=self.rgh(q);return r[0]-r[1]
 def gapjac(self,q):r,g,h=self.rgh(q);return g[0]-g[1]
 def stats(self,q):
  v=q[2]+self.Z@q[3:];r,g,h=self.rgh(q);return {'R':float(r.mean()),'gap':float(r[0]-r[1]),'b':float(v.mean()),'S':float(np.mean(v*v)),'a':float(q[0]),'intercept':float(q[1]),'h_norm':float(np.linalg.norm(q[3:])),'q':q.tolist()}
 def fit_erm(self):return minimize(self.risk,np.zeros(self.k+3),jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':2000})
 def fit_equal(self,start):
  f=minimize(self.risk,start,jac=True,constraints=[{'type':'eq','fun':self.gap,'jac':self.gapjac}],method='SLSQP',options={'ftol':1e-12,'maxiter':1500})
  r,g,h=self.rgh(f.x);dg=g[0]-g[1];gr=g.sum(0);mu=-gr@dg/(dg@dg);_,_,VT=np.linalg.svd(dg[None,:],full_matrices=True);T=VT[1:].T;H=h.sum(0)+mu*(h[0]-h[1]);s=self.stats(f.x);s.update(success=bool(f.success),kkt=float(np.linalg.norm(gr+mu*dg)),tangent_hessian_min=float(np.linalg.eigvalsh(T.T@H@T).min()),mu=float(mu));return f,s

