import numpy as np
from gmmflow import *
def P(c):  # two modes on x-axis, weight of left mode = c
    return np.array([c,1-c]), np.array([[-2.,0],[2.,0]]), np.array([.3,.3])
rng=np.random.default_rng(0)
x0=rng.standard_normal((20000,2))
for c in [0.5,0.3,0.1]:
    x1=flow(x0,P,c,n_steps=200)
    print(c,'left-mass',(x1[:,0]<0).mean(),'std mode',x1[x1[:,0]<0].std(0))
# Jacobian check
xs=rng.standard_normal((50,2))
_,F=flow(xs,P,0.3,jac=True,n_steps=200)
_,Ffd=ftle_fd(xs,P,0.3,n_steps=200)
print('max rel jac err',np.abs(F-Ffd).max()/np.abs(F).max())
# 3-mode check
def P3(c):
    ang=np.deg2rad([90,210,330]); mu=2.2*np.stack([np.cos(ang),np.sin(ang)],1)
    return np.array([.5,.3,.2]),mu,np.array([.3,.3,.3])
x1=flow(x0,P3,0.,n_steps=200)
lab=np.argmin(((x1[:,None]-P3(0)[1][None])**2).sum(-1),1)
print('3-mode masses',np.bincount(lab)/len(lab))
