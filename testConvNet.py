import OptimalControlProblems.ConvNet as cnet
import torch
import numpy as np


d = 8
K = 7000
T = 1

A = cnet.ConvNet(d, K, T)


u1 = torch.randn(A.sizeu)
W, b = A.reshaper(u1)
u2 = A.shaper(W,b)

print('test (re)shaper')
print('u1, W, b, u2 = shaper(W,b)')
print(u1)
print(W)
print(b)
print(u2)


print('test f')
t = 0.1
X0 = torch.randn(K, A.d)
X1 = A.f(t, X0, u1)


print('test ODE')
import ode
import torch
import Mesh

mesh = Mesh.Mesh(0,1)
f = lambda t, X : A.f(t,X,u1)
XT = ode.ODEsolve2(mesh, f, X0)
