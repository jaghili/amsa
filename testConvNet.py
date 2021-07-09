import OptimalControlProblems.ConvNet as cnet
import torch
import numpy as np

from sklearn import datasets

import Mesh
import ode

# torch.autograd.set_detect_anomaly(True)

# Import Digits and settings
digits = datasets.load_digits()
K, dimx, dimy = digits.images.shape
d = dimx*dimy
Xi = torch.from_numpy(np.array(digits.images.reshape(K, d), dtype=np.float32))
Xo = torch.from_numpy(np.array(digits.target, dtype=np.float32))

# ConvNet
T = 1
A = cnet.ConvNet(dimx, K, T)

# Mesh
mesh = Mesh.Mesh(0, 1.0)

# History
all_X  = torch.zeros(mesh.n, K, d)
all_Xp = torch.zeros(mesh.n, K, d)
all_P  = torch.zeros(mesh.n, K, d)
all_Pp = torch.zeros(mesh.n, K, d)
all_u  = torch.zeros(mesh.n, A.sizeu, 1)

# Optimizer settings
learning_rate = 0.01

def u(t):
  it = 0
  if t >= 1.0:
    it = mesh.n -1
  elif t <= 0.0:
    it = 0
  else:
    it = np.count_nonzero(mesh.points < t) -1
    
  return all_u[it, :, 0] 

# Augmented Hamiltonian
def H(t, bX, bP, u):
  return torch.sum(bP * A.f(t, bX, u))

def dxH(t, bX, bP, u):
  _H = H(t, bX, bP, u)
  _H.backward(retain_graph=True)
  return bX.grad

def aH(t, bX, bXp, bP, bPp, u):
  _H = H(t, bX, bP)
  return 

def computeFlows():
  
  # forward
  X0 = Xi.clone().detach()
  f = lambda t, X : A.f(t, X, u(t))
  all_X, all_Xp = ode.ODEsolve2(mesh, f, X0)

  # backward
  XT = all_X[-1, :, :].clone().detach().requires_grad_(True)
  minus_dxH = lambda t, P : - dxH(t, XT, P, u(t)) # returns XT.grad
  PT = (-A.dxphi(XT, Xo)).clone().detach()
  all_P, all_Pp = ode.ODEsolve2(mesh, minus_dxH, PT)
  
  return all_X, all_Xp, all_P, all_Pp

def maximizeH(t, bX, bP):
  
  uopt = torch.randn(A.sizeu, requires_grad=True)
  optimizer = torch.optim.Adam([uopt], lr=learning_rate)
  
  for i in range(60):
    loss = - H(t, bX, bP, uopt)
    loss.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()
    if i % 20 == 0:
      print(f'\ti={i}\tH={-loss}')

  return uopt, -loss

print('Starting EMSA')
emsa_iter = 10

for i in range(emsa_iter):

  print('---- i = {} ----'.format(i))
  
  print('Compute all flows')
  all_X, all_Xp, all_P, all_Pp = computeFlows()

  print('Maximizing H : find optimal ut')  
  for it,t in enumerate(mesh.points):
    print(f't={t}')
    Xt = all_X[it, :, :].clone().detach()
    Pt = all_P[it, :, :].clone().detach()
    u_opt, Hu_opt = maximizeH(t, Xt, Pt)
    all_u[it, :, 0] = u_opt

  XT = all_X[-1, :, :]
  cost = torch.sum(A.phi(XT, Xo)) / K
  print(f'cost phi(XT) ={cost}')




########################
























  
