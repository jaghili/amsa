import OptimalControlProblems.ConvNet as cnet
import torch
import numpy as np

print('test scikit learn dataset')
from sklearn import datasets
import Mesh
import ode

# Import Digits
digits = datasets.load_digits()
K, dimx, dimy = digits.images.shape
d = dimx
T = 1

A = cnet.ConvNet(d, K, T)
u1 = torch.randn(A.sizeu)
mesh = Mesh.Mesh(0, 1.0)


# Hamiltonian
def H(t, bX, bP, u):
  """
  input :
  - t time
  - bX : batch of X, size [K, d]
  - bP : batch of P, size [K, d]
  - u  : control u of size [sizeu]
  output:
  - scalar H = P * f(t,X,u) - L(u)
  """  
  return torch.sum(bP * A.f(t, bX, u), axis=1)


# inputs/outputs
Xi = torch.from_numpy(np.array(digits.images.reshape(K, dimx*dimy), dtype=np.float32))
Xo = torch.from_numpy(np.array(digits.target, dtype=np.float32))
                      
# forward
f = lambda t, X : A.f(t, X, u1)
X0 = Xi
all_X, all_Xp = ode.ODEsolve2(mesh, f, X0)

# backward
XT = all_X[-1,:,:]
PT = - A.dxphi(XT, Xo)


# maximize Hamiltonian
