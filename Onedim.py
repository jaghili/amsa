import math

import numpy as np
import torch as tc

from ocp import OptimalControlProblem

class BangBang(OptimalControlProblem):
  """
  One dimentional optimal control problem:
  x_u(t) is scalar
  L(u)   is scalar
  u(t)   is scalar

  tau = Tf - math.log(5/2)

  Taken from http://www.sfu.ca/~wainwrig/Econ331/Chapter20-bangbang-example.pdf

  x' = x + u
  x(0) = 4
  u in (0,2))
  """
  def __init__(self, Tf=2.0, eta=1.0):
    """Initializes a one-dimentional OC problem"""

    # Init OptimalControlProblem attributes
    super().__init__() 

    self.d          = 1
    self.K          = 1
    self.Tf         = Tf
    self.tau        = Tf - math.log(5/2)
    print(f'BangBang : tau = {self.tau}')
    
    # Control
    self.sizeu      = 1
    umin            = 0.0
    umax            = 2.0
    self.u_bound = [(umin, umax)]

    # Constraint
    self.f          = lambda t, X, u : X + u
    
    # Cost
    self.xhat       = 4*tc.ones(self.d, self.K)
    self.yhat       = -tc.ones(self.d, self.K)
    self.phi        = lambda all_X : tc.tensor(0.0)
    self.dxphi      = lambda all_X : tc.tensor(0.0)

    # Exact solns
    self.x_true = lambda t : (umax + self.xhat.numpy().squeeze() )*np.exp(t) - umax if t < self.tau \
      else ((umax + self.xhat.numpy().squeeze()) * np.exp(self.tau) - umax + umin) * np.exp(t-self.tau) - umin
    self.u_true = lambda t : umax if t < self.tau else umin
    
    # Regularization part
    self.L          = lambda X, u : eta * (3*u - 2*X)

    # Live Plot
  def draw_plot(self, mesh, all_X, all_u, all_J, axs):
    """
    Draw the state and the control
    """
    
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()
    
    axs[0].plot(mesh.points, all_u.detach().numpy()[:,0,0].squeeze(), '-', label=r'$u(t)$')
    axs[0].plot(mesh.points, [self.u_true(t) for t in mesh.points], 'x', label=r'$u*(t)$')

    axs[1].plot(mesh.points, all_X.detach().numpy()[:,0,0].squeeze(), '--', label=r'$X_u(t)$')
    axs[1].plot(mesh.points, [self.x_true(t) for t in mesh.points], 'o', label=r'$X_{u*}(t)$')

    axs[2].plot(all_J)
    
    axs[0].legend()
    axs[1].legend()
