import torch
import numpy as np
import matplotlib.pyplot as plt

from ode import ODEsolve2
from Mesh import Mesh

class Function:
  """Function Class f"""
  
  def __init__(self, mesh, all_f):
    """Initializes a control"""

    nt, fx, fy = all_f.shape
    
    assert nt == mesh.n
    assert fy == 1
    assert fx >= 1
    
    self.all_f = all_f
    self.mesh = mesh 
    self.sizef = fx


  def __call__(self, t):
    """
    Evaluate the function at some time t
    """
    it = np.count_nonzero(self.mesh.points<t)-1

    if t <= self.mesh.t0:
      it = 0
    elif t > self.mesh.T:
      it = self.mesh.n-1

    return self.all_f[it, :, :]
  


class OptimalControlProblem:
  """Abstract class of Optimal control problems"""
  
  def __init__(self):
    """
    Initializer of OC abstract class of the form    
    { inf_{u} phi(X_u(T)) + int_0^T L(X_u(s), u(s))ds
    where 
    { X'   = f(t,X,u)
    { X(0) = x0
    """
    self.Tf         = None
    self.sizeu      = None
    self.phi        = None
    self.f          = None
    self.L          = None  # Lagrangian    
    self.d          = None  # size of the vector X
    self.K          = None
    self.u_bound    = None
    self.xhat       = None
    self.yhat       = None    

  def solve(self, mesh, eta = 1.0, maxiter=100, learning_rate=0.1, u0=None):
    """
    Solve the primal optimal control problem
    """
    
    plt.ion()
    fig, axs = plt.subplots(1,2)
  
    # Initialize
    print('Init control u')

    all_u = u0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([all_u], lr=learning_rate)
    #optimizer = torch.optim.SGD([all_u], lr=learning_rate)    

    for i in range(maxiter):
      
      # States
      X = self.xhat      
      all_X = torch.zeros(mesh.n, self.d, self.K)
      all_X[0, :, :] = self.xhat

      # Control
      u = Function(mesh, all_u)
      
      for j in range(1, mesh.n):
        t = mesh.h * j
        X = X + mesh.h * self.f(t, X, u(t))
        all_X[j, :, :] = X
      
      loss = self.phi(X) + eta*mesh.h*torch.sum(self.L(all_X, all_u))
      #optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      with torch.no_grad(): # all commands will skip grad computations
        all_u.clamp_(self.u_bound[0][0], self.u_bound[0][1])

      self.draw_plot(mesh, all_X, all_u, axs)      
      plt.draw()
      plt.pause(0.0001)
      
      print(f'i={i+1}/{maxiter}\tJ={loss.squeeze()}')
    plt.ioff()
    
    return all_u, all_X
