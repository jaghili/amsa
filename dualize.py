import torch as tc
import matplotlib.pyplot as plt

class Dualize:
  """
  Initiate the dual problem associated to a primal optimal control problem
  """

  def __init__(self, ocp):
    """
    Initialize the dual problem  associated to the primal problem OCP
    """
    self.oc = ocp

  def solve(self, mesh, eta = 1.0, maxiter=10, rho=1.0, learning_rate=0.75, u0 = None):
    """
    Solve : maximization of the Hamiltonian
    """
    plt.ion()
    fig, axs = plt.subplots(1,2)
    
    all_u = u0.clone().detach()

    
    # Hamiltonian function 
    _H = lambda t, X, P, u : tc.sum(P * self.oc.f(t, X, u)) - eta * self.oc.L(X,u) 

    # EMSA algorithm
    
    for i in range(maxiter):
      print(f'\n================ i={i+1} ================')
      axs[0].cla()
      
      print('[dual/solve] Init')            
      all_X = [None for i in range(mesh.n)]
      all_P = [None for i in range(mesh.n)]

      # Forward      
      print('[dual/solve] Forward')
      all_X[0] = self.oc.xhat.clone().detach().requires_grad_(True)
      all_X[0].requires_grad_(True)      
      for j in range(1, mesh.n):
        t = mesh.h * j
        all_X[j] = all_X[j-1] + mesh.h * self.oc.f(t, all_X[j-1], all_u[j-1,:,0])
        all_X[j].retain_grad()
        #print(f'\tj={j}\t t={t}')

        
      # Backward
      print('[dual/solve] Backward')
      all_P[-1] = - self.oc.dxphi(all_X[-1]).clone().detach()
      for j in reversed(range(0, mesh.n-1)):
        t = mesh.h * j
        H = _H(t, all_X[j+1], all_P[j+1], all_u[j+1, :, 0])
        H.backward(retain_graph=True)
        all_P[j] = all_P[j+1] + mesh.h * all_X[j+1].grad
        #print(f'\tj={j}\t t={t}')

      # Maximize
      print('[dual/solve] Maximize H')
      for it,t in enumerate(mesh.points):
        print(f'... on t={t}')        
        u = tc.randn(self.oc.sizeu, requires_grad=True)
        optimizer = tc.optim.Adam([u], lr=learning_rate)

        for j in range(30):
          loss = - _H(t, all_X[it], all_P[it], u)
          loss.backward(retain_graph=True)
          optimizer.step()
          #print(f'it={it}\t j={j} \tH={-loss.squeeze()}')

        all_u[it, :, 0] = u
        
        with tc.no_grad(): # all commands will skip grad computations
          all_u.clamp_(self.oc.u_bound[0][0], self.oc.u_bound[0][1])
        
        
      #self.draw_plot(mesh, all_X, all_u, axs)
      axs[0].plot(mesh.points, all_u.detach().numpy().squeeze(), '-', label='control')
      plt.draw()
      plt.pause(0.0001)

    return all_u
