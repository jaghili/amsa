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

  def solve(self, mesh, maxiter=10, rho=1.0, learning_rate=0.01, u0 = None):
    """
    Solve : maximization of the Hamiltonian
    """
    plt.ion()
    fig, axs = plt.subplots(1, 3)
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()
    
    all_u = u0.clone().detach()

    # Hamiltonian function 
    H = lambda t, X, P, u : tc.sum(P * self.oc.f(t, X, u)) - self.oc.L( X, u)

    list_X  = [None] * mesh.n  
    list_P  = [None] * mesh.n  
    list_Xp = [None] * mesh.n 
    list_Pp = [None] * mesh.n 
    list_H  = [None] * mesh.n

    all_J = list()
    
    # EMSA algorithm    
    for i in range(maxiter):
      print(f'\n================ i={i+1} ================')
      #axs[0].cla()      
      
      print('[dual/solve] Init')
      print(all_u.squeeze())

      # Forward      
      print('[dual/solve] Forward')
      print(f'\t j = 0 extremité gauche' )
      list_X[0] = self.oc.xhat.clone().detach().requires_grad_(True)
      list_Xp[0] = self.oc.f(0.0, list_X[0], all_u[0, :, 0])
      
      list_X[0].retain_grad()
      list_Xp[0].requires_grad_(True)
      
      for j in range(1, mesh.n):
        print(f'\t j = {j}')
        tl = mesh.h * (j-1)
        t = mesh.h * j
        
        list_X[j] = list_X[j-1] + mesh.h * self.oc.f(tl, list_X[j-1], all_u[j-1,:,0])
        list_Xp[j] = self.oc.f(t, list_X[j], all_u[j, :, 0])
        
        list_X[j].retain_grad()

        
      # Backward
      print('[dual/solve] Backward')
      print(f'\t j = {mesh.n-1} extremité droite')
      list_P[-1] = - self.oc.dxphi(list_X[-1]).clone().detach()
      list_H[-1] = H(mesh.points[-1], list_X[-1], list_P[-1], all_u[-1, :, 0])
      list_H[-1].backward(retain_graph=True)      
      list_Pp[-1] = - list_X[-1].grad
      
      
      for j in reversed(range(0, mesh.n-1)):
        print(f'\t j = {j}')
        tr = mesh.h * (j+1)
        list_H[j+1] = H(tr, list_X[j+1], list_P[j+1], all_u[j+1, :, 0])
        list_H[j+1].backward(retain_graph=True) 
        
        list_P[j]  = list_P[j+1] + mesh.h * list_X[j+1].grad
        list_Pp[j] = - list_X[j].grad   # dH/dX_j
        #print(f'\tj={j}\t t={t}')

      # Maximization
      print('[dual/solve] Maximize H')
      for it,t in enumerate(mesh.points):
        print(f'... on t={t}')        
        u = tc.randn(self.oc.sizeu, requires_grad=True)
        optimizer = tc.optim.Adam([u], lr=learning_rate)

        for j in range(10):
          _H = H(t, list_X[it], list_P[it], u)
          _H.backward(retain_graph=True)
          
          minus_Hamiltonian = - _H                                               \
            + 0.5 * rho * tc.norm(list_Xp[it] - self.oc.f(t, list_X[it], u))**2  \
            + 0.5 * rho * tc.norm(list_Pp[it] + list_X[it].grad)**2
          minus_Hamiltonian.backward(retain_graph=True)
          optimizer.step()
          # print(f'it={it}\t j={j} \taH={-minus_Hamiltonian.squeeze()}')

        all_u[it, :, 0] = u
        
        with tc.no_grad(): # skip grad computations
          all_u.clamp_(self.oc.u_bound[0][0], self.oc.u_bound[0][1])

      all_X = tc.stack(list_X)
      
      loss = self.oc.phi(all_X[-1, :, :]) + mesh.h*tc.sum(self.oc.L(all_X, all_u))
      all_J.append(loss.squeeze())
      
      print(f'J = {loss}')
      
      self.oc.draw_plot(mesh, all_X, all_u, all_J, axs)
      # print(all_u.squeeze())
      plt.draw()
      plt.pause(0.0001)

    return all_u, all_X
