import torch as tc

from ocp import OptimalControlProblem

class Oscillator(OptimalControlProblem):
  """
  Two dim oscillator :
  u(t) is scalar
  X_u(t) is a 2dim vector sol of
  X'(t) = A @ X + B u
  B is a constant vector
  """
  def __init__(self, Tf):
    
    super ().__init__ ()

    self.d = 2
    self.K = 1
    self.Tf = Tf
    
    # Control
    self.sizeu = 1
    self.u_bound =  [(-10.0, 10.0)]  # min max values

    # Constraint
    A = tc.tensor([[0.0,1.0],[-1.0,0.0]])
    B = tc.tensor([[-1.0],[1.0]])
    self.f = lambda t, X, u : A @ X + B * u  # ERROR ??? It seems I can't A @ X 

    # Cost
    self.xhat = tc.tensor([[3.0],[-1.0]])
    self.yhat = tc.tensor([[2.0],[-2.0]])
    self.phi = lambda XT : 0.5 * tc.norm(XT - self.yhat)**2
    self.dxphi = lambda X : tc.ones(self.d, self.K)

    # Regularization
    self.L = lambda all_X, u: 0.5 * u**2


    # Live Plot
  def draw_plot(self, mesh, all_X, all_u, axs):
    """
    Draw the state and the control
    """
    
    axs[0].cla()
    axs[1].cla()
    
    axs[0].plot(mesh.points, all_u.detach().numpy()[:, 0, 0].squeeze(), '-', label=r'u(t)')
    axs[1].plot(all_X.detach().numpy()[:, 0, 0].squeeze(), \
                all_X.detach().numpy()[:, 1, 0].squeeze(), \
                '--', label=r'$X_u(t)$')
    xh = self.xhat.numpy().squeeze()
    yh = self.yhat.numpy().squeeze()
    axs[1].plot(xh[0],xh[1], 'x', label='Start')
    axs[1].plot(yh[0],yh[1], 'o', label='End')

    axs[0].legend()
    axs[1].legend()
