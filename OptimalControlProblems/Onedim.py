from oc import *


#
# 1D Problem X'(t) = X(t) + u
# u in (umin, umax)
#############################################################################
    
class OnedimProblem(OptimalControlProblem):

  def __init__(self, FinalTime):
    
    super().__init__()
    
    assert FinalTime > 0.

    umin, umax = 0, 5.0
    self.tau = FinalTime - np.log(5/2)

    # dimensions of the input/output data spaces 
    self.N = 1
    self.M = 1
    
    self.d = 1
    self.K = 1
    self.sizeu = 1
    self.u_bound =  [(umin, umax)]  # min-max values 

    # init data    
    self.X0 = np.array([[2.0]])
    self.XT = np.array([[0.0]])    # not used !
    self.Xi = self.XT
    
    assert isinstance(self.X0, np.ndarray)
    assert self.X0.ndim == 2
    
    self.T = FinalTime
    
    self.f   = lambda t, X, u : X + u
    self.duf = lambda t, X, u : np.array([[[1.0]]])
    self.dxf = lambda t, X, u : np.array([[[1.0]]])
    self.dudxf = lambda t, X, u : np.zeros((self.K, self.sizeu, self.d, self.d))

    self.phi_expr = 'Phi=0.0'
    self.phi = lambda all_X, XT: 0.0
    self.sum_phi = lambda all_X, XT: 0.0
    self.dxphi = lambda all_X, XT : np.array([[0.0]])
    
    self.L = lambda X, u, u0 : 3*u - 2*X
    self.duL = lambda X, u, u0 : np.array([[3.0]])
    self.dxL = lambda X, u, u0 : np.array([[-2.0]])
    self.dudxL = lambda X, u : np.array([0.0])

    self.x_true = lambda t : (umax + self.X0 )*np.exp(t)-umax if t < self.tau \
      else ((umax+self.X0) * np.exp(self.tau) - umax + umin) * np.exp(t-self.tau) - umin

    self.u_true = lambda t : umax if t < self.tau else umin

    
  # 
  def plot_prediction(self, algo, ax = None, cax = None, fig = None):
    """
    input : algo, ax, number of test points 
    output : test inputs for plottings x_in and x_out 
    """  

    x_in  = algo.mesh_X.points
    x_out = np.zeros(algo.mesh_X.n)
    x_e   = np.zeros(algo.mesh_X.n)
    
    for it,t in enumerate(x_in):
      x_out[it] = algo.all_X[it, 0, 0]
      x_e[it]   = self.x_true(t).squeeze()

    if ax is not None:
      ax.cla()
      ax.set_title('Fonction X')
      ax.set_xlabel(r'Time $t$')
      ax.set_ylabel(r'$X_t$')
      ax.plot(x_in, x_out, 'x')
      ax.plot(x_in, x_e,   '-')
      
    return x_in, x_out

  
  # plot controls
  def plot_U(self, algo, ax, title = 'Controls'):   
    ax.cla()
    ax.plot(algo.mesh_U.points, [self.u_true(t) for t in algo.mesh_U.points], '-', label='exact')
    for j in range(self.sizeu):
      ax.plot(algo.mesh_U.points, algo.u[:, j, 0],  'x-')
    ax.set_title(title)
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'$\theta^*_t$')
