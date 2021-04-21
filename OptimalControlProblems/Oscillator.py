from oc import * 

#
# 2D Oscillator :   X'(t) = f(t,X,u) = A.X + Bu
#
############################################################################################

class OscillatorProblem(OptimalControlProblem):

  def __init__(self, tf):
    
    super ().__init__ ()
    
    assert tf > 0.  # final time > 0

    # dimensions of the input/output data spaces 
    self.N = 2
    self.M = 2
    
    self.sizeu = 1
    self.u_bound =  [(None, None)]  # min max values

    # init data
    X0 = np.array([[3.0, -1.0]]).T
    XT = np.array([[2.0, -2.0]]).T
    
    self.X0 = X0  # HAS TO BE 2 DIM.
    self.XT = XT
    self.Xi = XT
    
    assert isinstance(self.X0, np.ndarray) and isinstance(self.XT, np.ndarray)
    assert self.X0.ndim == 2 and self.XT.ndim == 2

    self.d, self.K =  self.X0.shape
    
    A = np.array([[0, 1], [-1,0]])
    B = np.array([[0,1]]).T
    
    self.T = tf
    
    self.f   = lambda t, X, u : A @ X + B * u
    self.duf = lambda t, X, u : B.reshape(self.K, self.d, self.sizeu)
    self.dxf = lambda t, X, u : A
    self.dudxf = lambda t, X, u : np.zeros((self.d, self.d, self.sizeu))
    
    #
    self.phi_expr = str('0.5*||X(T) - XT||**2')
    self.phi = lambda all_X, XT : 0.5 * ((all_X - XT).T @ (all_X - XT)).squeeze()  # inner faster?
    self.sum_phi = lambda all_X, XT : 0.5 * ((all_X - XT).T @ (all_X - XT)).squeeze()  # inner faster?
    self.dxphi = lambda X, XT : all_X - XT
    
    self.L = lambda X, u, u0 : 0.5 * np.inner(u - u0, u - u0)
    self.duL = lambda X, u, u0 : u
    self.dxL = lambda X, u, u0 : 0.
    self.dudxL = lambda X, u : 0.0


    # Compute exact solutions 
    def _f(x):
      A,B,C,D = x[0], x[1], x[2], x[3]
      f1 =  A*np.sin(B)/4 + C*np.sin(D) - self.X0[0,0]
      f2 = -A*np.cos(B)/4 + C*np.cos(D) - self.X0[1,0]
      f3 =  A*np.sin(B + tf)/4 - A*tf*np.cos(B + tf)/2 + C*np.sin(tf + D) - self.XT[0, 0]
      f4 = -A*np.cos(B + tf)/4 + A*tf*np.sin(B + tf)/2 + C*np.cos(tf + D) - self.XT[1, 0]
      return np.array([f1, f2, f3, f4])
    sol = optimize.root(_f, np.random.rand(4), method='hybr')
    _A, _B, _C, _D = sol.x[0], sol.x[1], sol.x[2], sol.x[3]

    self.x_true = lambda t : \
      np.array([[ _A*np.sin(_B+t)/4 - _A*t*np.cos(_B+t)/2 + _C*np.sin(t+_D), \
                  -_A*np.cos(_B+t)/4 + _A*t*np.sin(_B+t)/2 + _C*np.cos(t+_D) ]]).T
    self.u_true = lambda t : np.array(_A * np.sin(_B + t))

    
    # things to plot during training
  def plot_prediction(self, algo, ax = None, cax = None, fig = None):
    """
    input : algo, ax, number of test points 
    output : test inputs for plottings x_in and x_out 
    """
    

    x_in  = algo.mesh_X.points
    x_out = np.zeros((self.M, algo.mesh_X.n))

    for it, t in enumerate(x_in):
      x_out[:, it] = algo.all_X[it, :, 0]

    
    x_e   = np.zeros((self.M, algo.test_points))
    for it,t in enumerate(np.linspace(0., self.T, algo.test_points)):
      x_e[:, it]   = self.x_true(t).squeeze()
    
    if ax is not None:
      ax.cla()
      ax.set_title(r'Function $\mathbf{X}_t$')
      ax.set_xlabel(r'$X_t^1$')
      ax.set_ylabel(r'$X_t^2$')
      
      # data 
      ax.plot(self.X0[0,0], self.X0[1,0], 'ro', label='start point')
      ax.plot(self.XT[0,0], self.XT[1,0], 'go', label='end point')

      # true solution
      ax.plot(x_out[0,:], x_out[1,:], 'x--', label='approx X_t')
      ax.plot(x_e[0,:],     x_e[1,:], '-', label='exact  X_t')
      
      ax.legend()

    return x_in, x_out

    
  # plot controls
  def plot_U(self, algo, ax, title = 'Controls'):   
    ax.cla()
    thin_mesh = np.linspace(0., self.T, algo.test_points)    
    ax.plot(thin_mesh, [self.u_true(t) for t in thin_mesh], '-', label='exact')
    for j in range(self.sizeu):
      ax.plot(algo.mesh_U.points, algo.u[:, j, 0],  'x--', label='approx')
    ax.set_title(title)
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'$\theta^*_t$')
    ax.legend()
