import sys
import numpy as np
from scipy import optimize
from ode   import ODEsolve2
from Mesh import *

# Mother class
class OptimalControlProblem:
  def __init__(self):

    self.d          = None   # dimension of the state vector
    self.sizeu      = None   # dimension of the control vector
    self.f          = None   # dynamic
    self.phi        = None
    self.L          = None
    self.u_bound    = None
    self.cost       = None
    self.N          = None
    self.M          = None
    

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

    
#
# Neural Network:   X'(t) = f(t,x,W,b) = tanh(W @ x + b)
#
############################################################################################
    
class NeuralNet(OptimalControlProblem):

  def __init__(self, d, K, FinalTime):
    """
    Defines a neural network 
    """
    super ().__init__()    
    assert FinalTime > 0.
    self.T = FinalTime
    
    self.d = d
    self.K = K

    # training data 
    self.X0      = None 
    self.XT      = None

    # test data    
    self.u_bound = None
    self.sizeu   = d*d+d
      
    def reshaper(u):
      """
      Convert a parameter vector to a matrix W and a vector b
      """
      W = np.array(u)[:self.d*self.d].reshape(self.d, self.d)
      b = np.array(u)[self.d*self.d:].reshape(self.d, 1)
      return W, b
    
    self.reshaper = reshaper

    def shaper(W, b):
      n, m = W.shape      
      assert n == self.d
      assert m == self.d

      n, m = b.shape
      assert n == self.d
      assert m == 1

      u = np.zeros(self.sizeu)
      u[:self.d*self.d] = W.reshape(self.d*self.d)
      u[self.d*self.d:] = b.reshape(self.d)

      return u
    
    self.shaper = shaper
    
    # For Flows
    def f(t, X, u):
      """
      input: 
      - X = (t, d, K) 
      - u = d*(d+1)
      output:
      f = (t, d, K)
      """
      W, b = self.reshaper(u)
      return np.tanh( W @ X + b)
    
    
    def dxf(t, X, u):
      """
      input: 
      - X = (d, K) 
      - u = d*(d+1)
      output:
      dxf = (K, d, d)
      """
      d, K = X.shape
      W, b = self.reshaper(u)
      tanhp = (1.0 - np.tanh( W @ X + b )**2).T.reshape(K, d, 1)
      return W * tanhp


    # For hamiltonian    
    def duf(t, X, u):
      """
      output: (K, d, s)
      """
      d, K = X.shape
      s = self.sizeu
      W, b = self.reshaper(u)

      F = 1.0 - np.tanh(W @ X + b)**2

      A = np.zeros((K, d, s))
      A[:, :, d*d:] = np.eye(d,d)
      for i in range(d):
        A[:, i, i*d:(i+1)*d] = X.T
      
      return A * F.T.reshape(K, d, 1)
    
    def dudxf(t, X, u):
      """
      input  : X: (d, K)
               u: (s)

      output : (K, s, d, d)
      """
      s = u.size
      W, b = self.reshaper(u)

      A = np.zeros((K, s, d, d))

      tanhp = 1 - np.tanh(W @ X + b)**2            # d, K
      tanhpp = - 2 * tanhp * np.tanh(W @ X + b)    # d, K
      
      # UNOPTIMIZED !
      for k in range(K):
        for i in range(d):          
          A[k, d*d + i, i, :] = W[i, :] * tanhpp[i, k]          
          for j in range(d):
            si = i*d + j
            A[k, si, i, :] = X[:,k] * A[k, d*d + i, i, :]
            A[k, si, i, j] += tanhp[i, k]
 
      return A
    
    self.f = f
    self.dxf = dxf
    self.duf = duf
    self.dudxf = dudxf

    self.eta = 1e-6
    self.L = lambda X, u, u0 : self.eta * 0.5 * np.inner(u - u0,u - u0)
    self.dxL = lambda X, u, u0 : 0. 
    self.duL = lambda X, u, u0 : self.eta * (u - u0)
    self.dudxL = lambda X, u : np.zeros((X.size, u.size))

    # output function : mean over components
    self.g   = lambda all_X : np.sum(all_X, axis=0) / self.d       # all_X is d x K; returns 1 x K
    self.dxg = lambda all_X : np.ones(all_X.shape)  / self.d       # g(X) is 1 x K, dxg is d x K
    
    # X is d x K
    self.phi_expr = str('0.5 * (sum(X^j(T)) - XT^j)')
    self.phi      = lambda all_X, XT : ( self.g(all_X) - XT[0, :] )**2 
    self.sum_phi  = lambda all_X, XT : np.mean(self.phi(all_X, XT)) #/ self.K              # 1
    self.dxphi    = lambda all_X, XT : 2 * self.dxg(all_X) * ( self.g(all_X) - XT[0, :]) # d x K
    #self.dxphi      = lambda all_X, XT : 2 * self.dxg(all_X) * np.sum(self.g(all_X) - self.XT[0,:]) / self.K  # d x K

  # plot U 
  def plot_U(self, algo, ax, title = 'Controls', style = 'classic'):
    """
    U is a matrix of size n_time_steps x sizeu
    """
    ax.cla()
    U = algo.u[:, :, 0]
    
    n, s = U.shape
    assert n == algo.mesh_U.n
    assert s == self.sizeu

    ax.set_title(title)
    
    if style == 'classic':
      
      assert n == algo.mesh_U.points.size
      
      ax.cla()
      ax.set_xlabel(r'Time $t$')
      ax.set_ylabel(r'$\theta^*_t$')
      ax.set_title('Controls')

      mesh_U_opt = np.linspace(0.0, self.T, algo.test_points)
      
      for ind_s in range(s):

        if ind_s < self.d * self.d:
          marker = ['b-','b-x']
        else:
          marker = ['r-','r-x']
          
        # true (if exists)
        if self.u_true is not None:
          ax.plot(mesh_U_opt, [self.u_true(t)[ind_s] for t in mesh_U_opt], marker[0])
          
        # approx
        ax.plot(algo.mesh_U.points, U[:, ind_s], marker[1])
        
      #ax.legend()
      
    elif style == 'nn':
      
      from matplotlib.patches import Circle
      # drawing parameters 
      cr = 0.1 # circle radius
      cs = 0.1 # circle space
      hx = 2*cr + cs
      no_test_points = algo.test_points

      # shorthands
      d = self.d

      # Draw neural net      
      X = np.zeros((d, n+1))
      Y = np.zeros((d, n+1))

      # layers
      for i in range(0, n+1):
        X[:,i] = i*hx
        Y[:,i] = np.linspace(2*cr, 2.0, d)

      ymin, ymax = np.min(Y) - cr, np.max(Y) + cr
      xmin, xmax = np.min(X) - cr, np.max(X) + cr

      ax.set_xlim(xmin, xmax)
      ax.set_ylim(ymin, ymax)

      # draw lines 
      for i in range(0, n+1):
        for j in range(d):
          x = X[j,i]
          y = Y[j,i]        

          if i > 0:
            W, b = self.reshaper(U[i-1, :])
            wmax = np.max(W)
            wmin = np.min(W)
            for k in range(d):
              w = np.abs((W[k, j] - wmin)/(wmax - wmin))
              xk, yk = X[k, i-1], Y[k, i-1]
              ax.plot([x,xk], [y, yk], '-', color='blue', alpha=w)

      # draw circles
      for i in range(0, n+1):
        for j in range(d):
          x = X[j,i]
          y = Y[j,i]        
          ax.add_artist(Circle((x, y), cr, color='blue', fill=False))
    else:
      raise Exception('Unknown style.')     
    


#
# Neural net test cases 
#

# 1d real function    
class NN_func1d(NeuralNet):
  
  def __init__(self, d, K, T, f, xbounds, width=0.0):
    """
    Cas test de la marche avec un volume 
    """
    super().__init__(d, K, T)

    # dimensions of the input/output data spaces 
    self.N = 1
    self.M = 1

    self.u_bound = [(-1.0, 1.0) for k in range(self.sizeu)]
    self.u_true  = None
    self.xmin, self.xmax = xbounds[0], xbounds[1]

    # trial set 
    Xr = np.linspace(self.xmin, self.xmax, K)
    randvalues = np.load('randvals.npy')[:Xr.size]
    self.X0 = Xr * np.ones((d, K))
    self.XT = np.array([f(x) - width + 2*width*r for x,r in zip(Xr,randvalues) ]) * np.ones((d,K))

    # exact output 
    self.Xi = np.array([f(x) for x in Xr ]) * np.ones((d,K))
    
    # test set
    # Xr = np.linspace(self.xmin, self.xmax, 2*K) # load un np array random et resize
    Xr = self.xmin + (self.xmax-self.xmin) * np.random.rand(2*K) # load un np array random et resize
    
    self.X0_test = Xr * np.ones((d, Xr.size))
    self.XT_test = np.array([f(x) - width + 2*width*np.random.rand() for x in Xr ]) * np.ones((d, Xr.size))

    self.Xi_test = np.array([f(x) for x in Xr ]) * np.ones((d, Xr.size))

    
  def plot_prediction(self, algo, ax = None, cax = None, fig = None):
    """
    input : algo, ax, number of test points 
    output : test inputs for plottings x_in and x_out 
    """
    no_test_points = algo.test_points
    
    x_in = np.linspace(self.xmin, self.xmax, no_test_points)
    x_out = np.zeros(no_test_points)
    
    x0 = np.ones((self.d, no_test_points)) * x_in
    #f  = lambda t, X: self.f(t, X, algo.P1Interp(t, algo.mesh_U, algo.u))
    f  = lambda t, X: self.f(t, X, algo.P1Interp(t, algo.mesh_U, algo.lowest_u))
    all_X, all_Xp = ODEsolve2(algo.mesh_X, f, x0, algo.solve_method) # returns n x d x K objects
    x_out = self.g(all_X[-1, :, :])

    # plot test curve
    if ax is not None:
      ax.cla()
      ax.set_title('Data set and best test plot')
      ax.plot(self.X0[0,:], self.XT[0,:], 'o',label='data', alpha=0.1)
      ax.plot(x_in, x_out, '--', label='test')
      ax.legend()

    return x_in, x_out

  

#
# One dim case where the control is known 
#
class NN_1d_exact(NeuralNet):

  def __init__(self, d, K, T, xbounds, n_sample = 100):
    """
    Cas test où le controle est approximativement bien connu
    """
    super().__init__(d, K, T)

    # dimensions of the input/output data spaces 
    self.N = 1
    self.M = 1

    self.xmin, self.xmax = xbounds[0], xbounds[1]
    Xr = np.linspace(self.xmin, self.xmax, K)
        
    self.u_bound = [(-1.0, 1.0) for k in range(self.sizeu)]
  
    #
    self.n_sample = n_sample
    self.u_true = lambda t : np.sin(3*t) * np.ones(self.sizeu)
    #self.u_true = lambda t : self.shaper(-0.5 * np.eye(d,d), 0.0 * np.ones((d,1)))
    
    # build data using u_exact

    # trial 
    ThX = Mesh(0., T, n_sample)
    self.X0 = Xr * np.ones((d, K))
    
    f = lambda t, X : self.f(t, X, self.u_true(t))
    all_x, all_xp = ODEsolve2(ThX, f, self.X0, 'euler')
    self.XT = all_x[-1, :, :]
    self.Xi = self.XT

    # test
    Xr = self.xmin + (self.xmax-self.xmin) * np.random.rand(2*K) # load un np array random et resize
    
    self.X0_test = Xr * np.ones((d, Xr.size))
    all_x, all_xp = ODEsolve2(ThX, f, self.X0_test, 'euler')
    self.XT_test = all_x[-1, :, :]
    self.Xi_test = self.XT_test

    

  def plot_prediction(self, algo, ax = None, cax = None, fig = None):
    """
    input : algo, ax, number of test points 
    output : test inputs for plottings x_in and x_out 
    """
    no_test_points = algo.test_points
    
    x_in = np.linspace(self.xmin, self.xmax, no_test_points)
    x_out = np.zeros(no_test_points)
    
    x0 = np.ones((self.d, no_test_points)) * x_in
    f  = lambda t, X: self.f(t, X, algo.P1Interp(t, algo.mesh_U, algo.lowest_u))
    all_X, all_Xp = ODEsolve2(algo.mesh_X, f, x0, algo.solve_method) # returns n x d x K objects
    x_out = self.g(all_X[-1, :, :])

    # plot test curve
    if ax is not None:
      ax.cla()
      ax.set_title('Data set and best test plot')
      ax.plot(self.X0[0,:], self.XT[0,:], 'o',label='data', alpha=0.1)
      ax.plot(x_in, x_out, '--', label='test')
      ax.legend()

    return x_in, x_out    



    
# classification
class NN_2d_classif(NeuralNet):
 
  def __init__(self, d, K, T, f, xbounds, ybounds):
    """
    Cas test de la marche avec un volume 
    """    
    super().__init__(2*d, K, T)

    # dimensions of the input/output data spaces 
    self.N = 2
    self.M = 1

    # output function : mean over components
    self.g   = lambda all_X : np.sum(all_X, axis=0) / self.d        # all_X is d x K; returns 1 x K
    self.dxg = lambda all_X : np.ones(all_X.shape)  / self.d       # g(X) is 1 x K, dxg is d x K
    
    self.u_true = None
    self.u_bound = [(-2.0, 2.0) for k in range(self.sizeu)]
    
    self.xmin, self.xmax = xbounds[0], xbounds[1]
    self.ymin, self.ymax = ybounds[0], ybounds[1]
    
    Xr = self.xmin + (self.xmax-self.xmin) * np.random.rand(K)
    Yr = self.ymin + (self.ymax-self.ymin) * np.random.rand(K)

    # sizes of the hidden layers
    self.d = 2*d 
    self.X0 = np.ones((self.d, K))
    self.X0[:d,:] *= Xr
    self.X0[d:,:] *= Yr
    
    self.XT = np.array([ f(x,y) for x,y in zip(Xr,Yr) ]) * np.ones((self.d, K))    
    self.Xi = self.XT 

    Xr = self.xmin + (self.xmax-self.xmin) * np.random.rand(2*K)
    Yr = self.ymin + (self.ymax-self.ymin) * np.random.rand(2*K)

    # sizes of the hidden layers
    self.X0_test = np.ones((self.d, Xr.size))
    self.X0_test[:d,:] *= Xr
    self.X0_test[d:,:] *= Yr
    
    self.XT_test = np.array([ f(x,y) for x,y in zip(Xr,Yr) ]) * np.ones((self.d, Xr.size))    
    self.Xi_test = self.XT_test 
    
    
  def plot_prediction(self, algo, ax = None, cax = None, fig = None):
    """
    input : algo, ax, number of test points 
    output : test inputs for plottings x_in and x_out 
    """

    no_test_points = algo.test_points
    
    # compute test curve
    X,Y = np.meshgrid(np.linspace(self.xmin, self.xmax, no_test_points), \
                      np.linspace(self.ymin, self.ymax, no_test_points))    

    #
    NN = no_test_points * no_test_points
    x_in = np.zeros((self.N, NN))
    x_in[0,:] = X.reshape(NN)
    x_in[1,:] = Y.reshape(NN)

    x_out = np.zeros((self.M, NN))

    # generate test 
    x0 = np.ones((self.d, NN))
    half = int(self.d/2)
    x0[:half, :] *= x_in[0,:]
    x0[half:, :] *= x_in[1,:]

    f  = lambda t, X: self.f(t, X, algo.P1Interp(t, algo.mesh_U, algo.lowest_u))
    #f  = lambda t, X: self.f(t, X, algo.P1Interp(t, algo.mesh_U, algo.u))
    all_X, all_Xp = ODEsolve2(algo.mesh_X, f, x0, algo.solve_method)
    #x_out = self.g(all_X[-1, :, :])
    x_out = np.heaviside(self.g(all_X[-1, :, :]) - 0.5, 1.0)

    # plot test curve
    if ax is not None:
      ax.cla()
      cf = ax.tricontourf(x_in[0,:], x_in[1,:], x_out, cmap="RdBu_r")

      cax.cla()
      fig.colorbar(cf, cax=cax)
      
      # plots 0
      jup = np.where(self.XT[0,:] > 0.9)
      ax.plot(self.X0[0,jup], self.X0[-1,jup], 'o', color='red', label='0')
    
      # plots 1
      jup = np.where(self.XT[0,:] < 0.1)
      ax.plot(self.X0[0,jup], self.X0[-1,jup], 'x', color='black', label='1')    
      
      ax.set_title('Data set and best test plot')

    return x_in, x_out


#
# Convolutional network : X'(t) = tanh(W # x + b)
#
############################################################################################

class ConvNet(OptimalControlProblem):

  def __init__(self, d, K, T):
    """
    Defines a convolutional network
    - We assume input images are made of dxd pixels 
    - K is the number of samples
    - T is the final time
    """
    
    super().__init__()

    from scipy import ndimage

    assert np.mod(d, 2) == 0
    assert FinalTime > 0.
    
    self.T = T             # some final time
    self.d = d*d           # number of pixels of an dxd image
    self.d_img = d         # side dimension d of an dxd image 
    self.K = K             # number of samples
    self.dk = 3            # filter size

    self.u_bound = None
    self.sizeu   = self.dk*self.dk + 4

    # training data
    self.X0 = None
    self.XT = None

    print('Init ConvNet:')
    print('--------------------------------:')
    print('T = {}'.format(self.d_img))
    print('dim vector X = {}'.format(self.d))
    print('no.sample K = {}'.format(K))
    print('size kernel dk = {}'.format(self.dk))


    def shaper(W,b):
      """
      Convert: W,b ----> u
      """
      n, m = W.shape
      assert n == self.dk
      assert m == self.dk

      n, m = b.shape
      assert n == self.d_img
      assert m == self.d_img

      u = np.zeros(self.sizeu)
      u[:self.dk*self.dk] = W.reshape(self.dk*self.dk)
      u[self.dk*self.dk:] = b.reshape(self.d_img*self.d_img)

      return u

    
    def reshaper(u):
      """
      Convert:  u  ---->   W, b
      """
      _a = np.array(np.linspace(0, 1, self.d_img) < 0.5, dtype = 'float')
      _b = np.array(np.linspace(0, 1, self.d_img) > 0.5, dtype = 'float')

      up_left = np.outer(_a, _a)
      up_right = np.outer(_a, _b)
      down_left = np.outer(_b, _a)
      down_right = np.outer(_b, _b)
      
      W = np.array(u)[:self.dk*self.dk].reshape(self.dk, self.dk)
      b = np.array(u)[self.dk*self.dk]  *up_left     \
        + np.array(u)[self.dk*self.dk+1]*up_right  \
        + np.array(u)[self.dk*self.dk+2]*down_left \
        + np.array(u)[self.dk*self.dk+3]*down_right
      return W, b
    
    self.reshaper = reshaper

    def f(t, X, u):
      """
      Dynamic x'(t) = f(t,x,u)

      """
      W, b = self.reshaper(u)
      return np.tanh(ndimage.convolve(X.reshape(self.d_img, self.d_img), W) + b)

    self.f = f

    # TODO: dxf
    
    # Lagrangian
    self.L = lambda X, u, u0 : self.eta * 0.5 * np.inner(u - u0,u - u0)
    
    # TODO: dxL
    self.g   = lambda all_X : np.sum(all_X, axis=0) / self.d
    
    # Cost
    self.phi_expr = str('0.5 * (sum(X^j(T)) - XT^j)')
    self.phi = lambda all_X, XT : (self.g(all_X) - XT[0,:])**2
    self.sum_pi = lambda all_X, XT : np.mean(self.phi(all_X, XT))
    self.dxphi = lambda all_X, XT : 2 * self.dxg(all_X) * ( self.g(all_X) - XT[0, :])


class NN_general_data(NeuralNet):

  def __init__(self, fn_Xin, fn_Xout, fn_Xin_test, fn_Xout_test, T):

    # Load training files
    print('DEBUG: loading data...')
    Xin = np.load(fn_Xin)
    Xout = np.load(fn_Xout)

    Xin_test = np.load(fn_Xin_test)
    Xout_test = np.load(fn_Xout_test)

    assert Xin.ndim == 2
    assert Xout.ndim == 2
    assert Xin_test.ndim == 2
    assert Xout_test.ndim == 2    
        
    # dimensions of the input/output data spaces    
    self.d, self.K = np.shape(Xin)
    self.N = self.d
    self.M = np.shape(Xout)[0]

    super().__init__(self.d, self.K, T)
    
    self.g   = lambda all_X : np.sum(all_X, axis = 0) / self.d
    self.dxg = lambda all_X : np.ones(all_X.shape) / self.d

    self.X0 = Xin
    self.XT = Xout
    self.Xi = self.XT

    self.xmin, self.xmax = 0.0, 1.0
    self.ymin, self.ymax = 0.0, 1.0    

    self.X0_test = Xin_test
    self.XT_test = Xout_test
    self.Xi_test = self.XT_test

    self.u_true = None
    self.u_bound = [(-1.0, 1.0) for k in range(self.sizeu)]


  def plot_prediction(self, algo, ax = None, cax = None, fig = None):
    """
    input : algo, ax, number of test points 
    output : test inputs for plottings x_in and x_out 
    """

    no_stations = 15


    # Test pred 
    x0 = self.X0_test
    f  = lambda t, X: self.f(t, X, algo.P1Interp(t, algo.mesh_U, algo.lowest_u))
    all_X, all_Xp = ODEsolve2(algo.mesh_X, f, x0, algo.solve_method) # returns n x d x K objects

    test_x_in  = self.X0_test
    test_x_out = self.g(all_X[-1, :, :])    

    # plot test curve
    if ax is not None:
      ax.cla()
      ax.set_title('Data set and best test plot')
      #for i in range(no_stations):
      i=0

      # test 
      ax.plot(test_x_in[2,i::no_stations], self.XT_test[0,i::no_stations], '-x', label='test data s={}'.format(i))  # prediction
      ax.plot(test_x_in[2,i::no_stations], test_x_out[i::no_stations], '--x', label='test pred s={}'.format(i))          # data
      
      # trial
      x0 = self.X0
      f  = lambda t, X: self.f(t, X, algo.P1Interp(t, algo.mesh_U, algo.lowest_u))
      all_X, all_Xp = ODEsolve2(algo.mesh_X, f, x0, algo.solve_method) # returns n x d x K objects

      trial_x_in  = self.X0
      trial_x_out = self.g(all_X[-1, :, :])
      
      ax.plot(trial_x_in[2,::10], self.XT[0,::10],    '-x', label='trial data') # data
      ax.plot(trial_x_in[2,::10], trial_x_out[::10], '--x', label='trial pred') # 
      
      ax.legend()

    return test_x_in, test_x_out    


  
