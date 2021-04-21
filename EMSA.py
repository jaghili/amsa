import sys
from mpi4py import MPI
import os
import numpy as np
from Mesh import * 
from ode import *
from copy import deepcopy
import scipy.optimize
import scipy.stats
import time

class EMSA:

  def __init__(self, OC, kmax, comm, settings, abn_mul = 1.0, rho_init = 1.0, ts = None):

    time_steps = comm.size if ts is None else ts

    self.rho_bounds   = [1.0, 5.0]
    self.rho          = rho_init
    self.oc           = OC
    self.ts_init      = time_steps
    self.solve_method = settings['ode_integrator']
    self.optimizer    = settings['optimizer']
    self.kmax         = kmax  
    self.comm         = comm
    self.test_points  = settings['adapt_max_nodes']
    self.amul         = abn_mul
    
    # init meshes
    self.mesh_X       = Mesh(0., OC.T, self.ts_init)
    self.mesh_P       = Mesh(0., OC.T, self.ts_init)    
    self.mesh_U       = Mesh(0., OC.T, self.ts_init)        

    # data
    self.best_u  = np.zeros(( self.mesh_U.n, self.oc.sizeu, 1 ))   # store optimal u (if available)
    self.init_u  = np.zeros(( self.mesh_U.n, self.oc.sizeu, 1 ))   # store u guess (for BFGS)
    self.last_aH = None                                            # store H landscape
    
    # Train variables
    self.u       = np.zeros(( self.mesh_U.n, self.oc.sizeu, 1 ))   # store current u

    # trial flows
    self.all_X  = np.zeros( (self.mesh_X.n,) + OC.X0.shape )    # size (n, d, K)
    self.all_Xp = np.zeros( (self.mesh_X.n,) + OC.X0.shape )    
    self.all_P  = np.zeros( (self.mesh_X.n,) + OC.X0.shape )
    self.all_Pp = np.zeros( (self.mesh_X.n,) + OC.X0.shape )

    # test flows
    self.all_X_test  = np.zeros( (self.mesh_X.n,) + OC.X0_test.shape )
    self.all_Xp_test = np.zeros( (self.mesh_X.n,) + OC.X0_test.shape )
    self.all_P_test  = np.zeros( (self.mesh_X.n,) + OC.X0_test.shape )
    self.all_Pp_test = np.zeros( (self.mesh_X.n,) + OC.X0_test.shape )


  # Linear P1 Interpolator
  def P1Interp(self, t, mesh, data):
    """
    Returns the value of the interpolated data at t
    """
    res = 0.0
    
    assert data is not None
    
    n, d, K = data.shape
    
    assert mesh.n == n
    
    t1, t2 = mesh.points[0], mesh.points[-1]

    # interpolate
    if t > t1 and t < t2:
      it     = mesh.find_idx(t) #
      assert it < mesh.n - 1      
      t1, t2 = mesh.points[it], mesh.points[it+1]
      a      = (t - t1)/(t2 - t1)
      res    = (1-a) * data[it, :, :]  + a * data[it+1, :, :]
    elif t >= t2:
      res = data[-1, :, :]
    else:
      res = data[0, :, :]
      
    return res 


  #
  # Integral of Regularizer
  def integralLagrangian(self):
    """
    returns the integral of L(X,U) over a given mesh
    """
    int_L = np.sum([self.oc.L(self.P1Interp(t, self.mesh_X, self.all_X),              \
                              self.u[it, :, 0],                                       \
                              0.) for it, t in enumerate(self.mesh_U.points) ]) / self.mesh_U.n
    
    return int_L

  
  #
  # Forward-Backward Integrators 
  def computeAllFlows(self):
    """
    Compute X, P and their derivatives
    """
    
    tic = time.time()
    
    # Forward
    x0 = self.oc.X0
    f  = lambda t, X: self.oc.f(t, X, self.P1Interp(t, self.mesh_U, self.u))
    self.all_X, self.all_Xp = ODEsolve2(self.mesh_X, f, x0, self.solve_method)
    
    # Backward
    pT = - self.amul * self.oc.dxphi(self.all_X[-1, :, :], self.oc.Xi)
    minus_dxH = lambda t, P : - self.dxH(t, self.P1Interp(t, self.mesh_X, self.all_X), P, \
                                         self.P1Interp(t, self.mesh_U, self.u))
    self.all_P, self.all_Pp = ODEsolve2(self.mesh_P, minus_dxH, pT, self.solve_method, backward = True)


    # Test
    
    # Forward
    x0 = self.oc.X0_test
    self.all_X_test, self.all_Xp_test = ODEsolve2(self.mesh_X, f, x0, self.solve_method)
    
    # Backward
    pT = - self.amul * self.oc.dxphi(self.all_X_test[-1, :, :], self.oc.XT_test)
    minus_dxH = lambda t, P : - self.dxH(t, self.P1Interp(t, self.mesh_X, self.all_X_test), P, \
                                         self.P1Interp(t, self.mesh_U, self.u))
    self.all_P_test, self.all_Pp_test = ODEsolve2(self.mesh_P, minus_dxH, pT, self.solve_method, backward = True)   
    
    # returns time
    toc = time.time()
    
    return toc-tic

  #
  # Lambda 
  def computeLambda(self, unew, uold):
    """
    unew and uold have to be both of size (n, s, 1)
    
    """
    K = self.oc.K    
    N = self.mesh_U.n

    mu = 0.0
    estim = 0.0
    for it in range(N):
      t = self.mesh_U.points[it]
      
      Unew = unew[it, :, 0]
      Uold = uold[it, :, 0]
      estim += np.linalg.norm(Unew - Uold)**2/N

      X = self.P1Interp(t, self.mesh_X, self.all_X)
      f_unew = self.oc.f(t, X, Unew) # d, K
      f_uold = self.oc.f(t, X, Uold)

      P = self.P1Interp(t, self.mesh_P, self.all_P)
      dxH_unew = self.dxH(t, X, P, Unew)
      dxH_uold = self.dxH(t, X, P, Uold)
    
      mu += (np.linalg.norm(np.sum(f_unew - f_uold, axis=1))**2 \
            + np.linalg.norm(np.sum(dxH_unew - dxH_uold, axis=1))**2)/K/N
    
    return np.sqrt(mu), np.sqrt(estim)


    
  #
  # Augmented Hamiltonians & derivatives 
  #
  def dxH(self, t, X, P, u):
    """
    input: 
    - X and P  = (d, K) 
    - u = d*(d+1)

    output:
    dxH = (d, K)
    """
    d, K = X.shape
    u0 = np.zeros(self.oc.sizeu)
    
    _dxH = P.T.reshape(K, 1, d) @ self.oc.dxf(t, X, u) - self.oc.dxL(X, u, u0)  
    return _dxH.reshape(K, d).T


  
  def dudxH(self, t, X, P, u):
    """
    input: X (d,K)
           P (d,K)
           u   (s)

    output: dudxH (K, d, s)
    """
    s = np.size(u)
    d, K = X.shape

    dudxH = (P.T.reshape(K, 1, 1, d) @ self.oc.dudxf(t, X, u)).reshape(K, s, d).transpose(0, 2, 1) 
    return dudxH


  
    
  def sum_aH(self, u, t):
    """
    Input : u of size s=d*(d+1)
    Output: augmented Hamiltonian (scalar)
    """
    u0 = np.zeros(self.oc.sizeu)
    
    X  = self.P1Interp(t, self.mesh_X, self.all_X) # renvoie X(t) : l'interpolé dans l'intervalle [tk, tk+1]
    Xp = self.P1Interp(t, self.mesh_X, self.all_Xp)
    P  = self.P1Interp(t, self.mesh_P, self.all_P)
    Pp = self.P1Interp(t, self.mesh_P, self.all_Pp)

    d, K = X.shape
    
    F = self.oc.f(t, X, u)
    H1 = np.sum(P * F) - self.amul * self.oc.L(X, u, u0)
    H2 = 0.5 * self.rho * np.sum((Xp - F)**2)
    H3 = 0.5 * self.rho * np.sum((Pp + self.dxH(t, X, P, u))**2)

    return H1 - H2 - H3

  def du_sum_aH(self, u, t):
    """
    input:  u   vector of size s = d*(d+1)
    output: daH vector of size s = d*(d+1)
    """
    u0 = np.zeros(self.oc.sizeu)
    
    X  = self.P1Interp(t, self.mesh_X, self.all_X)
    Xp = self.P1Interp(t, self.mesh_X, self.all_Xp)
    P  = self.P1Interp(t, self.mesh_P, self.all_P)
    Pp = self.P1Interp(t, self.mesh_P, self.all_Pp)

    d, K = X.shape

    F     = self.oc.f(t, X, u)         # (d, K)
    duF   = self.oc.duf(t, X, u)       # (K, d, s)
    dxH   = self.dxH(t, X, P, u)       # (d, K)
    dudxH = self.dudxH(t, X, P, u)     # (K, d, s)

    dH1 = np.sum(P.T.reshape(K, 1, d) @ duF, axis=0) - self.amul * self.oc.duL(X, u, u0)         # has to be of size s
    dH2 = - self.rho * np.sum((Xp - F).T.reshape(K, 1, d) @ duF, axis=0)
    dH3 = self.rho * np.sum((Pp + dxH).T.reshape(K, 1, d) @ dudxH, axis=0)

    return (dH1 - dH2 - dH3).squeeze()

  def adu_sum_aH(self, u, t):
    """
    finite difference approximation of the augmented Hamiltonian (faster?)
    """

    adH = np.zeros(self.oc.sizeu)

    eps = 1e-8 # nearly optimal
    
    for i in range(self.oc.sizeu):
      h = np.zeros(self.oc.sizeu)
      h[0] = 1.0
      adH[i] = (self.sum_aH(u + eps * h, t) - self.sum_aH(u, t)) / eps
    
    return adH

  
  def plot_aH(self, fig, ax, it, NN = 25):

    # plot to ax
    ax.cla()

    t = self.mesh_U.points[it]
        
    ax.set_title(r'H({:.2f}, u)'.format(t))        
    ax.set_ylabel(r'$u_2$')
    ax.set_xlabel(r'$u_1$')
        
    umin, umax = self.oc.u_bound[0]
    u1 = np.linspace(umin, umax, NN)
    u2 = np.linspace(umin, umax, NN)
    U1, U2 = np.meshgrid(u1, u2)
    Z1 = np.zeros((NN, NN))
    Z2 = np.zeros((NN, NN))
    #
    QX1 = np.zeros((NN, NN))
    QY1 = np.zeros((NN, NN))    
        
    for i in range(NN):
      for j in range(NN):
        uu = np.zeros(self.oc.sizeu)
        uu[:2] = np.array([ U1[i,j], U2[i,j] ])

        # Scalar field
        Z1[i,j] = self.sum_aH(uu, self.mesh_U.points[it])

        # Vector Field
        QQ = self.du_sum_aH(uu, self.mesh_U.points[it])
        QX1[i,j] = QQ[0]
        QY1[i,j] = QQ[1]

    QX1 *= 1.0/np.max(np.abs(QX1))
    QY1 *= 1.0/np.max(np.abs(QY1))           
            
    cs = ax.pcolormesh(U1, U2, Z1, cmap = "RdBu_r")
    # cs = ax.contourf(X, Y, z, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
    ax.plot(self.init_u[it, :, 0][0], self.init_u[it, :, 0][1], 'o', label='u_start') # u_init
    ax.plot(self.u[it, :, 0][0], self.u[it, :, 0][1], 'x', label='u_bfgs') # u_opt
    ax.quiver(U1, U2, QX1, QY1, units='width')
    ax.legend()
    
    cbar = fig.colorbar(cs)    

  def meshAdaptation(self, k, slope_lambda, settings):
    """
    Adapt the meshes X, P and U according to different strategies
    """

    # get settings 
    adapt_each        = settings['adapt_each']
    adapt_auto        = settings['adapt_auto']
    adapt_dyadic      = settings['adapt_dyadic']
    max_nodes_in_mesh = settings['adapt_max_nodes']
    no_layers_to_add  = settings['adapt_add_layers']
    comm              = self.comm
    
    nX = self.mesh_X.n
    nP = self.mesh_P.n
    nU = self.mesh_U.n

    assert nX == nP == nU
    assert nX > 1
    
    # adapt each k-iteration 
    if adapt_each > 0 and k % adapt_each == 0 and k > 1:

      nnew = min(nX + no_layers_to_add, max_nodes_in_mesh)
      
      if nX < max_nodes_in_mesh:
        
        self.mesh_X = Mesh(0., self.oc.T, nnew)
        self.mesh_P = Mesh(0., self.oc.T, nnew)
        self.mesh_U = Mesh(0., self.oc.T, nnew)
        
        #self.rho = self.rho_bounds[1] # 
        if comm.rank == 0:
          print('Arbitrary mesh adapt. -> {:d} nodes added, now having {:d} layers.'.format(no_layers_to_add, self.mesh_U.n))
          print('Rho -> {:.2f}'.format(self.rho))
          
    # adapt dyadically 
    elif adapt_dyadic > 0 and k % adapt_dyadic == 0 and k > 1:

      nnew = nX + nX - 1
      
      if nnew <= max_nodes_in_mesh:
        
        self.mesh_X.dyadic_refine()
        self.mesh_P.dyadic_refine()
        self.mesh_U.dyadic_refine()
        
        assert self.mesh_U.n == nnew
        assert self.mesh_P.n == nnew
        assert self.mesh_X.n == nnew

        #self.rho = self.rho_bounds[1] # rho max
        if comm.rank == 0:
          print('Dyadic mesh adapt. -> now having {:d} layers.'.format(self.mesh_U.n))
          print('Rho -> {:.2f}'.format(self.rho))
        
    # adapt automatically wrt to slope   
    elif adapt_auto > 0 and k > 1:
      
      if slope_lambda > 0:
        
        nnew = min(nX + no_layers_to_add, max_nodes_in_mesh) # check max

        if nX < max_nodes_in_mesh:
          self.mesh_X = Mesh(0., self.oc.T, nnew)
          self.mesh_P = Mesh(0., self.oc.T, nnew)
          self.mesh_U = Mesh(0., self.oc.T, nnew)

          #self.rho = self.rho_bounds[1] # rho max
          
          if comm.rank == 0:
            print('Auto mesh adapt.')
            print('-> slope = {:.2f}'.format(slope_lambda))
            print('-> {:d} nodes added, now having {:d} layers.'.format(nnew, self.mesh_U.n))          
            print('Rho -> {:.2f}'.format(self.rho))

        
    
  #
  # Hamiltonian Maximization 
  #
  def maximizeHamiltonian(self, it, t, k, bfgs_maxiter = 10, cmaes_maxiter = 10):
    """
    Maximize Hamiltonian at t
    """
    
    sum_aH_val = 0.0    
    niter = 0
    dtheta = 0.0
    H_rand_presel = 0.0
    
    # Functions (and its derivative) to optimize
    minus_aH = lambda u :  - self.sum_aH(u, t)
    du_minus_aH = lambda u : - self.du_sum_aH(u, t)

    #
    # preselection
    #
    umin, umax = self.oc.u_bound[0][0], self.oc.u_bound[0][1]

    # Stack init
    theta_stack = list()
    theta_stack.append(self.lowest_u[it, :, 0])
    
    for eps in [1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
      for i in range(25):
        theta_stack.append(self.lowest_u[it, :, 0] + eps * (umin + (umax-umin) * np.random.rand(self.oc.sizeu)))
        theta_stack.append(eps * (umin + (umax-umin) * np.random.rand(self.oc.sizeu)))

    if self.oc.u_true is not None:      
      theta_stack.append( self.oc.u_true(t) )

    # choose
    ind_best = np.argmax([ self.sum_aH(u, t) for u in theta_stack ])
    theta0 = theta_stack[ind_best]
    self.init_u[it, :, 0] = theta0
    
    # 
    H_presel = self.sum_aH(theta0, t)
    theta_opt = theta0

    #
    # BFGS only
    #    
    if self.optimizer == 'bfgs':

      # Optimize with BFGS
      sol = scipy.optimize.minimize(minus_aH,
                                    theta0,
                                    bounds = self.oc.u_bound,
                                    options = {'maxiter': bfgs_maxiter, 'gtol': 1e-300, 'ftol': 1e-300},
                                    jac = du_minus_aH,
                                    method='L-BFGS-B')
      theta_opt = sol.x
      niter += sol.nit            
      
    #
    # BFGS + CMA-ES 
    #          
    elif self.optimizer == 'cmaes':

      import cma      

      # CMA-ES
      theta_opt, es = cma.fmin2(minus_aH,
                                theta_opt,
                                0.1,
                                #gradf = du_minus_aH,
                                options = {'verbose' : -9, 'maxiter': cmaes_maxiter})
      niter += es.result[4]        
      # End 
      
    else:
      theta_opt = theta0
      niter = 0
      
    return theta_opt, niter, ind_best, np.max(np.abs(theta0 - theta_opt))
    #return theta_opt, niter, toc-tic, H_presel, np.max(np.abs(theta0 - theta_opt))

    
  #
  # Training
  #
  def train(self, settings, fig = None , Axes = None, cax = None):
    """
    mesh is a Mesh object 
    """
    
    # init 
    show_plots        = settings['show_plots']
    adapt_each        = settings['adapt_each']
    adapt_auto        = settings['adapt_auto']
    adapt_dyadic      = settings['adapt_dyadic']
    max_nodes_in_mesh = settings['adapt_max_nodes']
    no_layers_to_add  = settings['adapt_add_layers']
    
    comm              = self.comm            

    stats             = np.zeros((self.kmax, 10, comm.size))

    # shorthands
    T = self.oc.T    
    K = self.oc.K
    d = self.oc.d

    # init meshes
    self.mesh_X       = Mesh(0., T, self.ts_init)
    self.mesh_P       = Mesh(0., T, self.ts_init)    
    self.mesh_U       = Mesh(0., T, self.ts_init)

    self.u            = 0.1 * np.ones((self.mesh_U.n, self.oc.sizeu, 1))
    self.lowest_u     = np.zeros((self.mesh_U.n, self.oc.sizeu, 1)) # store last argmin J
    self.best_u       = np.zeros((self.mesh_U.n, self.oc.sizeu, 1)) # theoretical best control
    self.init_u       = np.zeros((self.mesh_U.n, self.oc.sizeu, 1)) # store BFGS init control
    
    # Initialize u with optimal u, if available
    if self.oc.u_true is not None:
      
      for it, t in enumerate(self.mesh_U.points):
        self.best_u[it, :, 0] = self.oc.u_true(t)
        
      self.u = self.best_u
      if comm.rank == 0:
        print('Warning : AMSA initialized with optimal u.')
      
    # Init uold 
    uold          = deepcopy(self.u) # copy and not ref copy
    lowest_u_old  = None  # cause some problems !!
    mesh_lowest_u = None
    
    #
    # Loop on k
    #
    err = np.inf
    
    for k in range(0, self.kmax):
      
      if comm.rank == 0:
        print('\n\n---- k = {:d} ----'.format(k))

      # 
      # Mesh adaptation 

      # compute slope of lambda
      a = 0.
      no_points_avg_mu = 10
      if k > no_points_avg_mu:
        xxx = np.arange(k - no_points_avg_mu, k)
        yyy = stats[k-no_points_avg_mu:k, 5, comm.rank]   # 4: cost ; 5: lambda
        a, b, r_value, p_value, std_err = scipy.stats.linregress(xxx, yyy)
        if comm.rank == 0:
          print('mu slope :', comm.rank, a, b, r_value, p_value, std_err)        

      # adapt mesh
      mesh_U_old = deepcopy(self.mesh_U)
      self.meshAdaptation(k, a, settings)

      # reinterpolate u
      if self.mesh_U.n > mesh_U_old.n:        
        self.u        = np.zeros((self.mesh_U.n, self.oc.sizeu, 1))
        self.lowest_u = np.zeros((self.mesh_U.n, self.oc.sizeu, 1))
        

        if comm.rank == 0 :
          print('Reinterpolating...')
          print('old lowest_u', lowest_u_old.shape)
          print('lowest u', self.lowest_u.shape)
          print('mesh U.n', self.mesh_U.n)
          print('mesh old U.n', mesh_U_old.n)
          print('Check : ', mesh_U_old.n, lowest_u_old.shape[0])
        
        for it, t in enumerate(self.mesh_U.points):
          self.u[it, :, :]        = self.P1Interp(t, mesh_U_old, uold)
          self.lowest_u[it, :, :] = self.P1Interp(t, mesh_lowest_u, lowest_u_old) # !!!
          
          
      #
      # Integrators
      #
      
      if comm.rank == 0:
        print('Rho ={:.2f}'.format(self.rho))
        print('Mesh sizes:\t X={:d}\t P={:d}'.format(self.mesh_X.n, self.mesh_P.n))

      integrators_time       = self.computeAllFlows()
      stats[k, 0, comm.rank] = integrators_time
      comm.Barrier()

      if comm.rank == 0:
        print('[0] Computed all flows with {} in {:0.2f}s.'.format(self.solve_method, integrators_time))
        
        print('\nMaximize Hamiltonian using {:d} procs...'.format(comm.size))
        print(('{:<12} '*11).format('mpirank', 't', 'aH', 'du_aH', 'th_aH_max', 'niter', 'time', 'aH_presel', 'aH(uopt)', 'du_aH(uopt)', '|u_bfgs-u_init|'))

        
      #
      #   Hamiltonian Maximization
      #
      
      # On each time interval
      bfgs_max  = settings['bfgs_maxiter']
      cmaes_max = settings['bfgs_maxiter']
      
      uold = deepcopy(self.u) # backup old       
      self.u = np.zeros((self.mesh_U.n, self.oc.sizeu, 1))
      self.init_u = np.zeros((self.mesh_U.n, self.oc.sizeu, 1)) # re-init ( in case of adaptivity )

      if self.oc.u_true is None:
        self.best_u = np.zeros((self.mesh_U.n, self.oc.sizeu, 1))

        
      for it in range(comm.rank, self.mesh_U.n, comm.size):

        # counter      
        stats[k, 3, comm.rank] += 1 # counter
        
        # Optimization
        t = self.mesh_U.points[it]
        tic = time.time()
        u_opt, niter, H_presel, dtheta = self.maximizeHamiltonian(it, t, bfgs_max, cmaes_max)
        toc = time.time()

        # 
        max_aH = np.sum(np.abs(self.all_P[it, :, :])) # theoretical max
        
        # save stats
        stats[k, 1, comm.rank] += toc-tic
        stats[k, 2, comm.rank] += niter # maximization iterations
        
        # update u locally
        self.u[it, :, 0] = u_opt    # self.u is n x s x 1    
        aH = self.sum_aH(u_opt, t).squeeze()
        du_aH = np.linalg.norm(self.du_sum_aH(u_opt, t).squeeze())

        # exact u
        aH_uopt = 0.0
        du_aH_uopt = 0.0
        if self.oc.u_true is not None:
          aH_uopt = self.sum_aH(self.oc.u_true(t), t)          
          du_aH_uopt = np.linalg.norm(self.du_sum_aH(self.oc.u_true(t), t))
        
        return_string = '{:<12d} {:<12.2f} {:<12.2E} {:<12.2E} {:<12.2E} {:<12d} {:<12.2f} {:<12.2E} {:<12.2E} {:<12.2E}'
        print(return_string.format(comm.rank, t, aH, du_aH, max_aH, niter, H_presel, aH_uopt, du_aH_uopt, np.linalg.norm(u_opt - self.init_u[it, :, 0])))
        
      comm.Barrier()
      # End Hamiltonian maximization
      
      # reduce u over all processes
      self.u      = comm.allreduce(self.u,      op = MPI.SUM)
      self.init_u = comm.allreduce(self.init_u, op = MPI.SUM)


      #
      # Compute training costs
      phi_x_trial = self.oc.sum_phi(self.all_X[-1, :, :], self.oc.Xi)
      phi_e = self.oc.sum_phi(self.oc.Xi, self.oc.Xi) # always zero ?
      lag_u = self.integralLagrangian()
      err = phi_x_trial + lag_u
      mu, dtheta = self.computeLambda(self.u, uold)

      #
      # Compute test costs
      phi_x_test = self.oc.sum_phi(self.all_X_test[-1, :, :], self.oc.Xi_test)


      # store
      comm.Barrier()
      
      stats[k, 4, :] = err
      stats[k, 5, :] = mu
      stats[k, 6, :] = self.oc.T/self.mesh_X.n
      stats[k, 7, :] = self.oc.T/self.mesh_P.n
      stats[k, 8, :] = self.oc.T/self.mesh_U.n
      stats[k, 9, :] = phi_x_test

      
      # Keep track of best u
      if k > 1 and err <= np.min(stats[:k, 4, 0]):
          self.lowest_u = self.u
          lowest_u_old  = deepcopy(self.lowest_u)
          mesh_lowest_u = deepcopy(self.mesh_U)
          if comm.rank == 0:
            print('Saving this control.')
            
            
      # Adapt rho
      if settings['adapt_rho'] > 0:
        #self.rho = max(0.95 * self.rho, self.rho_bounds[0])   # faire decroitre rho en temps normal      
        self.rho = min(1.05 * self.rho, self.rho_bounds[1])  # augmente rho 

      # Print costs 
      if comm.rank == 0:
        
        print('Costs:')
        print('phi trial= {:.4E}/{:.4E}\tphi test= {:.4E}/{:.4E}\t L={:.4E}\t err={:.4E}\t mu={:.4E}'.format(phi_x_trial, phi_e, phi_x_test, phi_e, lag_u, err, mu))        
                  
      
      
      #
      # Live plots
      #
      if show_plots and comm.rank == 0:

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

        plt.ion()
        
        assert fig is not None        
        assert Axes is not None
        # [[ax_costs, ax_Lambda], [ax_test, ax_U], [ax_aH1, ax_aH2]] = Axes
        [[ax_costs, ax_Lambda], [ax_test, ax_U]] = Axes

        print('Showing plots...')
        _ , _ = self.oc.plot_prediction(self, ax_test, cax, fig)    # test
        self.oc.plot_U(self, ax_U)                                  # controls

        # Cost 
        ax_costs.cla()
        ax_costs.set_yscale('log')
        #ax_costs.set_ylim(1e-16, None)
        ax_costs.plot(stats[:k, 4, comm.rank], label='train cost')
        ax_costs.plot(stats[:k, 9, comm.rank], label='test  cost')        
        ax_costs.plot(stats[:k, 8, comm.rank], '--', label='mesh size U')
        if phi_e + lag_u > 1e-10:
          ax_costs.plot((phi_e + lag_u) * np.ones(k), label='optim. train cost')
        ax_costs.legend()
        
        # Lambda 
        ax_Lambda.cla()
        ax_Lambda.set_yscale('log')
        ax_Lambda.plot(stats[:k, 5, comm.rank], label='Lambda')
        ax_Lambda.plot(stats[:k, 8, comm.rank], '--', label='mesh size U')
        ax_Lambda.legend()      
        
        # Plot everything 
        plt.pause(0.001)
        plt.ioff()

      # End 
      comm.Barrier()        
      k += 1

    stats = comm.allreduce(stats, op = MPI.SUM)
    # end loop on k    
    return stats


  
  
  def error_study(self, no_iters, settings):
    """
    Computes all kind of average data approximated with no_iters samples
    - the cost,
    - mu,
    - control values,
    - prediction values.
    """        
    
    assert no_iters > 0

    comm = self.comm

    show_plots = settings['show_plots']
    dirName = settings['output_dir']

    fig = None
    Axes = None
    cax = None
    
    if show_plots:
      
      import matplotlib.pyplot as plt
      from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

      fig, [[ax_costs, ax_Lambda], [ax_test, ax_U] ] = plt.subplots(2, 2)
      
      Axes = [[ax_costs, ax_Lambda], [ax_test, ax_U]]
      if self.oc.N == 2 and self.oc.M == 1:        
        cax = make_axes_locatable(ax_test).append_axes("right", size="5%", pad="2%")
      fig.tight_layout()
    
    
    if not os.path.exists(dirName) and comm.rank == 0:
      print('Creating dir ' + dirName + '...')
      os.makedirs(dirName)

    # general stats
    # g_stats(iter i, iter k, quantity id, mpi rank)      
    # 0 integrations : time
    # 1 maximization : total time 
    # 2 maximization : total number of bfgs iterations
    # 3 maximization : number of layers treated
    # 4 cost         : cost err
    # 5              : mu_k     
    # 6 mesh_sizes of: X
    # 7 mesh_sizes of: P
    # 8 mesh_sizes of: U
    # 9 test cost    : generalization error
    g_stats = np.zeros((no_iters, self.kmax, 10, comm.size))

    if comm.rank == 0:
      test_inputs  = list()
      test_outputs = list()

    # Realization loop 
    for i in range(no_iters):
      
      g_stats[i, :, :, :] = self.train(settings, fig, Axes, cax)

      if comm.rank == 0:
        print('\n\n')
        print('=' * 100)
        print('={:^98}='.format('i='+str(i)+'/'+str(no_iters-1)))
        print('=' * 100)
        test_in, test_out = self.oc.plot_prediction(self)    
        
        test_inputs.append(np.array(test_in, ndmin = 2))
        test_outputs.append(np.array(test_out, ndmin=2))
    
    # Store data
    if comm.rank == 0:
      print('=' * 100)
      print('={:^98}='.format('Saving data in '+dirName))
      print('=' * 100)

      np.save(dirName + 'data_X0.npy', self.oc.X0)
      np.save(dirName + 'data_XT.npy', self.oc.XT)      
      
      np.save(dirName + 'controls.npy',  self.lowest_u)
      np.save(dirName + 'gen_stats.npy', g_stats)
      np.save(dirName + 'settings.npy',  settings)
      np.save(dirName + 'test_in.npy',  np.stack(test_inputs[0])) # is 2 dimensional 
      np.save(dirName + 'test_out.npy', np.stack(test_outputs)) # is 3 dimensional

      # data      
      np.save(dirName + 'allX.npy',  self.all_X)
      np.save(dirName + 'allXp.npy', self.all_Xp)
      np.save(dirName + 'allP.npy',  self.all_P)
      np.save(dirName + 'allPp.npy', self.all_Pp)

      # meshes
      np.save(dirName + 'mesh_U.npy', self.mesh_U)
      np.save(dirName + 'mesh_X.npy', self.mesh_X)
      np.save(dirName + 'mesh_P.npy', self.mesh_P)      
      print('\n\n')
