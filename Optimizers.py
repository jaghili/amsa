import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

class Optimizer:
  """
  Abstract class
  """
  def __init__(self):
    self.fun      = None # function to minimize
    self.jac      = None # jacobian
    self.d        = None # vector size
    self.N        = None # pop size
    self.bounds   = None # bounds of each components
    self.x0       = None # start vector
    self.r        = None # relaxation coefficient


# class BFGS(Optimizer):

#   def __init__(self, f, bounds, x0, showInfos = False, pop_size = 500, relaxation = 1e-3, jac = None):

#     super().__init__()
    
#     self.d = x0.size
#     self.x0 = x0        # must be of size d
#     self.f  = f   # f acts on (d, *) matrices
#     self.showInfos  = showInfos
#     self.N = pop_size
#     self.bounds = bounds
#     self.r = relaxation
#     self.jac = jac    


#   def train(self, tol_err):

#     xmin, xmax = self.bounds[0], self.bounds[1]
#     d = self.d
#     N = self.N
#     focus = False
#     dfmax = 1e-6 # to detect whenever points are stuck in minima's
#     n_pick = 5
    
#     return x_opt, err, k


#
# Descente de gradient classique
#

  
class GradientDescent(Optimizer):

  def __init__(self, f, bounds, x0, showInfos = False, pop_size = 500, relaxation = 1e-3, jac = None):
    """
    f must act on (d, *) matrices !
    """
    
    super().__init__()
    
    self.d = x0.size
    self.x0 = x0        # must be of size d
    self.f  = f   # f acts on (d, *) matrices
    self.showInfos  = showInfos
    self.N = pop_size
    self.bounds = bounds
    self.r = relaxation
    self.jac = jac
    
    if self.showInfos:
      print('='*60)
      print('Init GD-GA ...')
      print('dim \t{:d}'.format(self.d))
      print('pop_size  \t {:d}'.format(self.N))
      print('relaxatin param  \t {:.2E}'.format(self.r))
      print('bounds  \t {}'.format(bounds))
      print('='*60)

  def ComputeApproxGradient(self, U, eps = 1e-6):
    d = self.d
    N = self.N
    
    tU     = (U.T.reshape(N, 1, d) * cp.ones((N, d, d))).reshape(N * d, d).T
    tU_eps = tU + eps * (cp.eye(d, d) * cp.ones((N, d, d))).reshape(N * d, d).T      
    return ((self.f(tU_eps) - self.f(tU))/eps).reshape(N, d).T


  
  def train(self, tol_err):
    """
    Give a tolerance error > 0
    returns optimum points + informations
    """
    assert tol_err > 0.
    
    xmin, xmax = self.bounds[0], self.bounds[1]
    d = self.d
    N = self.N
    focus = False
    dfmax = 1e-6 # to detect whenever points are stuck in minima's
    n_pick = 5
    
    # First guess    
    if self.showInfos:
      print('k\t errmin\t errmax\t df\t relax')
      print('='*60)

    k = 0
    U = xmin + (xmax-xmin) * cp.random.rand(d, N)
    if self.x0 is not None:
      U[:,0] = self.x0
    fU = self.f(U).get()
    errmin = np.min(fU)
    errmax = np.max(fU)
    fUold = fU 
    df = np.max(np.abs(fU))
    if self.showInfos:
      print('{:d}\t {:.2E}\t {:.2E}\t {:.2E}'.format(k, errmin, errmax, df, self.r))
    
    # descent loop
    plt.figure(1)
    plt.ion()
    while errmin > tol_err:

      k += 1
      
      # Compute/approx new dir vector
      if self.jac is None:
        dU = self.ComputeApproxGradient(U)
      else:
        dU = self.jac(U)

      # update & error 
      U -= self.r * dU
      fU = self.f(U).get()
      errmin = np.min(fU)
      errmax = np.max(fU)

      if focus is not True:
        df = np.max(np.abs(fU - fUold))

      if df < dfmax and focus is not True:
        if self.showInfos:
          print('*** focus: pick {:d} best'.format(n_pick))
        reordered = fU.argsort()
        U = U[:, reordered[:n_pick]]
        focus = True

      if df < dfmax:
        self.r /= 1.01

      fUold = fU

      if k % 100 == 0:
        if self.showInfos:
          print('{:d}\t {:.2E}\t {:.2E}\t {:.2E}\t {:.2E}'.format(k, errmin, errmax, df, self.r))
          plt.cla()
          # plt.xlim(xmin, xmax)
          # plt.ylim(xmin, xmax)
          plt.plot(U[0,:].get(), U[1,:].get(), 'x')
          plt.pause(0.000001)
          plt.draw()
      

    # converged
    plt.ioff()
    reordered = self.f(U).argsort()
    opt = U[:, reordered[0]]
    err = self.f(cp.array(opt, ndmin=2).T).get()
    if self.showInfos:
      print('{:d}\t {:.2E}'.format(k, err.squeeze()))
      plt.cla()
      # plt.xlim(xmin, xmax)
      # plt.ylim(xmin, xmax)      
      plt.plot(U[0,reordered[:n_pick]].get(), U[1,reordered[:n_pick]].get(), 'x')
      plt.pause(0.000001)
      plt.draw()

    # done
    return  opt, err, k







#
# Genetic Algorithm
#

class GeneticAlgorithm(Optimizer):

  def __init__(self,
               f,
               adn_size,
               bounds,
               showInfos = False,
               no_species = 100,
               tol = 1e-5):
    super().__init__()
    
    self.fitness     = f
    self.tol         = tol
    self.K           = no_species
    self.d           = adn_size
    self.bounds      = bounds    
    self.species     = cp.zeros((self.d, self.K))
    self.new_species = cp.zeros((self.d, self.K))
    self.showInfos   = showInfos
    
    # checks  
    assert self.K > 15

    if self.showInfos:
      print('='*60)
      print('Init GPU-GA')
      print('dim \t{:d}'.format(self.d))
      print('no. samples = \t {:d}'.format(self.K))
      print('='*30)
    
    # init species
    xmin, xmax = self.bounds[0], self.bounds[1]
    self.species = xmin + (xmax-xmin) * cp.random.rand(self.d, self.K)

  # Operators
  def crossover(self, sp_a, sp_b, t = 0.5):
    """
    Cross over between two species
    assumes sp is a vector
    """
    tc = cp.floor(t * len(sp_a))
    cross_sp = cp.zeros(sp_a.shape)
    cross_sp[:tc] = sp_a[:tc]
    cross_sp[tc:] = sp_b[tc:]
    return cross_sp

  def mutate(self, sp, ampl):
    """
    Mutate one individual, 
    assumes sp is a vector
    """
    r = cp.random.randint(len(sp))
    sp[r] += ampl

    r = cp.random.randint(len(sp))
    sp[r] -= ampl
    
    return sp

  
  def natural_selection(self):
    """
    Arbitrary choices
    """
    xmin, xmax = self.bounds[0], self.bounds[1]

    self.new_species = xmin + (xmax-xmin) * cp.random.rand(self.d, self.K)
    orders = self.fitness(self.species).argsort() # arg sort species in increasing order
    
    self.new_species[:, 0] = self.species[:, orders[0]]
    self.new_species[:, 1] = self.crossover(self.species[:, orders[0]], self.species[:, orders[1]], cp.random.rand())
    self.new_species[:, 2] = self.crossover(self.species[:, orders[1]], self.species[:, orders[2]], cp.random.rand())
    self.new_species[:, 3] = self.species[:, orders[0]] + xmin +  (xmax-xmin) *cp.random.rand()
    self.new_species[:, 4] = self.mutate(self.species[:, orders[0]], 1e-1)
    self.new_species[:, 5] = cp.sum(self.species[:, orders[0]]) / self.d
    self.new_species[:, 6] = self.mutate(self.species[:, orders[0]], 1e-2)
    self.new_species[:, 7] = self.mutate(self.species[:, orders[0]], 1e-3)
    self.new_species[:, 8] = self.mutate(self.species[:, orders[0]], 1e-4)
    self.new_species[:, 9] = self.mutate(self.species[:, orders[0]], 1e-5)
    self.new_species[:, 10] = self.mutate(self.species[:, orders[0]], 1e-6)
    self.new_species[:, 11] = self.mutate(self.species[:, orders[0]], 1e-7)
    self.new_species[:, 12] = self.mutate(self.species[:, orders[0]], 1e-8)
    self.new_species[:, 13] = self.mutate(self.species[:, orders[0]], 1e-9)
    
    self.species = self.new_species
    
    return 0

  
  def train(self, max_iter):

    no_species_to_show = 5
    err_list = list()

    if self.showInfos:
      print('\n\n')
      print('=' * 15 * no_species_to_show)
      print(('{:^15}' * no_species_to_show).format(*[i for i in range(no_species_to_show)]))
      print('=' * 15 * no_species_to_show)

      
    for k in range(max_iter):                

      scores = self.fitness(self.species).get()

      # print
      if self.showInfos:
        print(('{:>15.4E}' * no_species_to_show).format(*[scores[i] for i in range(no_species_to_show)]))
        
      err_list.append(scores[0])

      self.natural_selection()
    
    return self.species[:,0], err_list
