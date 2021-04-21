import numpy as np
import torch as tc
from OptimalControlProblems.oc import OptimalControlProblem

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
    
    assert np.mod(d, 2) == 0 # ensure that image side is a multiple of 2
    assert T > 0.

    self.eta = 1.0
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

    # info
    self.info = 'T = {}\n'.format(self.d_img)
    self.info += 'dim vector X = {}\n'.format(self.d)
    self.info += 'no.sample K = {}\n'.format(K)
    self.info += 'size kernel dk = {}'.format(self.dk)

    print('Init ConvNet:')
    print('--------------------------------:')
    print(self.info)
                                            



    def shaper(W, b): # Assume that W and b are torch tensors
      """
      Convert: W,b ----> u
      """
      n, m = W.shape
      assert n == self.dk
      assert m == self.dk

      n, m = b.shape
      assert n == self.d_img
      assert m == self.d_img

      u = tc.zeros(self.sizeu)
      u[:self.dk*self.dk] = W.reshape(self.dk*self.dk)
      u[self.dk*self.dk:] = tc.tensor([b[0,0], b[0,-1],b[-1,0],b[-1,-1]])

      return u

    self.shaper = shaper
    
    def reshaper(u):
      """
      Convert:  u  ---->   W, b
      - u is a torch vector
      - W,b are torch tensors 
      """
      _a = tc.tensor(tc.linspace(0, 1, self.d_img) < 0.5)
      _b = tc.tensor(tc.linspace(0, 1, self.d_img) > 0.5)
      
      W = u[:self.dk*self.dk].reshape(self.dk, self.dk)
      b = u[self.dk*self.dk]  *tc.outer(_a, _a)   \
        + u[self.dk*self.dk+1]*tc.outer(_a, _b)   \
        + u[self.dk*self.dk+2]*tc.outer(_b, _a)   \
        + u[self.dk*self.dk+3]*tc.outer(_b, _b)
      return W, b
    
    self.reshaper = reshaper

    def f(t, X, u):
      """
      Dynamic X'(t) = f(t,X,u)
      - Assume X is a matrix of size [K, d]
      - returns a vector of size [K, d]

      """
      W, b = self.reshaper(u) 
      
      _X = X.reshape(self.K, 1, self.d_img, self.d_img)
      _W = W.reshape(1,      1, self.dk,    self.dk)
      _b = b.reshape(1,      1, self.d_img, self.d_img)

      # conv2d([batch, no channels (input params), size x, size y],  [1, 1, dk, dk] )
      res = tc.nn.functional.conv2d(_X, _W, padding=1)
      
      return tc.tanh(res + _b).reshape(self.K, self.d)

    self.f = f

    # TODO: dxf
    
    # Lagrangian
    self.L = lambda X, u, u0 : self.eta * 0.5 * tc.inner(u - u0,u - u0)
    self.g   = lambda all_X : tc.sum(all_X, axis=0) / self.d
    
    # Cost
    self.phi_expr = str('0.5 * (sum(X^j(T)) - XT^j)')
    self.phi = lambda all_X, XT : (self.g(all_X) - XT[0,:])**2
    self.sum_pi = lambda all_X, XT : tc.mean(self.phi(all_X, XT))
    self.dxphi = lambda all_X, XT : 2 * self.dxg(all_X) * ( self.g(all_X) - XT[0, :])
