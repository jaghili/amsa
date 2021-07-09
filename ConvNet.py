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

      _a = np.array(np.linspace(0, 1, self.d_img) < 0.5, dtype=np.float32)
      _b = np.array(np.linspace(0, 1, self.d_img) > 0.5, dtype=np.float32)

      up_left = tc.from_numpy(np.outer(_a,_a))
      up_right = tc.from_numpy(np.outer(_a,_b))
      down_left = tc.from_numpy(np.outer(_b,_a))
      down_right = tc.from_numpy(np.outer(_b,_b))      
      
      W = u[:self.dk*self.dk].reshape(self.dk, self.dk)
      b = u[self.dk*self.dk]  *up_left   \
        + u[self.dk*self.dk+1]*up_right  \
        + u[self.dk*self.dk+2]*down_left \
        + u[self.dk*self.dk+3]*down_right
      return W, b
    
    self.reshaper = reshaper

    def f(t, X, u):
      """
      Dynamic X'(t) = f(t,X,u)
      - Assume u is a vector of size sizeu
      - Assume X is a matrix of size [K, d]
      - returns a vector of size [K, d]

      """
      W, b = self.reshaper(u)
      K, d = X.shape
      dimg = np.int(np.sqrt(d))
      
      _X = X.reshape(K, 1, dimg, dimg)
      _W = W.reshape(1, 1, self.dk,    self.dk)
      _b = b.reshape(1, 1, dimg, dimg)

      # conv2d([batch, no channels (input params), size x, size y],  [1, 1, dk, dk] )
      res = tc.nn.functional.conv2d(_X, _W, padding=1)
      
      return tc.tanh(res + _b).reshape(K, d)

    self.f = f

    
    # Lagrangian
    self.L = lambda X, u, u0 : self.eta * 0.5 * tc.inner(u - u0,u - u0)
    self.g   = lambda batch_X : tc.sum(batch_X, axis=1) / self.d  # batch_X is K x d, g(X) is K,
    self.dxg = lambda batch_X : tc.ones(batch_X.shape)  / self.d  # dxg is K x d full of 1/d
    
    # Cost
    self.phi_expr = str('0.5 * (sum(X^j(T)) - XT^j)')
    self.phi = lambda batch_X, XT : (self.g(batch_X) - XT)**2     # size K
    self.sum_phi = lambda batch_X, XT : tc.mean(self.phi(batch_X, XT))
    self.dxphi = lambda batch_X, XT : 2 * self.dxg(batch_X) * (self.g(batch_X) - XT).unsqueeze(0).T
