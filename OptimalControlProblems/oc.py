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
