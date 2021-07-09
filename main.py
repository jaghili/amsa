import torch as tc
import matplotlib.pyplot as plt

from Onedim import BangBang
from Twodim import Oscillator
from Mesh import Mesh
from ocp import Function
from dualize import Dualize

tc.autograd.set_detect_anomaly(True) # debug

# Final time
Tf = 2.0 

# Mesh
mesh = Mesh(0.0, Tf, n_points=20)

# Primal problem
primal_pb = BangBang(Tf)
#primal_pb = Oscillator(Tf)

# Dual problem
dual_pb = Dualize(primal_pb)

# Solve
u0 = tc.randn(mesh.n, primal_pb.sizeu, 1)
#u, xu = primal_pb.solve(mesh, u0=u0)
u, xu  = dual_pb.solve(mesh, u0=u0, maxiter=10)
