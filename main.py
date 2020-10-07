import sys
from mpi4py import MPI
import argparse
from EMSA import *
from OptimalControlProblems import *

# Mpi init
comm = MPI.COMM_WORLD

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-tf',         help='Final time')
parser.add_argument('-eta',        help='Regularizer amplitude')
parser.add_argument('-n',          help='Number of time steps ')
parser.add_argument('-d',          help='ResNet input size d')
parser.add_argument('-K',          help='Number of sample K')
parser.add_argument('-navg',       help='Number of realizations to approximate expectations')
parser.add_argument('-kmax',       help='Maximum number or iterations on control')
parser.add_argument('-maxiter',    help='Maximum number of BFGS/Newton iterations')
parser.add_argument('-rho',        help='Penalization parameter of E-MSA rho')
parser.add_argument('-o',          help='Output directory')
parser.add_argument('-odesolver',  help='ODE solver')
parser.add_argument('-adapteach',  help='adaptivity: adapt each #')
parser.add_argument('-adaptdyadic',help='adaptivity: dyadic adapt at #')
parser.add_argument('-adaptauto',  help='adaptivity: adapt mesh automatically according to lambda')
parser.add_argument('-maxlayers',  help='adaptivity: maximum number of layers allowed')
parser.add_argument('-minlayers',  help='adaptivity: min number of layers')
parser.add_argument('-addlayers',  help='adaptivity: Number of layers to add at each refinement')
parser.add_argument('-adaptrho',   help='adaptivity: Adapt rho automatically')
parser.add_argument('-problem',    help='select problem: simplest | oscillator | knownu | sinus | step | classif ')
args = parser.parse_args()


# Casting arguments and default values

T            = float(args.tf) if args.tf is not None                   else 5.0                   # Final time 
time_steps   = int(args.n) if args.n          is not None              else 4                     # Initial number of time steps
kmax         = int(args.kmax) if args.kmax is not None                 else 300                   # Max number of training iterations
res_dir      = str(args.o) if args.o is not None                       else './results/default/'  # Output directory
rho          = float(args.rho) if args.rho is not None                 else 2.0                   # E-MSA penalization parameter 
maxiter      = int(args.maxiter) if args.maxiter is not None           else 10                    # Maximum number of BFGS iterations
odesolver    = str(args.odesolver) if args.odesolver is not None       else 'euler'               # ODE solver

# adapt settings
adapt_max_layers = int(args.maxlayers) if args.maxlayers is not None   else 128
adapt_add_layers = int(args.addlayers) if args.addlayers is not None   else 16

# strategy
adapt_auto   = int(args.adaptauto) if args.adaptauto is not None       else 0                     # enable/disable automatic adaptation
adapt_each   = int(args.adapteach) if args.adapteach is not None       else 10                    # add no_add layers each # iterations
adapt_dyadic = int(args.adaptdyadic) if args.adaptdyadic is not None   else 0                     # adapt dyadically # steps
adapt_rho    = int(args.adaptrho) if args.adaptrho is not None         else 0                     # adapt rho

# for 2D oscillator and neural net
eta          = float(args.eta) if args.eta is not None                 else 0.0                   # Lagrangian scaling
abn_mul      = 1.0  # abnormal multiplier

# for neural net only
d            = int(args.d) if args.d is not None                       else 3                     # layer size
K            = int(args.K) if args.K        is not None                else 50                    # 
no_avg_iters = int(args.navg) if args.navg is not None                 else 2                     # Repeat training 

prb          = str(args.problem) if args.problem is not None           else 'sinus'               # Problem class


# Check
assert d > 0 and  K > 0
assert T > 0. and time_steps > 1
assert kmax > 0 and eta >= 0


# Init and train
settings = {
  'adapt_each'           : adapt_each,
  'adapt_auto'           : adapt_auto,
  'adapt_max_nodes'      : adapt_max_layers,
  'adapt_add_layers'     : adapt_add_layers,
  'adapt_dyadic'         : adapt_dyadic,
  'adapt_rho'            : adapt_rho,
  'show_plots'           : True,
  'optimizer'            : 'bfgs',
  'ode_integrator'       : odesolver,
  'bfgs_maxiter'         : maxiter,
  'cmaes_maxiter'        : maxiter,
  'output_dir'           : res_dir
}


if comm.rank == 0:
  print('\n------------------------------------------------------------')
  print('{:50} {}'.format('Output directory',        settings['output_dir']))
  print('{:50} {:.1f}'.format('final time',          T))
  print('{:50} {:d}'.format('mesh size X',           time_steps))
  print('{:50} {:d}'.format('(NN) d',                d))
  print('{:50} {:d}'.format('(NN) no. samples',      K))
  print('{:50} {:.2f}'.format('EMSA rho',            rho))
  print('{:50} {:d}'.format('EMSA kmax',             kmax))
  print('{:50} {}'.format('Optimizer',               settings['optimizer']))
  print('{:50} {}'.format('Adapt rho: ',             True if settings['adapt_rho'] > 0 else False))
  print('{:50} {}'.format('Mesh adapt: ponctual',    settings['adapt_each']))
  print('{:50} {}'.format('Mesh adapt: auto',        True if settings['adapt_auto'] > 0 else False))
  print('{:50} {}'.format('Mesh adapt: max',         settings['adapt_max_nodes']))
  print('{:50} {}'.format('Mesh adapt: add size',    settings['adapt_add_layers']))  
  print('{:50} {:.2f}'.format('(NN+2D) eta',         eta))
  print('{:50} {:.2f}'.format('EMSA abnormal mul',   abn_mul))   
  print('--------------------------------------------------------------')



# Data and Problem
#------------------------------------------------#

if settings['show_plots']:
  import matplotlib.pyplot as plt

  
# Problems
prob = None

if prb == 'onedim':
  prob = OnedimProblem(T)
  
elif prb == 'oscillator':
  prob = OscillatorProblem(T)

# probleme 1d connu
elif prb == 'knownu':
  xbounds = [-1.0, 1.0]
  no_layers = settings['adapt_max_nodes']
  prob = NN_1d_exact(d, K, T, xbounds, no_layers) # generate data with a kown control with 32 euler steps

elif prb == 'sinus':
  prob = NN_func1d(d, K, T, lambda x : np.sin(x), [-np.pi, np.pi], 0.0)  # données non bruités

elif prb == 'step':
  prob = NN_func1d(d, K, T, lambda x : 0.5 if x < 0 else -0.5, [-1.0, 1.0], 0.2)

elif prb == 'classif':
  prob = NN_2d_classif(d, K, T, lambda x,y : 1.0 if x**2+y**2 < 0.25 else 0.0, [-1.0, 1.0], [-1.0, 1.0])  

else:
  raise Exception('Problem not known, terminating.')


prob.eta = 0.0
algo = EMSA(prob, kmax, comm, settings, abn_mul, rho, time_steps)
algo.error_study(no_avg_iters, settings)
