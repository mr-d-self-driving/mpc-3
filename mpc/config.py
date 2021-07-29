import os
import casadi as cd

from solvers.mpc_colloc import build_solver as mpc_colloc
from solvers.mpc_rk4 import build_solver as mpc_rk4

solver = mpc_rk4
ipopt_solver = 'mumps'
solve_method = 'rk4'

gen_compiled = False
use_compiled = False
compiled_path = '' # not using compiled

anim_save_file = os.path.join(os.getcwd(), 'out', 'mpc_' + solve_method +'.gif')

T = 10. # Time horizon
N = 40  # number of control intervals
inter_axle = 0.5   # inter-axle distance

ts = .04 # time-step
e = 0.1 # epsilon (value for when solving stops)

init_ts = [0, 0, cd.pi/4, 0, 0]
xt, yt = [2], [3]

num_targets_final = 1