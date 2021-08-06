import os
import casadi as cd

ipopt_solver = 'mumps'
solve_method = 'rk4'

curr_path = os.path.dirname(os.path.dirname(__file__)) # sorta hacky
out_path = os.path.join(curr_path, 'out')
os.makedirs(out_path, exist_ok=True)

gen_compiled = False
use_compiled = False
compiled_path = '' # not using compiled

anim_save_file = os.path.join(out_path, 'casadi_mpc_' + solve_method +'.gif')

T = 10. # Time horizon
N = 40  # number of control intervals
inter_axle = 0.5   # inter-axle distance

ts = .033 # time-step
e = 0.07 # epsilon (value for when solving stops)

init_ts = [2, 1, cd.pi/2, 0, 0]
xf, yf = [0, -3, -2, 1, 2], [3, 0, -3, -2, 1]

num_targets_final = len(xf)