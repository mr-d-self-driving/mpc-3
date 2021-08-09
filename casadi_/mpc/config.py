import os
import casadi as cd
from casadi.casadi import solve

ipopt_solver = 'ma57'
solve_method = 'colloc'

gen_compiled = False
use_compiled = True

prefix = '_'.join(['mpc', ipopt_solver, solve_method])
if use_compiled: prefix += '_compiled'

curr_path = os.path.dirname(os.path.dirname(__file__)) # sorta hacky
out_path = os.path.join(curr_path, 'out_mpc')
os.makedirs(out_path, exist_ok=True)
os.makedirs(os.path.join(curr_path, 'out_mpc', 'log'), exist_ok=True)
os.makedirs(os.path.join(curr_path, 'out_mpc', 'time'), exist_ok=True)
os.makedirs(os.path.join(curr_path, 'out_mpc', 'time_simple'), exist_ok=True)
os.makedirs(os.path.join(curr_path, 'out_mpc', 'eval'), exist_ok=True)

out_log_file = os.path.join(out_path, 'log', '_'.join([prefix, 'out.txt']))

log_simple_time = True
log_time = False
time_csv = os.path.join(out_path, 'time', '_'.join([prefix, 'time.csv']))
simple_time_csv = os.path.join(out_path, 'time_simple', '_'.join([prefix, 'simple_time.csv']))

anim_save_file = os.path.join(out_path, prefix +'.gif')

T = 10. # Time horizon
N = 40  # number of control intervals
inter_axle = 0.5   # inter-axle distance

ts = .08 # time-step
e = 0.07 # epsilon (value for when solving stops)

# init_ts = [2, 1, cd.pi/2, 0, 0]
# xf, yf = [0, -3, -2, 1, 2], [3, 0, -3, -2, 1]

init_ts = [0, 0, cd.pi/4, 0, 0]
xf, yf = [2], [3]

num_targets_final = len(xf)