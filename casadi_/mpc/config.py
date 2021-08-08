import os
import casadi as cd
from casadi.casadi import solve

ipopt_solver = 'ma57'
solve_method = 'rk4'

gen_compiled = False
use_compiled = False

prefix = '_'.join(['mpc', ipopt_solver, solve_method])
if use_compiled: prefix += '_compiled'

curr_path = os.path.dirname(os.path.dirname(__file__)) # sorta hacky
out_path = os.path.join(curr_path, 'out')
os.makedirs(out_path, exist_ok=True)
os.makedirs(os.path.join(curr_path, 'out', 'log'), exist_ok=True)
os.makedirs(os.path.join(curr_path, 'out', 'time'), exist_ok=True)
os.makedirs(os.path.join(curr_path, 'out', 'eval'), exist_ok=True)

out_log_file = os.path.join(out_path, 'log', '_'.join([prefix, 'out.txt']))

log_time = True
time_csv = os.path.join(out_path, 'time', '_'.join([prefix, 'time.csv']))

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