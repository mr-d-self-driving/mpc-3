import casadi as cd
import os

ipopt_solver = 'ma57'
solve_method = 'rk4'

gen_compiled = True
use_compiled = True

prefix = '_'.join(['mpcc', ipopt_solver, solve_method])
if use_compiled: prefix += '_compiled'

curr_path = os.path.dirname(os.path.dirname(__file__)) # sorta hacky
out_path = os.path.join(curr_path, 'out_mpcc')
os.makedirs(out_path, exist_ok=True)
os.makedirs(os.path.join(curr_path, 'out_mpcc', 'log'), exist_ok=True)
os.makedirs(os.path.join(curr_path, 'out_mpcc', 'time'), exist_ok=True)
os.makedirs(os.path.join(curr_path, 'out_mpcc', 'eval'), exist_ok=True)

out_log_file = os.path.join(out_path, 'log', '_'.join([prefix, 'out.txt']))

log_time = True
time_csv = os.path.join(out_path, 'time', '_'.join([prefix, 'time.csv']))

anim_save_file = os.path.join(out_path, prefix +'.gif')

pred_csv = os.path.join(out_path, 'pred.csv')
true_csv = os.path.join(out_path, 'true.csv')

T = 10. # Time horizon
N = 40  # number of control intervals
inter_axle = 0.5   # inter-axle distance

ts = .08 # time-step
e = 0.1 # epsilon (value for when solving stops)

# 5th-order
curve_1 = {'init_ts': [0, 0, cd.pi/3, 0, 0, 0],
           'xpts': [0, .5, 2, 3.3],
           'ypts': [0, 1, 3, 2],
           'order': 5}
          
curve_2 = {'init_ts': [3.29, 2.09, -1.66, 0, 0, 0],
           'xpts': [3.3, 2.7, 2, 3],
           'ypts': [2, .5, -1, -2],
           'order': 5}

curve_3 = {'init_ts': [3, -2, 0, 0, 0, 0],
           'xpts': [3, 3.5, 4, 3.5, 1],
           'ypts': [-2, -2.5, -3.5, -4.5, -4.5],
           'order': 5}

curve_4 = {'init_ts': [1, -4.5, 3*cd.pi/4, 0, 0, 0],
           'xpts': [1, 0, -1.5, -3, -2.5],
           'ypts': [-4.5, -3.5, -2.5, -1.5, .5],
           'order': 5}

curve_5 = {'init_ts': [-2.5, .5, cd.pi/4, 0, 0, 0],
           'xpts': [-2.5, -1.5, 0, 3, 2, 0],
           'ypts': [.5, 1.25, 2, 0, -1, 0],
           'order': 5}

curves_lst = [curve_1]
num_targets_final = len(curves_lst)