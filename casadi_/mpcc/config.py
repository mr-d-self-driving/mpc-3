import casadi as cd
import os

ipopt_solver = 'mumps'
solve_method = 'rk4'

curr_path = os.path.dirname(os.path.dirname(__file__)) # sorta hacky
out_path = os.path.join(curr_path, 'out')
compiled_path = os.path.join(curr_path, 'compiled')

os.makedirs(out_path, exist_ok=True)
os.makedirs(compiled_path, exist_ok=True)

gen_compiled = False
use_compiled = False
compiled_path = os.path.join(compiled_path, 'nlp.so')

log_pred = False

anim_save_file = os.path.join(out_path, '1_casadi_mpcc_' + solve_method +'.gif')
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

curves_lst = [curve_1, curve_2, curve_3, curve_4, curve_5]
num_targets_final = len(curves_lst)