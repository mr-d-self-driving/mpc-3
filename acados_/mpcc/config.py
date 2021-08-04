import os
import numpy as np

T = 10. # Time horizon
N = 40  # number of control intervals
D = 0.5

curr_path = os.path.dirname(os.path.dirname(__file__)) # sorta hacky
out_path = os.path.join(curr_path, 'out')
code_export_dir = os.path.join(curr_path, 'mpcc_acados_generated_code')
json_path = 'mpcc_acados_ocp.json'

os.makedirs(out_path, exist_ok=True)
os.makedirs(code_export_dir, exist_ok=True)

anim_save_file = os.path.join(out_path, 'acados_mpcc.gif')

ts = .04
e = 0.1

# 5th-order
curve_1 = {'init_ts': np.array([0, 0, np.pi/3, 0, 0, 0]),
           'xpts': [0, .5, 2, 3.3],
           'ypts': [0, 1, 3, 2],
           'order': 5}

curves_lst = [curve_1]
num_targets_final = len(curves_lst)